//! # rig-budget-guard
//!
//! Token budget enforcement middleware for [Rig](https://github.com/0xPlaygrounds/rig)-based
//! LLM applications.
//!
//! `RigBudgetGuard` wraps any Rig [`Agent`](rig_core::agent::Agent) and intercepts
//! [`prompt`](rig_core::completion::request::Prompt), [`chat`](rig_core::completion::request::Chat),
//! and [`completion`](rig_core::completion::request::Completion) calls to track token spending
//! against per-model budgets using
//! [`ConservationChecker`](conservation_checker::ConservationChecker).
//!
//! # Quick start
//!
//! ```rust,ignore
//! use rig_budget_guard::RigBudgetGuard;
//! // Wrap any Rig agent:
//! // let guarded = RigBudgetGuard::new(agent)
//! //     .with_budget("gpt-4o", 100_000)
//! //     .with_warn_threshold(0.8);
//! // let response = guarded.prompt("Hello!").await?;
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                    Your App                          │
//! │  prompt("...") / chat("...", ...) / completion(...) │
//! └──────────────┬──────────────────────────────────────┘
//!                │
//! ┌──────────────▼──────────────────────────────────────┐
//! │              RigBudgetGuard<M>                      │
//! │  ┌──────────────────────────────────────────────┐   │
//! │  │  ConservationChecker: token budget tracking  │   │
//! │  │  Phase detection (Pre→Transition→Post)       │   │
//! │  │  Per-model limits & Serde snapshots          │   │
//! │  └──────────────────────────────────────────────┘   │
//! │  ┌──────────────────────────────────────────────┐   │
//! │  │  Inner Agent<M> (delegated)                  │   │
//! │  └──────────────────────────────────────────────┘   │
//! └──────────────┬──────────────────────────────────────┘
//!                │
//! ┌──────────────▼──────────────────────────────────────┐
//! │              LLM Provider (OpenAI, etc.)            │
//! └─────────────────────────────────────────────────────┘
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use chrono::Utc;
use conservation_checker::{ConservationChecker, Phase};
use rig_core::completion::message::Message;
use rig_core::completion::request::CompletionError;
use rig_core::completion::{
    CompletionModel,
    message::UserContent,
    request::{
        Chat, Completion, CompletionRequestBuilder, Prompt, PromptError,
    },
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// Phase is re-exported via the initial `use conservation_checker::{ConservationChecker, Phase};` line

// ── Public types ──────────────────────────────────────────────────────

/// Per-model budget configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBudget {
    /// Maximum total tokens (input + output) budgeted for this model.
    pub max_tokens: f64,
    /// Current budget spent so far.
    pub spent_tokens: f64,
}

impl ModelBudget {
    /// Create a new budget with a max token limit.
    pub fn new(max_tokens: u64) -> Self {
        Self {
            max_tokens: max_tokens as f64,
            spent_tokens: 0.0,
        }
    }

    /// Remaining tokens in this budget.
    pub fn remaining(&self) -> f64 {
        (self.max_tokens - self.spent_tokens).max(0.0)
    }

    /// Fraction of budget consumed (0.0 = none, 1.0 = exhausted).
    pub fn usage_ratio(&self) -> f64 {
        if self.max_tokens <= 0.0 {
            return 1.0;
        }
        (self.spent_tokens / self.max_tokens).clamp(0.0, 1.0)
    }
}

/// A single recorded audit event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetEvent {
    /// Unique event identifier.
    pub id: String,
    /// ISO-8601 timestamp.
    pub timestamp: String,
    /// Model name this event relates to.
    pub model_name: String,
    /// Tokens consumed in this event.
    pub tokens_consumed: u64,
    /// Total spent so far.
    pub total_spent: f64,
    /// Budget limit for this model.
    pub budget_limit: f64,
    /// Phase at the time of the event.
    pub phase: String,
    /// Event type label.
    pub event_type: String,
    /// Optional prompt snippet (truncated to 100 chars).
    pub prompt_preview: Option<String>,
}

/// Serde-serializable budget guard snapshot for audit trails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetSnapshot {
    /// Snapshot identifier.
    pub snapshot_id: String,
    /// ISO-8601 timestamp.
    pub timestamp: String,
    /// Per-model budgets.
    pub budgets: HashMap<String, ModelBudget>,
    /// Phase for each model.
    pub phases: HashMap<String, String>,
    /// Recent events.
    pub events: Vec<BudgetEvent>,
    /// Models exceeding the warning threshold.
    pub warnings_active: Vec<String>,
}

/// Warnings emitted by the budget guard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetWarning {
    /// The model this warning applies to.
    pub model: String,
    /// Human-readable message.
    pub message: String,
    /// Current usage ratio.
    pub usage_ratio: f64,
    /// Detected phase (stored as string for serde compatibility).
    pub phase: String,
}

impl BudgetWarning {
    fn new(model: String, message: String, usage_ratio: f64, phase: &Phase) -> Self {
        Self {
            model,
            message,
            usage_ratio,
            phase: format!("{:?}", phase),
        }
    }
}

/// Configuration for token-spending phase thresholds.
///
/// | Phase | Meaning | Default threshold |
/// |---|---|---|
/// | PreTransition | Spending is accelerating — early warning | usage > 60% |
/// | Transitioning | Critically close to limit | usage > 85% |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseConfig {
    /// Usage ratio threshold for PreTransition warning (default: 0.6).
    pub pret_transition_threshold: f64,
    /// Usage ratio threshold for Transitioning/critical (default: 0.85).
    pub transitioning_threshold: f64,
}

impl Default for PhaseConfig {
    fn default() -> Self {
        Self {
            pret_transition_threshold: 0.6,
            transitioning_threshold: 0.85,
        }
    }
}

// ── Internal state ────────────────────────────────────────────────────

/// Internal shared mutable state behind Arc<Mutex<>>.
#[derive(Debug, Clone)]
struct SharedState {
    budgets: HashMap<String, ModelBudget>,
    model_names: Vec<String>,
    checker: ConservationChecker,
    events: Vec<BudgetEvent>,
}

// ── The Guard ─────────────────────────────────────────────────────────

/// Token budget enforcement middleware for Rig agents.
///
/// Wraps any agent using [`CompletionModel`] and intercepts `prompt`, `chat`,
/// and `completion` calls to track token consumption. Uses
/// [`ConservationChecker`] for phase detection.
///
/// All mutable state lives behind [`Arc`]`<`[`Mutex`]`<`[`SharedState`]`>>` so
/// that the Rig trait methods (which take `&self`) can mutate budgets.
///
/// # Type parameters
///
/// * `M` — The [`CompletionModel`] type used by the wrapped agent.
#[derive(Clone)]
pub struct RigBudgetGuard<M: CompletionModel> {
    /// Inner agent that receives the actual LLM calls.
    inner: Arc<rig_core::agent::Agent<M>>,
    /// Shared mutable state (budgets, checker, events).
    state: Arc<Mutex<SharedState>>,
    /// Warning threshold (0.0–1.0) as fraction of budget.
    warn_threshold: f64,
    /// Phase configuration.
    phase_config: PhaseConfig,
    /// Optional instance name.
    name: String,
    /// Model name override (if not using CompletionModel::model_name).
    pub model_name: String,
}

// ── Construction & builder API ────────────────────────────────────────

impl<M: CompletionModel> RigBudgetGuard<M> {
    /// Create a new budget guard wrapping an existing agent.
    ///
    /// No budgets are configured initially; use [`with_budget`](Self::with_budget)
    /// or [`with_budgets`](Self::with_budgets) to add them.
    pub fn new(agent: rig_core::agent::Agent<M>) -> Self {
        Self {
            inner: Arc::new(agent),
            state: Arc::new(Mutex::new(SharedState {
                budgets: HashMap::new(),
                model_names: Vec::new(),
                checker: ConservationChecker::new(),
                events: Vec::new(),
            })),
            warn_threshold: 0.8,
            phase_config: PhaseConfig::default(),
            name: "rig-budget-guard".to_string(),
            model_name: "default".to_string(),
        }
    }

    /// Assign a name to this guard instance (useful when running multiple guards).
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Add a budget for a specific model identified by string name.
    ///
    /// Also registers the quantity with the internal `ConservationChecker`
    /// (tolerance = 5% of max tokens).
    pub fn with_budget(mut self, model_name: &str, max_tokens: u64) -> Self {
        let name = model_name.to_string();
        let budget = ModelBudget::new(max_tokens);

        let mut state = self.state.lock().unwrap();
        state.budgets.insert(name.clone(), budget);
        state.model_names.push(name.clone());
        state.checker.register(&name, max_tokens as f64, max_tokens as f64 * 0.05);
        drop(state);
        self.model_name = name;
        self
    }

    /// Set budgets for multiple models at once.
    pub fn with_budgets(self, budgets: impl IntoIterator<Item = (&'static str, u64)>) -> Self {
        let mut g = self;
        for (model_name, max_tokens) in budgets {
            g = g.with_budget(model_name, max_tokens);
        }
        g
    }

    /// Set the warning threshold (0.0–1.0). Default: `0.8`.
    ///
    /// When the usage ratio exceeds this value, `prompt` and `chat` responses
    /// get a `[Budget: …% used • Phase: …]` header prepended.
    pub fn with_warn_threshold(mut self, threshold: f64) -> Self {
        self.warn_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Override the default phase configuration.
    pub fn with_phase_config(mut self, config: PhaseConfig) -> Self {
        self.phase_config = config;
        self
    }

    /// Return the inner agent by reference.
    pub fn inner(&self) -> &rig_core::agent::Agent<M> {
        &self.inner
    }

    /// Consume the guard and return the inner agent.
    pub fn into_inner(self) -> rig_core::agent::Agent<M> {
        Arc::into_inner(self.inner)
            .expect("RigBudgetGuard has more than one reference to inner agent")
    }
}

// ── Observability ─────────────────────────────────────────────────────

impl<M: CompletionModel> RigBudgetGuard<M> {
    /// Snapshot the current budget state (thread-safe clone of all data).
    pub fn snapshot(&self) -> BudgetSnapshot {
        let state = self.state.lock().unwrap();

        let phases: HashMap<String, String> = state
            .budgets
            .keys()
            .map(|name| {
                let phase = if state.checker.registered().contains(name) {
                    format!("{:?}", state.checker.phase(name))
                } else {
                    "Unknown".to_string()
                };
                (name.clone(), phase)
            })
            .collect();

        let warnings_active: Vec<String> = state
            .budgets
            .iter()
            .filter(|(_, b)| b.usage_ratio() >= self.warn_threshold)
            .map(|(name, _)| name.clone())
            .collect();

        BudgetSnapshot {
            snapshot_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now().to_rfc3339(),
            budgets: state.budgets.clone(),
            phases,
            events: state.events.clone(),
            warnings_active,
        }
    }

    /// Serialize the current budget state as a pretty-printed JSON string.
    pub fn snapshot_json(&self) -> String {
        serde_json::to_string_pretty(&self.snapshot()).unwrap_or_else(|_| "{}".to_string())
    }

    /// Get the current conservation phase for a model.
    ///
    /// Returns `None` if the model has no registered budget.
    pub fn phase(&self, model_name: &str) -> Option<Phase> {
        let state = self.state.lock().unwrap();
        if state.checker.registered().contains(&model_name.to_string()) {
            Some(state.checker.phase(model_name))
        } else {
            None
        }
    }

    /// Classify a usage ratio into a phase label based on thresholds alone
    /// (no history required, but less nuanced than the ConservationChecker).
    ///
    /// The mapping is:
    /// - `0.00–0.39` → [`Stable`](Phase::Stable)
    /// - `0.40–0.59` → [`Resolving`](Phase::Resolving)
    /// - `0.60–0.84` → [`PreTransition`](Phase::PreTransition)
    /// - `0.85–1.00` → [`Transitioning`](Phase::Transitioning)
    pub fn phase_from_ratio(ratio: f64) -> Phase {
        if ratio >= 0.85 {
            Phase::Transitioning
        } else if ratio >= 0.60 {
            Phase::PreTransition
        } else if ratio >= 0.40 {
            Phase::Resolving
        } else {
            Phase::Stable
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────

impl<M: CompletionModel> RigBudgetGuard<M> {
    fn record_event(
        &self,
        model_name: &str,
        tokens_consumed: u64,
        event_type: &str,
        prompt_preview: Option<String>,
    ) {
        let mut state = self.state.lock().unwrap();
        let budget = state.budgets.get(model_name).cloned();
        let phase = if state.checker.registered().contains(&model_name.to_string()) {
            Some(format!("{:?}", state.checker.phase(model_name)))
        } else {
            None
        };

        state.events.push(BudgetEvent {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now().to_rfc3339(),
            model_name: model_name.to_string(),
            tokens_consumed,
            total_spent: budget.as_ref().map(|b| b.spent_tokens).unwrap_or(0.0),
            budget_limit: budget.as_ref().map(|b| b.max_tokens).unwrap_or(0.0),
            phase: phase.unwrap_or_else(|| "Unknown".to_string()),
            event_type: event_type.to_string(),
            prompt_preview: prompt_preview.map(|p| {
                if p.len() > 100 {
                    format!("{}...", &p[..100])
                } else {
                    p
                }
            }),
        });

        // Cap at 1000 events to prevent unbounded memory growth.
        if state.events.len() > 1000 {
            let drain_to = state.events.len() - 1000;
            state.events.drain(..drain_to);
        }
    }

    /// Update budgets and ConservationChecker after a completion.
    fn track_usage(&self, model_name: &str, input_tokens: u64, output_tokens: u64) {
        let total = input_tokens + output_tokens;
        let mut state = self.state.lock().unwrap();

        // Check if registered first
        let is_registered = state.checker.registered().contains(&model_name.to_string());
        
        if let Some(budget) = state.budgets.get_mut(model_name) {
            budget.spent_tokens += total as f64;

            if is_registered {
                let remaining = (budget.max_tokens - budget.spent_tokens).max(0.0);
                state.checker.update(model_name, remaining);
            }
        }
        state.checker.snapshot();
    }

    fn check_warnings(&self) -> Vec<BudgetWarning> {
        let state = self.state.lock().unwrap();
        let mut warnings = Vec::new();
        for (model_name, budget) in &state.budgets {
            let ratio = budget.usage_ratio();
            let phase = if state.checker.registered().contains(model_name) {
                state.checker.phase(model_name)
            } else {
                Phase::Stable
            };

            if ratio >= self.phase_config.transitioning_threshold {
                warnings.push(BudgetWarning::new(
                    model_name.clone(),
                    format!(
                        "CRITICAL: Token budget at {:.1}% for model '{}'",
                        ratio * 100.0,
                        model_name
                    ),
                    ratio,
                    &phase,
                ));
            } else if ratio >= self.phase_config.pret_transition_threshold {
                warnings.push(BudgetWarning::new(
                    model_name.clone(),
                    format!(
                        "WARNING: Token budget at {:.1}% for model '{}'",
                        ratio * 100.0,
                        model_name
                    ),
                    ratio,
                    &phase,
                ));
            }
        }
        warnings
    }

    
}

// ── Trait implementations ─────────────────────────────────────────────



// ── Standalone helper functions ──────────────────────────────────────

/// Extract text from a Message for preview purposes.
pub(crate) fn extract_text(message: &Message) -> String {
        match message {
            Message::System { content } => content.clone(),
            Message::User { content } => {
                for item in content.iter() {
                    if let UserContent::Text(t) = item {
                        return t.text.clone();
                    }
                }
                String::new()
            }
            Message::Assistant { content, .. } => {
                for item in content.iter() {
                    if let rig_core::completion::message::AssistantContent::Text(t) = item {
                        return t.text.clone();
                    }
                }
                String::new()
            }
        }
    }

fn estimate_tokens(text: &str) -> u64 {
    (text.len() / 4).max(1) as u64
}

impl<M: CompletionModel + 'static> Prompt for RigBudgetGuard<M> {
    #[allow(refining_impl_trait_reachable)]
    async fn prompt(&self, prompt: impl Into<Message> + Send) -> Result<String, PromptError> {
        let message: Message = prompt.into();
        let preview = extract_text(&message);
        let model_name = self.model_name.clone();

        let warnings = self.check_warnings();
        for w in &warnings {
            eprintln!("[rig-budget-guard] WARNING: {}", w.message);
        }

        let result = self.inner.prompt(message.clone()).await;

        match &result {
            Ok(response) => {
                let input_est = estimate_tokens(&preview);
                let output_est = estimate_tokens(response);
                let estimated_total = input_est + output_est;

                self.track_usage(&model_name, input_est, output_est);
                self.record_event(&model_name, estimated_total, "prompt", Some(preview));

                let over_threshold = {
                    let s = self.state.lock().unwrap();
                    s.budgets
                        .get(&model_name)
                        .is_some_and(|b| b.usage_ratio() >= self.warn_threshold)
                };

                if over_threshold {
                    let ratio = {
                        let s = self.state.lock().unwrap();
                        s.budgets
                            .get(&model_name)
                            .map(|b| b.usage_ratio())
                            .unwrap_or(0.0)
                    };
                    let post_phase = self.phase(&model_name);
                    let phase_label = post_phase.unwrap_or(Phase::Stable);
                    let header = format!(
                        "[Budget: {:.0}% used • Phase: {:?}]\n",
                        ratio * 100.0,
                        phase_label
                    );
                    return Ok(format!("{}{}", header, response));
                }
            }
            Err(_) => {
                self.record_event(&model_name, 0, "prompt_error", Some(preview));
            }
        }

        result
    }
}

impl<M: CompletionModel + 'static> Chat for RigBudgetGuard<M> {
    async fn chat(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: &mut Vec<Message>,
    ) -> Result<String, PromptError> {
        let message: Message = prompt.into();
        let preview = extract_text(&message);
        let model_name = self.model_name.clone();

        let warnings = self.check_warnings();
        for w in &warnings {
            eprintln!("[rig-budget-guard] WARNING: {}", w.message);
        }

        let result = self.inner.chat(message.clone(), chat_history).await;

        match &result {
            Ok(response) => {
                let input_est = estimate_tokens(&preview);
                let output_est = estimate_tokens(response);
                let estimated_total = input_est + output_est;

                self.track_usage(&model_name, input_est, output_est);
                self.record_event(&model_name, estimated_total, "chat", Some(preview));

                let over_threshold = {
                    let s = self.state.lock().unwrap();
                    s.budgets
                        .get(&model_name)
                        .is_some_and(|b| b.usage_ratio() >= self.warn_threshold)
                };

                if over_threshold {
                    let ratio = {
                        let s = self.state.lock().unwrap();
                        s.budgets
                            .get(&model_name)
                            .map(|b| b.usage_ratio())
                            .unwrap_or(0.0)
                    };
                    let post_phase = self.phase(&model_name);
                    let phase_label = post_phase.unwrap_or(Phase::Stable);
                    let header = format!(
                        "[Budget: {:.0}% used • Phase: {:?}]\n",
                        ratio * 100.0,
                        phase_label
                    );
                    return Ok(format!("{}{}", header, response));
                }
            }
            Err(_) => {
                self.record_event(&model_name, 0, "chat_error", Some(preview));
            }
        }

        result
    }
}

impl<M: CompletionModel + 'static> Completion<M> for RigBudgetGuard<M> {
    async fn completion<I, T>(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: I,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError>
    where
        I: IntoIterator<Item = T> + Send,
        T: Into<Message>,
    {
        let message: Message = prompt.into();
        let preview = extract_text(&message);
        let model_name = self.model_name.clone();

        let warnings = self.check_warnings();
        for w in &warnings {
            eprintln!("[rig-budget-guard] WARNING: {}", w.message);
        }

        let input_est = estimate_tokens(&preview);
        let builder = self.inner.completion(message, chat_history).await;

        match &builder {
            Ok(_) => {
                self.track_usage(&model_name, input_est, 0);
                self.record_event(&model_name, input_est, "completion", Some(preview));
            }
            Err(_) => {
                self.record_event(&model_name, 0, "completion_error", Some(preview));
            }
        }

        builder
    }
}

/// Extension trait to add budget-guard support to any Rig agent.
pub trait BudgetGuardExt<M: CompletionModel> {
    /// Wrap this agent in a `RigBudgetGuard`.
    fn with_budget_guard(self) -> RigBudgetGuard<M>;
}

impl<M: CompletionModel + 'static> BudgetGuardExt<M> for rig_core::agent::Agent<M> {
    fn with_budget_guard(self) -> RigBudgetGuard<M> {
        RigBudgetGuard::new(self)
    }
}

/// Classify a usage ratio into a phase label using threshold-based detection.
pub fn classify_phase(ratio: f64) -> Phase {
    if ratio >= 0.85 {
        Phase::Transitioning
    } else if ratio >= 0.60 {
        Phase::PreTransition
    } else if ratio >= 0.40 {
        Phase::Resolving
    } else {
        Phase::Stable
    }
}

// ══════════════════════════════════════════════════════════════════════
// TESTS
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::completion::message::Message;

    // ── ModelBudget ──────────────────────────────────────────────────

    #[test]
    fn test_model_budget_new() {
        let b = ModelBudget::new(1000);
        assert_eq!(b.max_tokens, 1000.0);
        assert_eq!(b.spent_tokens, 0.0);
    }

    #[test]
    fn test_remaining_full() {
        assert_eq!(ModelBudget::new(1000).remaining(), 1000.0);
    }

    #[test]
    fn test_remaining_partial() {
        let mut b = ModelBudget::new(1000);
        b.spent_tokens = 300.0;
        assert_eq!(b.remaining(), 700.0);
    }

    #[test]
    fn test_remaining_exhausted() {
        let mut b = ModelBudget::new(1000);
        b.spent_tokens = 1500.0;
        assert_eq!(b.remaining(), 0.0);
    }

    #[test]
    fn test_usage_ratio_empty() {
        assert_eq!(ModelBudget::new(1000).usage_ratio(), 0.0);
    }

    #[test]
    fn test_usage_ratio_half() {
        let mut b = ModelBudget::new(1000);
        b.spent_tokens = 500.0;
        assert!((b.usage_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_usage_ratio_full() {
        let mut b = ModelBudget::new(1000);
        b.spent_tokens = 1000.0;
        assert!((b.usage_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_usage_ratio_overflow() {
        let mut b = ModelBudget::new(1000);
        b.spent_tokens = 2000.0;
        assert!((b.usage_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_usage_ratio_zero_max() {
        assert_eq!(ModelBudget::new(0).usage_ratio(), 1.0);
    }

    #[test]
    fn test_remaining_exact_boundary() {
        let mut b = ModelBudget::new(100);
        b.spent_tokens = 100.0;
        assert_eq!(b.remaining(), 0.0);
        assert!((b.usage_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_spending_cannot_exceed_max() {
        let mut b = ModelBudget::new(100);
        b.spent_tokens = 50.0;
        assert!((b.usage_ratio() - 0.5).abs() < f64::EPSILON);
        b.spent_tokens += 60.0;
        assert!((b.usage_ratio() - 1.0).abs() < f64::EPSILON);
        assert_eq!(b.remaining(), 0.0);
    }

    // ── Phase classification ─────────────────────────────────────────

    #[test]
    fn test_classify_phase_stable() {
        assert_eq!(classify_phase(0.0), Phase::Stable);
        assert_eq!(classify_phase(0.1), Phase::Stable);
        assert_eq!(classify_phase(0.3), Phase::Stable);
    }

    #[test]
    fn test_classify_phase_pretransition() {
        assert_eq!(classify_phase(0.6), Phase::PreTransition);
        assert_eq!(classify_phase(0.7), Phase::PreTransition);
        assert_eq!(classify_phase(0.84), Phase::PreTransition);
    }

    #[test]
    fn test_classify_phase_transitioning() {
        assert_eq!(classify_phase(0.85), Phase::Transitioning);
        assert_eq!(classify_phase(0.9), Phase::Transitioning);
        assert_eq!(classify_phase(1.0), Phase::Transitioning);
    }

    #[test]
    fn test_classify_phase_resolving() {
        assert_eq!(classify_phase(0.4), Phase::Resolving);
        assert_eq!(classify_phase(0.5), Phase::Resolving);
        assert_eq!(classify_phase(0.59), Phase::Resolving);
    }

    #[test]
    fn test_classify_phase_edge_cases() {
        assert_eq!(classify_phase(0.39), Phase::Stable);
        assert_eq!(classify_phase(0.40), Phase::Resolving);
        assert_eq!(classify_phase(0.59), Phase::Resolving);
        assert_eq!(classify_phase(0.60), Phase::PreTransition);
        assert_eq!(classify_phase(0.84), Phase::PreTransition);
        assert_eq!(classify_phase(0.85), Phase::Transitioning);
    }

    // ── Phase config ─────────────────────────────────────────────────

    #[test]
    fn test_phase_config_default() {
        let c = PhaseConfig::default();
        assert!((c.pret_transition_threshold - 0.6).abs() < f64::EPSILON);
        assert!((c.transitioning_threshold - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_phase_config_serde_roundtrip() {
        let c = PhaseConfig::default();
        let json = serde_json::to_string(&c).unwrap();
        let de: PhaseConfig = serde_json::from_str(&json).unwrap();
        assert!((de.pret_transition_threshold - 0.6).abs() < f64::EPSILON);
        assert!((de.transitioning_threshold - 0.85).abs() < f64::EPSILON);
    }

    // ── Serde round-trips ────────────────────────────────────────────

    #[test]
    fn test_model_budget_serde_roundtrip() {
        let b = ModelBudget::new(50000);
        let json = serde_json::to_string(&b).unwrap();
        let de: ModelBudget = serde_json::from_str(&json).unwrap();
        assert!((de.max_tokens - 50000.0).abs() < f64::EPSILON);
        assert!((de.spent_tokens - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_budget_snapshot_serde_roundtrip() {
        let mut budgets: HashMap<String, ModelBudget> = HashMap::new();
        budgets.insert("gpt-4".into(), ModelBudget::new(100_000));
        let mut phases = HashMap::new();
        phases.insert("gpt-4".into(), "Stable".into());

        let snap = BudgetSnapshot {
            snapshot_id: "test-id".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            budgets,
            phases,
            events: vec![],
            warnings_active: vec![],
        };
        let json = serde_json::to_string(&snap).unwrap();
        let de: BudgetSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(de.snapshot_id, "test-id");
        assert_eq!(de.budgets.len(), 1);
        assert_eq!(de.phases.len(), 1);
    }

    #[test]
    fn test_budget_event_serde_roundtrip() {
        let e = BudgetEvent {
            id: "evt-1".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            model_name: "gpt-4".into(),
            tokens_consumed: 150,
            total_spent: 1500.0,
            budget_limit: 10000.0,
            phase: "PreTransition".into(),
            event_type: "prompt".into(),
            prompt_preview: Some("Hello".into()),
        };
        let json = serde_json::to_string(&e).unwrap();
        let de: BudgetEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(de.id, "evt-1");
        assert_eq!(de.tokens_consumed, 150);
    }

    #[test]
    fn test_budget_warning_serde_roundtrip() {
        let w = BudgetWarning {
            model: "gpt-4".into(),
            message: "Budget low".into(),
            usage_ratio: 0.9,
            phase: "Transitioning".into(),
        };
        let json = serde_json::to_string(&w).unwrap();
        let de: BudgetWarning = serde_json::from_str(&json).unwrap();
        assert_eq!(de.model, "gpt-4");
    }

    // ── Prompt truncation ────────────────────────────────────────────

    #[test]
    fn test_short_prompt_not_truncated() {
        let short = String::from("Hello");
        let result = if short.len() > 100 {
            format!("{}...", &short[..100])
        } else {
            short.clone()
        };
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_long_prompt_truncated() {
        let long = "x".repeat(200);
        let result = if long.len() > 100 {
            format!("{}...", &long[..100])
        } else {
            long
        };
        assert_eq!(result.len(), 103);
        assert!(result.ends_with("..."));
    }

    // ── Snapshot JSON ────────────────────────────────────────────────

    #[test]
    fn test_snapshot_json_format() {
        let mut budgets: HashMap<String, ModelBudget> = HashMap::new();
        budgets.insert("gpt-4".into(), ModelBudget::new(100_000));
        let mut phases = HashMap::new();
        phases.insert("gpt-4".into(), "Stable".into());

        let snap = BudgetSnapshot {
            snapshot_id: "snap-1".into(),
            timestamp: "2026-06-01T12:00:00Z".into(),
            budgets,
            phases,
            events: vec![],
            warnings_active: vec![],
        };
        let json = serde_json::to_string_pretty(&snap).unwrap();
        assert!(json.contains("snapshot_id"));
        assert!(json.contains("gpt-4"));
    }

    // ── Estimate tokens ─────────────────────────────────────────────

    #[test]
    fn test_estimate_tokens_short() {
        assert_eq!(estimate_tokens("abcd"), 1);
    }

    #[test]
    fn test_estimate_tokens_longer() {
        let s = "a".repeat(100);
        assert_eq!(estimate_tokens(&s), 25);
    }

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 1);
    }

    // ── Events capping ──────────────────────────────────────────────

    #[test]
    fn test_events_capped_at_1000() {
        let mut events: Vec<BudgetEvent> = (0..1500)
            .map(|i| BudgetEvent {
                id: format!("evt-{}", i),
                timestamp: "2026-01-01T00:00:00Z".into(),
                model_name: "test".into(),
                tokens_consumed: 10,
                total_spent: (i * 10) as f64,
                budget_limit: 10000.0,
                phase: "Stable".into(),
                event_type: "prompt".into(),
                prompt_preview: None,
            })
            .collect();

        if events.len() > 1000 {
            let drain_to = events.len() - 1000;
            events.drain(..drain_to);
        }
        assert_eq!(events.len(), 1000);
        assert_eq!(events[0].id, "evt-500");
    }

    // ── Multiple model budgets ──────────────────────────────────────

    #[test]
    fn test_multiple_model_budgets() {
        let mut budgets: HashMap<String, ModelBudget> = HashMap::new();
        budgets.insert("gpt-4".into(), ModelBudget::new(100_000));
        budgets.insert("gpt-3_5".into(), ModelBudget::new(50_000));
        budgets.insert("claude-3".into(), ModelBudget::new(200_000));
        assert_eq!(budgets.len(), 3);
        assert!(budgets.get("gpt-4").unwrap().max_tokens > 0.0);
        assert!(budgets.get("gpt-3_5").unwrap().max_tokens > 0.0);
        assert!(budgets.get("claude-3").unwrap().max_tokens > 0.0);
    }

    // ── Thread safety ───────────────────────────────────────────────

    #[test]
    fn test_snapshot_is_thread_safe() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BudgetSnapshot>();
        assert_send_sync::<ModelBudget>();
        assert_send_sync::<BudgetEvent>();
        assert_send_sync::<PhaseConfig>();
        assert_send_sync::<BudgetWarning>();
    }

    // ── Warning message formatting ─────────────────────────────────

    #[test]
    fn test_warning_message_formatting() {
        let w = BudgetWarning {
            model: "gpt-4".into(),
            message: "CRITICAL: Token budget at 90.0% for model gpt-4".into(),
            usage_ratio: 0.9,
            phase: "Transitioning".into(),
        };
        assert!(w.message.contains("CRITICAL"));
        assert!(w.message.contains("gpt-4"));
    }

    // ── Extract text from Message ──────────────────────────────────

    #[test]
    fn test_extract_text_system_message() {
        let msg = Message::System {
            content: String::from("system msg"),
        };
        let text = extract_text(&msg);
        assert_eq!(text, String::from("system msg"));
    }

    #[test]
    fn test_extract_text_empty_message() {
        let msg = Message::System {
            content: String::new(),
        };
        let text = extract_text(&msg);
        assert_eq!(text, String::new());
    }
}
