# rig-budget-guard

[![CI](https://github.com/SuperInstance/rig-budget-guard/actions/workflows/ci.yml/badge.svg)](https://github.com/SuperInstance/rig-budget-guard/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/rig-budget-guard)](https://crates.io/crates/rig-budget-guard)
[![docs.rs](https://img.shields.io/docsrs/rig-budget-guard)](https://docs.rs/rig-budget-guard)

**Token budget enforcement middleware for [Rig](https://github.com/0xPlaygrounds/rig)-based LLM applications.**

`rig-budget-guard` wraps any Rig agent and intercepts `prompt`, `chat`, and `completion` calls to track token spending against per-model budgets using [conservation-checker](https://crates.io/crates/conservation-checker) for phase detection.

---

## The Problem

LLM APIs are **expensive**. When you deploy a Rig-based application — a chat bot, a reasoning pipeline, or an agent workflow — you need to answer:

- How many tokens has this model consumed so far?
- Are we approaching our monthly or per-deployment budget?
- When should we start warning users or throttling usage?
- Which models are burning through their allocation too fast?

Without a budget layer, every `prompt()` call is a blind spend. You only discover the cost when the invoice arrives.

## The Solution

`rig-budget-guard` wraps any Agent, tracks every completion against a model-level budget, detects spending **phases** (from stable to critical), prepends warming headers to responses when thresholds are exceeded, and exports full Serde snapshots for audit and monitoring.

## Architecture

```
┌─────────────────────────────────────────┐
│           Your Application              │
│  prompt("...") / chat("...", ...)       │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│         RigBudgetGuard<M>               │
│  ┌─────────────────────────────────┐    │
│  │  ConservationChecker            │    │
│  │  . Phase detection              │    │
│  │  . Budget tracking              │    │
│  │  . Event log (capped at 1000)   │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │  Inner Agent<M> (delegated)     │    │
│  └─────────────────────────────────┘    │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│         LLM Provider (OpenAI, etc.)     │
└─────────────────────────────────────────┘
```

### Phase Detection

Token spending follows a predictable progression:

| Phase | Meaning | Default Threshold | Action |
|---|---|---|---|
| **Stable** | Spending is well within budget | < 40% | No action |
| **Resolving** | Approaching watch zone | 40–59% | Quiet monitoring |
| **PreTransition** | Spending accelerating | 60–84% | Log warnings, optional alerts |
| **Transitioning** | Critically close to limit | 85%+ | Prepend budget header to responses |

### Key Design Decisions

- **One-sided conservation**: Spending increases normally (budget → 0), but the guard tracks remaining budget. The consumer naturally slows down as budget depletes.
- **Per-model tracking**: Different budgets for different models (e.g., GPT-4 gets \$0.03/1K tokens, GPT-3.5 gets \$0.0015/1K).
- **Serde snapshots**: Full audit trail serializable to JSON. Ship to a log aggregator, dashboard, or SIEM.
- **Arc\<Mutex\<>> interior mutability**: All Rig trait methods take `&self`, so budgets live behind shared state.

## Quick Start

### 1. Add the dependency

```toml
[dependencies]
rig-budget-guard = "0.1"
rig-core = "0.38"
conservation-checker = "0.1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["rt", "macros"] }
uuid = { version = "1", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
```

### 2. Wrap your agent

```rust
use rig_budget_guard::RigBudgetGuard;

// Build your Rig agent as usual
let agent = openai.agent("gpt-4")
    .preamble("You are a helpful assistant.")
    .build();

// Wrap with a budget
let guarded = RigBudgetGuard::new(agent)
    .with_budget("gpt-4", 100_000)   // 100K token budget for GPT-4
    .with_warn_threshold(0.8);        // Warn at 80% usage

// Use just like a normal agent
let response = guarded.prompt("Hello, world!").await?;
println!("{}", response);
```

### 3. Use with multiple models

```rust
use rig_budget_guard::RigBudgetGuard;

let gpt4_agent = openai.agent("gpt-4").build();
let gpt35_agent = openai.agent("gpt-3.5-turbo").build();

let gpt4_guarded = RigBudgetGuard::new(gpt4_agent)
    .with_budget("gpt-4", 100_000)    // 100K tokens for GPT-4
    .with_warn_threshold(0.9);         // Stricter for expensive model

let gpt35_guarded = RigBudgetGuard::new(gpt35_agent)
    .with_budget("gpt-3.5-turbo", 500_000)    // 500K for cheap model
    .with_warn_threshold(0.8);
```

## User Guide

### Builder API

```rust
let guarded = RigBudgetGuard::new(agent)
    .named("production-chatbot")                          // Optional instance name
    .with_budget("gpt-4", 100_000)                        // Per-model token budget
    .with_budgets([                                       // Bulk set budgets
        ("gpt-4", 100_000),
        ("gpt-3.5-turbo", 500_000),
    ])
    .with_warn_threshold(0.85)                            // Warning threshold
    .with_phase_config(PhaseConfig {                      // Custom phase thresholds
        pret_transition_threshold: 0.5,
        transitioning_threshold: 0.75,
    });
```

### Phase Detection

The guard uses two mechanisms:

1. **ConservationChecker** tracks history-aware phase transitions through `Stable → Resolving → PreTransition → Transitioning`.
2. **`classify_phase()`** convenience function provides static ratio-based classification.

```rust
use rig_budget_guard::classify_phase;
use conservation_checker::Phase;

// Threshold-based:
let phase = classify_phase(0.82);  // Phase::PreTransition

// Check per-model phase:
if let Some(phase) = guard.phase("gpt-4") {
    match phase {
        Phase::Transitioning => eprintln!("CRITICAL: gpt-4 budget exhausted!"),
        Phase::PreTransition => eprintln!("WARNING: gpt-4 approaching budget!"),
        _ => {} // Stable or Resolving
    }
}
```

### Snapshots & Audit

```rust
// Get a full snapshot (thread-safe)
let snapshot: BudgetSnapshot = guard.snapshot();
println!("{}", serde_json::to_string_pretty(&snapshot)?);

// Or get pretty-printed JSON directly
println!("{}", guard.snapshot_json());
```

Example snapshot output:

```json
{
  "snapshot_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2026-06-02T01:00:00-08:00",
  "budgets": {
    "gpt-4": {
      "max_tokens": 100000.0,
      "spent_tokens": 42350.0
    }
  },
  "phases": {
    "gpt-4": "PreTransition"
  },
  "events": [
    {
      "id": "evt-0001",
      "timestamp": "2026-06-02T01:00:00-08:00",
      "model_name": "gpt-4",
      "tokens_consumed": 150,
      "total_spent": 42350.0,
      "budget_limit": 100000.0,
      "phase": "PreTransition",
      "event_type": "prompt",
      "prompt_preview": "What is the capital of France?"
    }
  ],
  "warnings_active": ["gpt-4"]
}
```

### Warning Headers

When usage exceeds the warning threshold, the guard prepends a header to responses:

```
[Budget: 86% used • Phase: PreTransition]
The capital of France is Paris. It has been the capital since...
```

### Thread Safety

All budget state is behind `Arc<Mutex<>>`. You can clone `RigBudgetGuard` and share it across threads/tasks. `BudgetSnapshot`, `ModelBudget`, `BudgetEvent`, `PhaseConfig`, and `BudgetWarning` all implement `Send + Sync`.

## Templates

### OpenAI Integration

```rust
use rig_budget_guard::RigBudgetGuard;
use rig_core::providers::openai;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = openai::Client::new(
        std::env::var("OPENAI_API_KEY")?.into()
    )?;
    let agent = client
        .agent("gpt-4o")
        .preamble("You are a coding assistant.")
        .build();

    let guard = RigBudgetGuard::new(agent)
        .named("code-assistant")
        .with_budget("gpt-4o", 500_000)
        .with_warn_threshold(0.8);

    let response = guard.prompt("Write a Rust function to reverse a string.").await?;
    println!("{response}");

    // Export snapshot for monitoring
    std::fs::write("/var/log/budget-snapshot.json", guard.snapshot_json())?;
    Ok(())
}
```

### Dashboard Integration

```rust
use rig_budget_guard::BudgetSnapshot;

// Periodically collect snapshots:
fn report_budgets(guard: &RigBudgetGuard<impl CompletionModel>) -> String {
    let snapshot: BudgetSnapshot = guard.snapshot();
    serde_json::to_string_pretty(&snapshot).unwrap()
}

// For Prometheus-style metrics:
fn metrics(guard: &RigBudgetGuard<impl CompletionModel>) -> String {
    let snap = guard.snapshot();
    let mut out = String::new();
    for (model, budget) in &snap.budgets {
        let pct = (budget.spent_tokens / budget.max_tokens * 100.0) as u64;
        out.push_str(&format!(
            "rig_budget_guard_used{} {}",
            model, budget.spent_tokens
        ));
        out.push_str(&format!(
            "rig_budget_guard_remaining{} {}",
            model, budget.remaining()
        ));
    }
    out
}
```

### Multi-Tenant Budget

```rust
use std::collections::HashMap;

struct TenantBudgets {
    guards: HashMap<String, RigBudgetGuard<YourModel>>,
}

impl TenantBudgets {
    async fn process_request(
        &self,
        tenant: &str,
        prompt: &str,
    ) -> Result<String, PromptError> {
        let guard = self.guards.get(tenant)
            .ok_or_else(|| PromptError("Unknown tenant"))?;

        let phase = guard.phase("gpt-4");
        if matches!(phase, Some(Phase::Transitioning)) {
            return Err(PromptError("Budget exhausted for tenant"));
        }

        guard.prompt(prompt).await
    }
}
```

## Comparison with Alternatives

| Feature | rig-budget-guard | Manual tracking |
|---|---|---|
| Automatic middleware | ✅ Yes | ❌ Requires wrapper code |
| Per-model budgets | ✅ Yes | ❌ Manual |
| Phase detection | ✅ Yes (ConservationChecker) | ❌ |
| Audit trail (Serde) | ✅ Yes | ❌ |
| Thread-safe | ✅ Yes (Arc<Mutex<>>) | ❌ |
| Warning headers | ✅ Yes | ❌ |
| Event log (capped) | ✅ Yes (1000 events) | ❌ |
| Open source | ✅ Yes | N/A |

## Changelog

### 0.1.0
- Initial release
- `RigBudgetGuard<M>` wrapping any Rig agent
- Per-model token budgets with `ConservationChecker`
- Phase detection: Stable, Resolving, PreTransition, Transitioning
- Serde-serializable snapshots for audit
- Warning headers when thresholds exceeded
- `BudgetGuardExt` extension trait
- 30+ unit tests
- Clippy-clean

## License

MIT or Apache-2.0, at your option.
