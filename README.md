# bhoga

**Zero-server Python library that routes AI harnesses to the best available provider for a model, based on live subscription quota.**

bhoga (Sanskrit: *route, path, enjoyment of resources*) discovers quota automatically from Claude Code, Codex CLI, and HTTP response headers — no LLM calls, no servers, no configuration required to get started.

## Quick start

```python
from bhoga import Router, apply_to_hermes

router = Router()                          # background quota discovery starts immediately

rec = router.best_for("claude-opus-4")    # → RouterRecommendation | None
print(rec.hermes_model)                    # "claude-opus-4"  (normalised for Anthropic API)
print(rec.provider_id, rec.quota_pct)      # "claude_code"  0.80

# After each API call, record the turn (non-blocking)
router.record_turn(
    pid="anthropic_api",
    model="claude-opus-4",
    input_tokens=1200,
    output_tokens=800,
    headers=dict(response.headers),
    status_code=response.status_code,
)

# Write recommendation directly into Hermes config.yaml
apply_to_hermes(router, "claude-opus-4")
```

## Installation

```bash
pip install bhoga
```

Requires Python 3.11+.

---

## How it works

### No LLM dependency

Quota discovery uses structured APIs and regex parsing only:

| Provider | Discovery method |
|---|---|
| **Claude Code** | Reads `~/.claude/.credentials.json` → OAuth API (`/api/oauth/usage`) → `claude -p /usage` fallback |
| **Codex CLI** | `codex --non-interactive /status` (JSON-first, TUI text fallback) |
| **API providers** | Calibrated from HTTP response headers after each turn |

### Automatic routing

`best_for()` walks the provider hierarchy for a model family and picks the provider with the most remaining quota. Priority breaks ties.

| Family | Priority 1 | Priority 2 | Priority 3 |
|---|---|---|---|
| anthropic | `claude_code` | `anthropic_api` | `github-copilot` |
| openai | `openai-codex` | `openai_api` | `github-copilot` |

### Dual cadence for subscription providers

Claude Code and Codex have two rate-limit windows:

- **burst** (5 hours) — short-term usage cap
- **weekly** (7 days) — longer-term usage cap

`best_for()` uses the **tighter** window (minimum remaining %) for routing decisions.

---

## `Router.set_quota()` — manual fallback

When automatic discovery is unavailable (no CLI installed, running in CI, known quota state), set quota explicitly:

```python
router = Router()

# Windowed providers (burst + optional separate weekly)
router.set_quota("openai-codex", pct_remaining=75.0)
router.set_quota("claude_code",  pct_remaining=90.0, weekly_pct=60.0)

# Request/token-based providers
router.set_quota("github-copilot", pct_remaining=85.0)
router.set_quota("anthropic_api",  pct_remaining=50.0)

# Target a specific model (default is "*" = all models for this provider)
router.set_quota("anthropic_api", pct_remaining=60.0, model="claude-opus-4")
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `pid` | `str` | Provider ID — `"claude_code"`, `"openai-codex"`, `"github-copilot"`, `"anthropic_api"`, `"openai_api"` |
| `pct_remaining` | `float` | Percentage of quota remaining, 0–100. For windowed providers, sets the burst window. |
| `weekly_pct` | `float \| None` | Weekly window remaining percentage (windowed providers only). Defaults to `pct_remaining`. |
| `model` | `str` | Model key. Default `"*"` covers all models for this provider. |

State is persisted to `~/.cache/bhoga/state.json` (override with `BHOGA_STATE` env var) via atomic rename, so it survives restarts.

### Typical harness pattern

```python
router = Router()

# Try automatic discovery first
rec = router.best_for("claude-opus-4")

# If we know quota from our own monitoring, override it
if my_monitoring.claude_code_burst_pct is not None:
    router.set_quota("claude_code",
                     pct_remaining=my_monitoring.claude_code_burst_pct,
                     weekly_pct=my_monitoring.claude_code_weekly_pct)
    rec = router.best_for("claude-opus-4")

apply_to_hermes(router, "claude-opus-4")
```

---

## Codex model discovery

```python
from bhoga import get_codex_models

models = get_codex_models()
# ['gpt-5.4', 'gpt-5.3-codex', 'gpt-5.4-mini', ...]
```

Resolution order (no network call):

1. `~/.codex/models_cache.json` — written by Codex CLI on first run
2. `~/.codex/config.toml` — the model the user has configured
3. Built-in defaults (current Codex model line-up)

---

## `apply_to_hermes()` — Hermes integration

Writes the routing recommendation directly into `~/.hermes/config.yaml`:

```python
apply_to_hermes(router, "claude-opus-4")
# Writes:
#   model:
#     default: claude-opus-4
#     provider: anthropic
#     api_mode: anthropic_messages
#   compression:
#     summary_provider: anthropic

apply_to_hermes(router, "gpt-5.4", write_auxiliary=True)
# Also writes all auxiliary task providers (vision, compression, …)
```

**Provider → Hermes mapping:**

| bhoga `provider_id` | Hermes canonical ID | `api_mode` |
|---|---|---|
| `claude_code` | `anthropic` | `anthropic_messages` |
| `anthropic_api` | `anthropic` | `anthropic_messages` |
| `openai-codex` | `openai-codex` | `codex_responses` |
| `openai_api` | `openrouter` | `chat_completions` |
| `github-copilot` | `copilot` | `chat_completions` |

**Model name normalisation** (per `hermes_cli/model_normalize.py`):

| Target provider | Format | Example |
|---|---|---|
| `openrouter` | `vendor/model` | `openai/gpt-5.4` |
| `anthropic` | bare, dots→hyphens | `claude-sonnet-4-6` |
| `copilot` | bare, dots preserved | `claude-sonnet-4.6` |
| `openai-codex` | bare, dots preserved | `gpt-5.4` |

For GitHub Copilot, a non-destructive `providers.copilot` stub is written so Hermes knows the base URL and auth env var (`COPILOT_GITHUB_TOKEN`).

---

## GitHub Copilot auth

bhoga resolves the Copilot token in this order (matching `hermes_cli/copilot_auth.py`):

```
COPILOT_GITHUB_TOKEN → GH_TOKEN → GITHUB_TOKEN
```

Supported token types: `gho_` (OAuth), `github_pat_` (fine-grained PAT), `ghu_` (GitHub App).  
Classic PATs (`ghp_`) are **not** supported by the Copilot API.

---

## API reference

### `Router`

```python
Router(*, state_path: Path | None = None, eager: bool = True)
```

- `state_path` — override default state file location
- `eager=True` — start background quota discovery immediately

**Methods:**

| Method | Description |
|---|---|
| `best_for(model)` | Returns `RouterRecommendation \| None` |
| `record_turn(pid, model, input_tokens, output_tokens, headers, status_code)` | Non-blocking; queued for background calibration |
| `set_quota(pid, pct_remaining, *, weekly_pct=None, model="*")` | Manual quota override |
| `quotas()` | Returns a snapshot of all tracked `ProviderQuota` objects |

### Module-level helpers

| Function | Description |
|---|---|
| `apply_to_hermes(router, model, *, config_path=None, write_auxiliary=False)` | Write routing recommendation to Hermes config |
| `get_codex_models()` | List available Codex model IDs from local files or defaults |
| `model_family(model)` | Map a model name to its family (`"anthropic"`, `"openai"`, `"google"`, `"other"`) |
| `to_hermes_model(pid, mid)` | Normalise a model name for the given bhoga provider |
| `calibrate(q, headers)` | Update `ProviderQuota` from HTTP response headers |
| `mark_throttled(q, retry_after)` | Mark a provider as throttled |
| `check_quota(pid)` | Run quota discovery for a subscription provider |

### `RouterRecommendation`

```python
@dataclass
class RouterRecommendation:
    provider_id:  str           # e.g. "claude_code"
    model_id:     str           # e.g. "claude-opus-4"
    quota_pct:    float         # 0.0–1.0, or -1.0 if unknown
    status:       ProviderStatus
    hermes_model: str           # normalised model name for target provider
    priority:     int
```

---

## State persistence

Quota state is stored in `~/.cache/bhoga/state.json`.  Override with `BHOGA_STATE`:

```bash
BHOGA_STATE=/tmp/my-bhoga-state.json python my_harness.py
```

All writes are atomic (write `.tmp` → `rename`) so concurrent processes cannot corrupt state.

---

## Thread safety

All `Router` methods are thread-safe.  Background threads:

- `_init_bg` — initial quota discovery (one-shot, daemon)
- `_worker_loop` — single persistent daemon that drains the `record_turn` queue serially, eliminating TOCTOU races and unbounded thread spawning

