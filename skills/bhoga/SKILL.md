# bhoga — Hermes Skill

> Route AI harnesses to the best available provider based on live subscription quota.

## When to use

- Before making an API call, determine which provider has the most remaining quota
- After each API call, record the turn to keep quota tracking accurate
- When switching models in Hermes, apply the best provider to config

## Usage

### Check best provider for a model

```python
from bhoga import Router

router = Router()
rec = router.best_for("claude-opus-4")
if rec:
    print(f"Use {rec.hermes_model} ({rec.quota_pct:.0%} remaining)")
```

### Record API turns

```python
router.record_turn(
    pid="anthropic_api",
    model="claude-opus-4",
    input_tokens=1200,
    output_tokens=800,
    headers=response_headers,
    status_code=200,
)
```

### Apply to Hermes config

```python
from bhoga import Router, apply_to_hermes

router = Router()
apply_to_hermes(router, "claude-opus-4")
# Writes to ~/.hermes/config.yaml:
#   model: anthropic/claude-opus-4
#   compression.summary_provider: anthropic
#   auxiliary.*.provider: anthropic
```

### Handle 429 throttling

```python
router.record_turn(
    pid="openai_api", model="gpt-4.1",
    headers={"retry-after": "60"},
    status_code=429,
)
# Provider auto-backed-off for 60s, next best_for() skips it
```

## Provider IDs

| bhoga ID | Hermes ID | Source |
|---|---|---|
| `claude_code` | `anthropic` | OAuth API / CLI parse |
| `anthropic_api` | `anthropic` | HTTP headers |
| `openai-codex` | `openai-codex` | CLI parse |
| `openai_api` | `openrouter` | HTTP headers |
| `github-copilot` | `github-copilot` | HTTP headers |

## Dependencies

- `fastcore>=1.7`
- `httpx>=0.27`
- `python-dateutil>=2.9`
- `pyyaml>=6.0`
