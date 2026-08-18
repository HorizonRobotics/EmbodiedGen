# GPT Agent Setup

Most EmbodiedGen pipelines use `GPT_CLIENT` from
`embodied_gen/utils/gpt_clients.py`. Select one backend in
`embodied_gen/utils/gpt_config.yaml`, or override its values with environment
variables.

| Backend | Best for | Authentication |
|---|---|---|
| Azure OpenAI | Shared services and managed deployments | Azure endpoint, API key, and API version |
| OpenRouter | OpenAI-compatible hosted models | OpenRouter API key |
| Codex CLI | Local developers already using Codex | Existing `codex login` session |

Do not commit real API keys or Codex credentials to the repository.

## Azure OpenAI

Select the Azure configuration and provide the deployment details:

```yaml
agent_type: gpt-5.4

gpt-5.4:
  endpoint: https://YOUR-RESOURCE.openai.azure.com/
  api_key: YOUR_AZURE_OPENAI_API_KEY
  api_version: YOUR_AZURE_API_VERSION
  model_name: YOUR_DEPLOYMENT_NAME
```

Azure uses `model_name` as the deployment name. The deployment must support
the text or image inputs required by the selected EmbodiedGen pipeline.

## OpenRouter

Select an OpenRouter model and update its API key:

```yaml
agent_type: gemma-4-31b

gemma-4-31b:
  endpoint: https://openrouter.ai/api/v1
  api_key: YOUR_OPENROUTER_API_KEY
  api_version: null
  model_name: google/gemma-4-31b-it:free
```

Model availability and free-tier limits can change. Confirm that the selected
model supports image inputs before using an image-based pipeline.

## Codex CLI

The Codex backend is intended for local development. It starts a temporary,
non-interactive `codex exec` request for each `GPT_CLIENT.query()` call and
reuses the developer's existing Codex authentication. It does not read an API
key from `gpt_config.yaml`.

Install Codex, sign in, and verify the login:

```bash
codex login
codex login status
```

This integration was validated with `codex-cli 0.146.0`. Upgrade Codex if
`codex exec` does not recognize `--ephemeral`, `--ignore-user-config`,
`--ignore-rules`, or `--output-last-message`.

Then select the Codex backend:

```yaml
agent_type: codex

codex:
  provider: codex
  endpoint: null
  api_key: null
  api_version: null
  model_name: null
```

Leaving `model_name` as `null` uses the Codex default model. To select a
specific model for EmbodiedGen, set `model_name` or export `MODEL_NAME`.

The adapter runs Codex with an ephemeral session and a read-only sandbox.
It also ignores user-level Codex configuration, MCP servers, and project rules
so that only the existing authentication is reused. The subprocess receives a
minimal environment and does not inherit EmbodiedGen API keys. Text and image
prompts are supported, but launching the CLI for every request has more
overhead than an API backend. The Codex agent can still use built-in read-only
tools, so use this backend only with trusted local prompts. Use Azure OpenAI or
OpenRouter for containers, hosted services, batch workloads, untrusted input,
and Hugging Face Spaces.

Per-request API sampling options such as `temperature`, `top_p`, and
`max_tokens` are not mapped to Codex CLI flags. The optional `model` and
`model_reasoning_effort` entries in `params` provide per-request Codex
overrides, for example `params={"model_reasoning_effort": "high"}`. Supported
effort levels depend on the selected Codex model; EmbodiedGen uses `medium` when
no per-request effort is specified.

For Codex authentication details, see the
[official Codex authentication documentation](https://developers.openai.com/codex/auth).

## Environment Variables

Environment variables take precedence over `gpt_config.yaml`:

```bash
export GPT_PROVIDER=azure       # azure, openai, or codex
export ENDPOINT=...
export API_KEY=...
export API_VERSION=...
export MODEL_NAME=...
export GPT_TIMEOUT=120
```

For Codex, only `GPT_PROVIDER=codex`, `MODEL_NAME`, and `GPT_TIMEOUT` are
relevant. Setting `GPT_PROVIDER` starts a clean provider override and does not
inherit endpoint, key, API version, or model values from the selected YAML API
backend. The existing Codex login supplies authentication.

## Verify the Configuration

Run a small text request from the repository root:

```bash
python - <<'PY'
from embodied_gen.utils.gpt_clients import GPT_CLIENT

print(GPT_CLIENT.query("Reply with: EmbodiedGen GPT setup works."))
PY
```

If Codex cannot be found, confirm that `codex` is available in `PATH`. For API
backends, verify the endpoint, API key, API version, and model or deployment
name.
