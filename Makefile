EXAMPLES_DIR := crates/agent/examples

.PHONY: build test fmt clean example litellm

# Build the project (warnings are errors)
build:
	RUSTFLAGS="-D warnings" cargo build

# Run all tests (warnings are errors)
test:
	RUSTFLAGS="-D warnings" cargo test

# Format all code
fmt:
	cargo fmt

# Remove build artifacts
clean:
	cargo clean

# Override model or base URLs for examples
# Usage: make example name=code_review
#        make example name=code_review ANTHROPIC_MODEL=claude-haiku-4-5-20251001
#        make example name=code_review ANTHROPIC_BASE_URL=https://custom.api.com
export ANTHROPIC_MODEL ?= claude-sonnet-4-20250514
export MISTRAL_MODEL ?= mistral-medium-2508
export ANTHROPIC_BASE_URL
export LITELLM_API_URL
export LITELLM_API_KEY
export LITELLM_MODEL
export MISTRAL_API_KEY
export MISTRAL_BASE_URL
export MISTRAL_MODEL

example:
ifdef name
	cargo run -p agent --example $(name) $(ARGS)
else
	@echo "Available examples:"
	@ls $(EXAMPLES_DIR)/*.rs | xargs -n1 basename | sed 's/\.rs$$//' | sed 's/^/  /'
	@echo ""
	@echo "Run with: make example name=<example>"
endif

# Start a LiteLLM proxy on localhost:4000
# Forwards the provider's API key from your environment (never leaked in commands)
# Usage: make litellm                  (default: anthropic, uses ANTHROPIC_API_KEY)
#        make litellm provider=openai  (uses OPENAI_API_KEY)
provider ?= anthropic

# Map provider name to its API key env var
ifeq ($(provider),anthropic)
  LITELLM_KEY_ENV   := ANTHROPIC_API_KEY
  LITELLM_MODEL_ENV := ANTHROPIC_MODEL
else ifeq ($(provider),mistral)
  LITELLM_KEY_ENV   := MISTRAL_API_KEY
  LITELLM_MODEL_ENV := MISTRAL_MODEL
else ifeq ($(provider),openai)
  LITELLM_KEY_ENV   := OPENAI_API_KEY
  LITELLM_MODEL_ENV := OPENAI_MODEL
else
  LITELLM_KEY_ENV   :=
  LITELLM_MODEL_ENV :=
endif

LITELLM_MODEL_VAL := $(or $($(LITELLM_MODEL_ENV)),claude-sonnet-4-20250514)

litellm:
ifndef LITELLM_KEY_ENV
	$(error Unsupported provider "$(provider)". Supported: anthropic, openai)
endif
	@printf '%s\n' \
		'model_list:' \
		'  - model_name: $(LITELLM_MODEL_VAL)' \
		'    litellm_params:' \
		'      model: $(provider)/$(LITELLM_MODEL_VAL)' \
		'      api_key: os.environ/$(LITELLM_KEY_ENV)' \
		> /tmp/agent_litellm_config.yaml
	docker run --rm \
		-e $(LITELLM_KEY_ENV) \
		-v /tmp/agent_litellm_config.yaml:/app/config.yaml:ro \
		-p 4000:4000 \
		docker.litellm.ai/berriai/litellm:main-stable \
		--config /app/config.yaml
