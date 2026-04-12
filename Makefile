.PHONY: build test test_integration fmt clean update use-case litellm

# Build the project (warnings are errors)
build:
	RUSTFLAGS="-D warnings" cargo build

# Run unit tests (warnings are errors)
test:
	RUSTFLAGS="-D warnings" cargo test --lib

# Run integration tests (requires a live LLM provider, skips if none available)
test_integration:
	RUSTFLAGS="-D warnings" cargo test --tests -- --nocapture --test-threads=1

# Format all code
fmt:
	cargo fmt

# Remove build artifacts
clean:
	cargo clean

# Update dependencies
update:
	cargo update

# Run a use-case binary
# Usage: make use-case name=deep-research args="Should we use Rust or Go?"
# Note: use args= not -- to pass arguments
use-case:
ifdef name
	cargo run -p use-cases --bin $(name) -- $(args)
else
	@echo "Available use cases:"
	@grep -A1 '^\[\[bin\]\]' crates/use-cases/Cargo.toml | grep 'name' | sed 's/.*"\(.*\)"/  \1/'
	@echo ""
	@echo "Run with: make use-case name=<name> args=\"...\""
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
