.PHONY: build test test_integration fmt clean update use-case litellm bump publish

# Build the project (warnings are errors)
build:
	RUSTFLAGS="-D warnings" cargo build

# Run unit tests (warnings are errors)
test:
	RUSTFLAGS="-D warnings" cargo test --lib

# Run integration tests (requires a live LLM provider)
# Usage: make test_integration              (run all)
#        make test_integration name=bash_usage  (run one)
test_integration:
ifdef name
	RUSTFLAGS="-D warnings" cargo test --test $(name) -- --nocapture
else
	RUSTFLAGS="-D warnings" cargo test --tests -- --nocapture --test-threads=1
endif

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

# Bump version: make bump part=patch (default), minor, or major
part ?= patch
bump:
	@current=$$(grep '^version' crates/agentcore/Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/'); \
	IFS='.' read -r major minor patch <<< "$$current"; \
	case "$(part)" in \
		major) major=$$((major + 1)); minor=0; patch=0;; \
		minor) minor=$$((minor + 1)); patch=0;; \
		patch) patch=$$((patch + 1));; \
		*) echo "Unknown part: $(part). Use major, minor, or patch."; exit 1;; \
	esac; \
	new="$$major.$$minor.$$patch"; \
	sed -i '' "s/^version = \"$$current\"/version = \"$$new\"/" crates/agentcore/Cargo.toml; \
	echo "$$current → $$new"

# Publish to crates.io: make publish
publish: test
	cargo publish -p agentcore

# Start a LiteLLM proxy on localhost:4000
# Forwards the provider's API key from your environment (never leaked in commands)
# Usage: make litellm                  (default: anthropic, uses ANTHROPIC_API_KEY)
#        make litellm provider=openai  (uses OPENAI_API_KEY)
provider ?= anthropic

# Map provider name to its API key env var
ifeq ($(provider),anthropic)
  LITELLM_KEY_ENV     := ANTHROPIC_API_KEY
  LITELLM_MODEL_ENV   := ANTHROPIC_MODEL
  LITELLM_DEFAULT_MDL := claude-sonnet-4-20250514
else ifeq ($(provider),mistral)
  LITELLM_KEY_ENV     := MISTRAL_API_KEY
  LITELLM_MODEL_ENV   := MISTRAL_MODEL
  LITELLM_DEFAULT_MDL := mistral-small-2603
else ifeq ($(provider),openai)
  LITELLM_KEY_ENV     := OPENAI_API_KEY
  LITELLM_MODEL_ENV   := OPENAI_MODEL
  LITELLM_DEFAULT_MDL := gpt-4o
else
  LITELLM_KEY_ENV     :=
  LITELLM_MODEL_ENV   :=
  LITELLM_DEFAULT_MDL :=
endif

LITELLM_MODEL_VAL := $(or $($(LITELLM_MODEL_ENV)),$(LITELLM_DEFAULT_MDL))

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
