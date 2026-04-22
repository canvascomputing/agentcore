.PHONY: build test test_integration fmt clean update use_case litellm bump

# Build the project (warnings are errors)
build: fmt
	RUSTFLAGS="-D warnings" cargo build

# Run unit tests (warnings are errors) — inline `#[cfg(test)]` + mock-driven tests in tests/unit/
test:
	RUSTFLAGS="-D warnings" cargo test --lib --test unit

# Run integration tests (requires a live LLM LITELLM_PROVIDER)
# Usage: make test_integration              (run all)
#        make test_integration name=bash_usage  (run one file)
test_integration:
ifdef name
	RUSTFLAGS="-D warnings" cargo test --test integration $(name) -- --nocapture
else
	RUSTFLAGS="-D warnings" cargo test --test integration -- --nocapture --test-threads=1
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

# Run a use_case binary
# Usage: make use_case name=deep-research args="Should we use Rust or Go?"
# Note: use args= not -- to pass arguments
use_case:
ifdef name
	cargo run -p use-cases --bin $(name) -- $(args)
else
	@echo "Available use cases:"
	@grep -A1 '^\[\[bin\]\]' crates/use-cases/Cargo.toml | grep 'name' | sed 's/.*"\(.*\)"/  \1/'
	@echo ""
	@echo "Run with: make use_case name=<name> args=\"...\""
endif

# Bump version, test, commit, and tag for release: make bump part=patch (default), minor, or major
# GitHub Actions handles the crates.io publish via trusted publishing after you push the tag
part ?= patch
bump: test
	@current=$$(grep '^version' crates/agentwerk/Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/'); \
	IFS='.' read -r major minor patch <<< "$$current"; \
	case "$(part)" in \
		major) major=$$((major + 1)); minor=0; patch=0;; \
		minor) minor=$$((minor + 1)); patch=0;; \
		patch) patch=$$((patch + 1));; \
		*) echo "Unknown part: $(part). Use major, minor, or patch."; exit 1;; \
	esac; \
	new="$$major.$$minor.$$patch"; \
	sed -i '' "s/^version = \"$$current\"/version = \"$$new\"/" crates/agentwerk/Cargo.toml; \
	git add -A && git commit -m "v$$new" && \
	git tag "v$$new" && \
	echo "$$current → $$new" && \
	echo "Tagged v$$new — now run:" && \
	echo "  git push && git push --tags"

# Start a LiteLLM proxy on localhost:4000
# Forwards the provider's API key from your environment (never leaked in commands)
# Usage: make litellm                  (default: anthropic, uses ANTHROPIC_API_KEY)
#        make litellm LITELLM_PROVIDER=openai  (uses OPENAI_API_KEY)
LITELLM_PROVIDER ?= anthropic

# Map LITELLM_PROVIDER to its API key env var
ifeq ($(LITELLM_PROVIDER),anthropic)
  LITELLM_KEY_ENV     := ANTHROPIC_API_KEY
  LITELLM_MODEL_ENV   := ANTHROPIC_MODEL
  LITELLM_DEFAULT_MDL := claude-sonnet-4-20250514
else ifeq ($(LITELLM_PROVIDER),mistral)
  LITELLM_KEY_ENV     := MISTRAL_API_KEY
  LITELLM_MODEL_ENV   := MISTRAL_MODEL
  LITELLM_DEFAULT_MDL := mistral-small-2603
else ifeq ($(LITELLM_PROVIDER),openai)
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
	$(error Unsupported LITELLM_PROVIDER "$(LITELLM_PROVIDER)". Supported: anthropic, mistral, openai)
endif
	@printf '%s\n' \
		'model_list:' \
		'  - model_name: $(LITELLM_MODEL_VAL)' \
		'    litellm_params:' \
		'      model: $(LITELLM_PROVIDER)/$(LITELLM_MODEL_VAL)' \
		'      api_key: os.environ/$(LITELLM_KEY_ENV)' \
		> /tmp/agent_litellm_config.yaml
	docker run --rm \
		-e $(LITELLM_KEY_ENV) \
		-v /tmp/agent_litellm_config.yaml:/app/config.yaml:ro \
		-p 4000:4000 \
		docker.litellm.ai/berriai/litellm:main-stable \
		--config /app/config.yaml
