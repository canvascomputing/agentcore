EXAMPLES_DIR := crates/agent/examples

.PHONY: build test fmt clean example

build:
	RUSTFLAGS="-D warnings" cargo build

test:
	RUSTFLAGS="-D warnings" cargo test

fmt:
	cargo fmt

clean:
	cargo clean

example:
ifdef name
	cargo run -p agent --example $(name) $(ARGS)
else
	@echo "Available examples:"
	@ls $(EXAMPLES_DIR)/*.rs | xargs -n1 basename | sed 's/\.rs$$//' | sed 's/^/  /'
	@echo ""
	@echo "Run with: make example name=<example>"
endif
