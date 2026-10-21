.PHONY: backend-build backend-check

backend-build:
	$(CARGO) build --release --locked --manifest-path backend/Cargo.toml --package desynth-cli --bin desynth

backend-check:
	$(CARGO) fmt --manifest-path backend/Cargo.toml --all --check
	$(CARGO) clippy --locked --manifest-path backend/Cargo.toml --workspace --all-targets -- -D warnings
	$(CARGO) test --locked --manifest-path backend/Cargo.toml --workspace

