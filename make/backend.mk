.PHONY: backend-build backend-check

backend-build:
	$(CARGO) build --release --locked --manifest-path backend/Cargo.toml

backend-check:
	$(CARGO) fmt --manifest-path backend/Cargo.toml --check
	$(CARGO) clippy --locked --manifest-path backend/Cargo.toml --all-targets -- -D warnings
	$(CARGO) test --locked --manifest-path backend/Cargo.toml

