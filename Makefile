.PHONY: all build test clean lint

all: build

build:
	cargo build --release

test:
	cargo test

lint:
	cargo clippy --all-targets --all-features -- -D warnings
	cargo fmt --check

clean:
	cargo clean
