install:
	pip install -e . -v
	cargo install --path . --profile release
