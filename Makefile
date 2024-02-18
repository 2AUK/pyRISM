install:
	#RUSTFLAGS="-L /opt/software/openblas/gcc-8.5.0/0.3.21/lib/" pip install -e . -v
	#RUSTFLAGS="-L /opt/software/openblas/gcc-8.5.0/0.3.21/lib/" cargo install --path . --profile release
	pip install -e . -v
	cargo install --path . --profile release
