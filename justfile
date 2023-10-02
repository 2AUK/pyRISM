build:
    pip install -e . -v

test:
    cargo test -- --nocapture

run:
    cat pyrism/data/cSPCE_XRISM.toml
    pyrism pyrism/data/cSPCE_XRISM.toml -v

full: build run