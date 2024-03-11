# smirk: A Chemically Consistent SMILES Tokenizer

## Installing for development

> This is only needed if working with smirk directly. Poetry will handle building smirk at the MIST level

1. Install [maturin](https://www.maturin.rs): `pipx install maturin`
2. Install [rustup](https://www.rust-lang.org/tools/install)
3. Create a virtual environment for smirk: `python -m venv .venv` (Runs at `smirk/` to create `smirk/.venv`)
4. Build smirk: `maturin develop` and install python bindings into .venv
5. Use it: `python3 -c 'import smirk'`

### Misc Commands

All commands should be run from the `smirk` directory

- **Test**: run `cargo test`
- **Build**: run `cargo build`

