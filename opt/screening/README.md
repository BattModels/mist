# High-Throughput Screening with MIST


This is our high-throughput screening workflow using [FASMIFRA](https://doi.org/10.1186/s13321-021-00566-4) for molecular generation and MIST as high-quality critic for filtering.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) [FASMIFRA](
2. Install [FASMIFRA](https://github.com/UnixJunkie/FASMIFRA) placing the binary at `./vendor/fasmifra`
3. See `uv run python screen.py --help` for run arguments

## Sample Screening Campaigns

- [Electrolytes](./config.yaml)
- [Fragrance, Non-Toxic Liquids](./config_fragrance.yaml)
- [Tom Ford's Eau de Soleil Fragrance](./config_tf_eau_de_soleil.yaml)
- [Tom Ford's Oud Wood Fragrance](./config_tf_oud_wood.yaml)

