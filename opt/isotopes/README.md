The isotope workflow now fits new MIST-28M and MIST-1.8B models in
`opt/benchmarks`, using the available Wolfram cohort. It no longer requires an
unavailable historical checkpoint, a separate supplied cohort, or a user-written
projection protocol. The resulting figure is a new fit using the existing paper
analysis code, rather than an assertion of identical historical coordinates.

`benchmark-isotopes-materialize` downloads the [Wolfram JSON](https://www.wolframcloud.com/download/59368163-54cb-4881-97bb-0ee26df191c7?filename=Radioactive-Isotope-Half-Lives.json)
to `data/releases/isotopes-wolfram-1.0.0/Radioactive-Isotope-Half-Lives.json`
when the file is missing, verifying SHA256
`ecab89819c24167c830bf7c98ff5b44596a1655e0080b2154d6a2ba621618df8`.
An existing matching file is reused; a different file is rejected. The source
JSON is an output of the same materialization task. The task reuses
`preprocess_isotopes.py` and the original seed 42 80/10/10 splitter,
producing `data/isotopes/wolfram-seed42/{cohort,train,validation,test}.csv` and
`protocol.json`. The available source yields 924 rows: 739 training, 92 validation
and 93 test. The cohort keeps source order and assigns stable row IDs.

The original filter removes suffixes m/m1/m2 but retains 22 higher metastable
rows. Together with repeated measurements, this leaves 899 distinct isotope
SMILES among the 924 rows. These rows are preserved, without deduplication or
averaging. The existing row-level split shares 4 isotope SMILES between training
and validation, 5 between training and test, and 2 between validation and test.
Validation/test therefore contain held-out **rows**, not necessarily unseen
isotope identities. The materializer records these counts explicitly.

Each base uses the existing seven-rate benchmark sweep (2e-7 through 1.6e-4),
5,000 optimizer steps per cell, effective batch 16, seed 42, unfrozen encoder,
AdamW and the existing cosine schedule with beta2 warmup. The 28M microbatch is
16; the 1.8B microbatch is 4 with accumulation 4. Validation runs every 100 training
batches; selection uses validation loss before opening test rows. This preserves
the historical targets, batch size, split and step budget while adopting the
repository's current unfrozen LR-selection convention rather than its old
28M-specific progressive-thaw callback. Production fits are not shortened to the
smoke-test budget.

The two targets retain the original stored representation: `half_life` is a
scaled positive coefficient and `log_time` is the decimal time-scale exponent;
half-life in seconds is `half_life * 10**log_time`. Both targets are standardized
using training data. No conversion to a single log-half-life target is made.
The existing benchmark test report aggregates raw coefficient/exponent errors;
training logs also include per-channel MAE and R².

Task names are `benchmark-{mist-28m,mist-1p8b}-isotopes-lr-sweep-bounded` and
`benchmark-{mist-28m,mist-1p8b}-isotopes-test`. Selected model directories are
exported as `ISOTOPE_MODELS`; the materialization task is `ISOTOPE_DATA`.
`isotopes-paper-embeddings` and `isotopes-mist-1p8b-embeddings` consume the
corresponding selected model and all 924 cohort rows in order. Their `figure`
tasks produce `isotope_clusters.pdf/png` and `isotope-projection.csv` under
`results/isotopes/{mist-28m,mist-1p8b}/figure/`. `FIGURES_BY_MODEL` exposes both
sets; `FIGURES` retains the original 28M `isotope_clusters` assembly key.

`opt/isotopes/protocol.json` explicitly adopts the checked-in plotting defaults:
first-token embeddings, unmodified isotope SMILES, column standardization,
UMAP seed 42, 10 neighbors, min_dist=2 and Manhattan distance, plus decay tolerance
`0.8 + 0.02Z`. `legacy_numeric` includes neutron/proton counts alongside embedding
columns, matching the existing String/Symbol column-filter behavior. The paper
text's slope 0.2 and embedding-only alternative are not silently substituted.
Additional UMAP keyword overrides remain empty, using the installed UMAP.jl
version's defaults. The workflow Julia environment installs UMAP.jl when needed.

Both models reuse the already inventoried pinned MIST base snapshots. There are
no additional Hugging Face repositories or nested model/tokenizer downloads;
training and inference use the shared offline environment after explicit staging.
