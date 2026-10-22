# Interpretable Embeddings

`workflow.pomset` produces the original token-embedding main/SI panels and the
PAH/aromaticity embedding panels under `results/interpretability/<model>/`.
Both `mist-28m` and `mist-1p8b` contain
`{token-embeddings,token-updates,all-token-updates,pah}`. Directories describe
the analysis rather than its placement in the paper. `FIGURES_BY_MODEL` exposes
both complete sets; `FIGURES` preserves the original assembly choices.
It exports `FIGURES` with `token_embeddings_updates`,
`token_embeddings_updates_si`, and `interp_embeddings` PDFs.

The token workflow preserves the original `plots.jl` model ordering and subsets.
It extracts only the word-embedding tensor plus tokenizer from each staged model
before invoking the existing Julia plotting code. This handles sharded releases
without loading the full model. Extracted tensors and t-SNE coordinates are
explicit directory artifacts. The SI plot now honors its existing `fig_size`
argument; its old implementation ignored that argument.
Token t-SNE uses the existing library defaults without a settings file.
The plotting CLI accepts optional `--settings FILE` overrides for `seed`, `p`,
`maxiter`, `exploreiter`, `exaggeration`, `tol`, and `initialize`.

PAH inference uses each pretrained model's final layer's first-token hidden
state. Both model sizes share prepared public COMPAS cohorts, labels, and split
membership. The embedding width comes from the model. Projections use explicit
UMAP settings and the original `figure_embedding` layout. Linear SVM diagnostics
reuse the existing evaluation/margin helpers with train-only standardization
and the prepared train/validation membership. The historical
`shallow_models.py` demonstration used a different RBF grid search and random
split; the workflow uses the linear SVM specified in the paper methods.

The benchmark Python environment is reused. The Julia environment task
instantiates this project's declared dependencies. Inference is offline and
uses explicit local directories.
The olfaction workflow still owns the separate scent-dendrogram figure.
Isotope plots are separate from this workflow.

## Public PAH cohorts

`embeddings-fetch-compas{1,2,3}` downloads the three public tables pinned in
`compas.json` to COMPAS commit `72e61e3d2bb1a22ecc333ad0e12b92b6ae6b6a04`.
The existing file-download helper verifies their SHA-256 hashes and byte counts.
`embeddings-prepare-pah-cohorts` writes these shared artifacts under
`data/interpretability/pah/`:

| Artifact | Contents |
| --- | --- |
| `aromaticity.csv` | 1,000 fully aromatic and 1,000 other neutral COMPAS-2D molecules, with source IDs, SMILES, benzene counts, labels, and split membership. |
| `condensation.csv` | 1,000 cata-condensed COMPAS-1D and 1,000 peri-condensed COMPAS-3D molecules, with source IDs, SMILES, labels, and split membership. |
| `aromaticity-settings.json`, `condensation-settings.json` | Explicit UMAP and linear-SVM settings shared by both models. |
| `provenance.json` | Source revision/hashes, RDKit version, sampling policy, candidate/selected counts, and excluded overlaps. |

The preparation follows the paper's label rules: aromaticity requires every
heavy-atom bond to satisfy RDKit `GetIsAromatic()`, and condensation uses each
COMPAS dataset's class. Canonical isomeric SMILES remove duplicate graphs;
structures appearing in both condensation classes are excluded. Aromaticity
uses neutral records to avoid repeating a scaffold's charge states.

Sampling and the stratified 80/20 train/validation split use seed 42. UMAP uses
two components, 15 neighbors, minimum distance 0.1, Euclidean distance, and
both seeds set to 42. The linear SVM uses `C=100`. These are explicit new cohorts
and settings authorized for regeneration; the unreleased original sample and
projection are not claimed to be recovered. The same CSVs/settings feed both
MIST sizes. Existing inference, projection, SVM diagnostics, and plotting code
remain in use, with no required `data/user/interpretability` files.

## Models fetched during real execution

All entries are MIST repositories with pinned revisions. Existing benchmark
and tmQM staging tasks are reused; only 28M ToxCast adds a staging task. The
1.8B token panels reuse the corresponding benchmark heads. Their missing ToxCast
head is fitted by the canonical benchmark LR sweep, without adding it to the
24-pair five-seed report. Tokenizers come from the same model directories when
present, or from the matching staged base model when a released head omits its
tokenizer file. There are no implicit extra backbone or
tokenizer downloads, and token extraction does not instantiate models.

| Use | Hugging Face repository | Revision |
| --- | --- | --- |
| 28M pretrained tokens and PAH embeddings | `mist-models/mist-28M-ti624ev1` | `bfeebb7a2619a61d321de922b66376f5cb682eef` |
| 1.8B pretrained tokens and PAH embeddings | `mist-models/mist-1.8B-dh61satt` | `ed6df0875e4425ede9b12019bbd8c2a3c439052d` |
| ToxCast tokens | `mist-models/mist-28M-ttqcvt6fs-toxcast` | `e2728a5218fa2bc426b7cac83228510d7c97fa6c` |
| freesolv tokens | `mist-models/mist-28M-0uiq7o7m-freesolv` | `e38d8a35b3ef273a2ca8ec9c2ff8d014f0033682` |
| qm9 tokens | `mist-models/mist-26.9M-kkgx0omx-qm9` | `65ceeed479609e9dcaef04e687556e2b39e25f23` |
| qm8 tokens | `mist-models/mist-28M-gzwqzpcr-qm8` | `1af94460e96bcfeb796d475c3feadb55e241aa37` |
| lipo tokens | `mist-models/mist-28M-xzr5ulva-lipo` | `ad071251a575a850895e6018ecce382a33cda47c` |
| bace tokens | `mist-models/mist-28M-8loj3bab-bace` | `b2e9d87f6970e1a164bdbff4fdb21922c3ca4a93` |
| esol tokens | `mist-models/mist-28M-kcwb9le5-esol` | `e62653622a4a5efc951128c3652d8850d76f7379` |
| hiv tokens | `mist-models/mist-28M-8fh43gke-hiv` | `d68d5b6c5abbdaf4145aa4d2026318c2108fa7d5` |
| tox21 tokens | `mist-models/mist-28M-kw4ks27p-tox21` | `5eef66e410a59708b133b22c4562416cc2f05819` |
| sider tokens | `mist-models/mist-28M-z8qo16uy-sider` | `28ceeee92e66d33ac8ec06e4c15a1aa8ebcafbce` |
| muv tokens | `mist-models/mist-28M-yr1urd2c-muv` | `333919d2d7232bc931731e0ecdb3ecf728ecaf01` |
| bbbp tokens | `mist-models/mist-28M-3xpfhv48-bbbp` | `f8867105652658ea6d0233aafa24a020a5ed1b21` |
| 1.8B bbbp tokens | `mist-models/mist-1.8B-fbdn8e35-bbbp` | `15773eab5cd6bb9a6de9f7e55643e0ae16beee08` |
| 1.8B hiv tokens | `mist-models/mist-1.8B-1a4puhg2-hiv` | `db87e162b1cc143a9e6d273b2d5f5efc8ffe5d87` |
| 1.8B bace tokens | `mist-models/mist-1.8B-m50jgolp-bace` | `e55a29814a08b0dea84e2dde34c5cf4272610517` |
| 1.8B sider tokens | `mist-models/mist-1.8B-l1wfo7oa-sider` | `5f1aa0c7e3c227f1e623f160694295b181866d91` |
| 1.8B esol tokens | `mist-models/mist-1.8B-hxiygjsm-esol` | `492b867c874317c7b91aca36774561e701400280` |
| 1.8B freesolv tokens | `mist-models/mist-1.8B-iwqj2cld-freesolv` | `7576ca249b027b1e1f1a92af29894e7f41874a08` |
| 1.8B lipo tokens | `mist-models/mist-1.8B-jvt4azpz-lipo` | `abb0fc0a34bc6047b1fcd23b9cbca58ab4735c9b` |
| 1.8B qm8 tokens | `mist-models/mist-1.8B-8nd1ot5j-qm8` | `3cd8c8844b0618111d14f18977976d8566b1293b` |
| 1.8B tox21 tokens | `mist-models/mist-1.8B-uop1z0dc-tox21` | `da3ab8111dcfc2f347635af73eb4e7e2264377ad` |

Both tmQM heads and the 1.8B QM9, MUV, and ToxCast heads are local benchmark
fits from the staged bases. They do not add model repository downloads.

## Checks

`python -m pytest -q test/test_interpretability_workflow.py` verifies sharded
embedding extraction, hidden-width handling, unchanged supplied validation
membership, deterministic public-cohort preparation, and shared model inputs.
It downloads or runs no Hub model. `pomset task show 'embeddings-*' --workflow
opt/interp_embeddings/workflow.pomset --json` checks graph evaluation.
