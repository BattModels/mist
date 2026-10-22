Code for generating ready-to-train datasets for MIST from [Omol25](https://huggingface.co/facebook/OMol25) and reproducing stereochemistry plots in paper.

`paper_workflow.pomset` adds the original paper application panels without training
another model. It reuses the existing `tmqm-mist-28m-release-model` directory
(`mist-models/mist-28M-ggd8iisr-tmQM` at
`e0587490dd35dd11c0bbba35f6a485b8eaa74479`) and the released validation split.
The original predictor defaults to that 10,157-row split; all four annotated
examples, LEYLAC, LIMTIK, ROMVAQ and ROMVEU, occur there and are absent from the
test split. With the current RDKit grouping it contains 40 members of 19
stereoisomer groups, including the three scaffolds in the original gap bars.

`tmqm-paper-validation-predictions` retains the original molecule IDs, SMILES,
reference properties and `<property>_mist` predictions in JSONL. The figure task
calls the existing `tmqm.jl` functions to produce `tmQM.pdf/png` (the main parity
panel), `tmQM-isomer-gaps.pdf/png` and `tmQM-complexes.pdf/png`, plus the selected
pair and annotated-example JSONL tables. Input/reference gaps stay in Hartree;
the original plotting functions convert them to eV. No additional user-supplied
paper inputs or HF repositories are needed for these panels. The existing
ten-backbone training/selection benchmark remains a separate workflow branch.

ChemBERTa-v3's benchmark model artifact includes the separately pinned MolFormer
Python code. The packaging task changes only cross-repository `auto_map`
references to local module names and links the original model files. Training
and testing load that directory, so Pomset export/import does not depend on an
unexported Hugging Face cache ref. Model weights and scientific settings are
unchanged.

Implementation checks exercised the predictor using a tiny locally initialized
MIST task model and temporary Arrow data, including ID/reference preservation and
local-only loading. The actual Julia CLI rendered all three figures and the two
tables using the released validation cohort with synthetic prediction columns
in a temporary directory; those smoke artifacts are not paper predictions.
The selected cohort contained the expected 40 rows and four annotated IDs. The
full released model inference has not been run as part of this implementation.
