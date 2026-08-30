# BigSolDB Token Distribution

Compares the SMILES token distribution of the BigSolDB finetuning set against
the Enamine REAL Space pretraining corpus.

## Preprocessing

```shell
python prepare.py "$DATA/BigSolDB.csv"
```

Filters raw BigSolDB to one solvent and temperature window, keeps one row per
molecule, and adds `logS`.

## Token Counting

```shell
python token_counts.py csv "$DATA/dataset.csv" \
    --encoding smiles-kekule

sbatch submit_counts.sh
```

The pretraining config applies no re-encoding, so the corpus is counted as
stored.

## Plotting

```shell
python plot_tokens.py histogram \
    "$DATA/dataset.csv" \
    "$DATA/pretraining-dataset.csv" \
    --labels BigSolDB --labels "REAL Space" --log \
    --output "$DATA/token_histograms.png"

python plot_tokens.py compare \
    "$DATA/BigSolDB_water_room_temp_unique_tokens.csv" \
    "$DATA/pretraining-dataset.csv" \
    --label-a BigSolDB --label-b "REAL Space" \
    --output "$DATA/token_comparison.png"
```

`histogram` draws one panel per dataset on a shared token axis; `compare` draws
a share-of-tokens dumbbell and the log2 enrichment of A over B.
