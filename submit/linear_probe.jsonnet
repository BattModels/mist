{
  model: {
    model: {
      class_path: 'electrolyte_fm.models.prod_finetune.MISTFinetuned.from_pretrained',
      init_args: {
        name_or_path: 'models/mist-ti624ev1-moleculenet/tmQM',
      },
    },
    probes: {
      class_path: 'electrolyte_fm.models.linear_probe.per_layer_probe',
      init_args: {
        hidden_size: 512,
        features: 5,
        location: 'output',
        n_layers: 8,
      },
    },
  },
  data: {
    class_path: 'electrolyte_fm.data_modules.lipinski_dataset.LipinskiDataModule',
    init_args: {
      name_or_path: 'tox21',
      tokenizer: 'smirk',
      encoding: 'smiles-kekule',
      num_workers: 16,
      batch_size: 16,
    },
  },
}
