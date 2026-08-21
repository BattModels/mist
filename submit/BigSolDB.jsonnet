{
  walltime: '4:00:00',
  nodes: 1,
  env: {
    TOKENIZERS_PARALLELISM: true,
  },
  train: {
    tags: ['finetuning', 'bigsol', 'solubility'],
    data: {
      class_path: 'electrolyte_fm.data_modules.CSVDataModule',
      init_args: {
        path: 'BigSolDB_water_room_temp.csv',
        smi_column: 'SMILES',
        target_columns: ['Solubility'],
        encoding: 'smiles-kekule',
        batch_size: 32,
        val_batch_size: 64,
        tokenizer: $.train.model.init_args.encoder_ckpt,
        num_workers: 4,
        prefetch_factor: 8,
      },
    },
    model: {
      class_path: 'electrolyte_fm.models.LMFinetuning',
      init_args: {
        encoder_ckpt: 'mist-models/mist-28M-ti624ev1',
        task: 'regression',
        metrics: ['mae', 'rmse', 'r2'],
        freeze_encoder: false,
        dropout: 0.1,
        output_size: std.length($.train.data.init_args.target_columns),
        target_columns: $.train.data.init_args.target_columns,
        transform: [
          'log_transform',
          'standardize',
        ],

        optimizer: {
          class_path: 'torch.optim.AdamW',
          init_args: {
            lr: 1.6e-4,
            weight_decay: 0.01,
          },
        },

        lr_schedule: {
          class_path: 'electrolyte_fm.utils.lr_schedule.RelativeCosineWarmup',
          init_args: {
            num_training_steps: $.train.trainer.max_steps,
            num_warmup_steps: 'beta2',
            rel_decay: 0.1,
          },
        },
      },
    },
    trainer: {
      max_steps: 50000,
      precision: 'bf16-true',
      enable_progress_bar: false,
    },
  },
}
