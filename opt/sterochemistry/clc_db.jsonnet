{
  queue: 'venkvis-a100,venkvis-h100',
  walltime: '1:0:0',
  nodes: 1,  // Multi-node is not currently supported. Config is only on leader node
  env: {
    TOKENIZERS_PARALLELISM: true,
  },
  train: {
    tags: ['finetuning', 'clc_db'],
    data: {
      class_path: 'electrolyte_fm.data_modules.property_prediction_dataset.HFDataset',
      init_args: {
        name_or_path: "/scratch/venkvis_root/venkvis/awadell/clc_db",
        batch_size: 64,
        val_batch_size: 2 * self.batch_size,
        tokenizer: "smirk-cls",
        smiles_column: "smiles",
        encoding: "smiles",
        randomize: true,
        target_columns: [
            "zero_point_correction",
            "molecular_weight",
            "thermal_correction_energy",
            "thermal_correction_gibbs",
            "thermal_correction_enthalpy",
            "sum_electronic_zero_point",
            "sum_electronic_thermal_energy",
            "sum_electronic_thermal_enthalpy",
            "sum_electronic_thermal_free_energy",
            "homo_energy",
            "lumo_energy",
            "homo_lumo_gap",
        ],
        num_workers: 4,
        prefetch_factor: 8,
      },
    },
    model: {
      class_path: 'electrolyte_fm.models.LMFinetuning.',
      init_args: {
        encoder_ckpt: '/scratch/venkvis_root/venkvis/awadell/models/mist-ti624ev1',
        task: "regression",
        metrics: ["rmse", "mae", "r2"],
        freeze_encoder: false,
        output_size: std.length($.train.data.init_args.target_columns),
        target_columns: $.train.data.init_args.target_columns,

        // Duplicate pre-training optimizer config
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
      max_steps: 100000,
      precision: 'bf16-true',
      enable_progress_bar: false,
    },
  }
}
