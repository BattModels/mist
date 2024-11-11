{
  walltime: '1:0:0',
  nodes: 1,
  gpus_per_node: 1,
  train: {
    tags: ['finetuning', 'tmQM'],
    trainer: {
      max_steps: 100000,
      precision: '32-true',
      enable_progress_bar: false,
      num_nodes: $.nodes,
      devices: $.gpus_per_node,
    },
    data: {
      class_path: 'electrolyte_fm.data_modules.tmQMDataModule',
      init_args: {
        tokenizer: $.train.model.init_args.encoder_ckpt,
        path: 'opt/tmQM/data',
        batch_size: 128,
        val_batch_size: 2 * self.batch_size,
        target_columns: [
          "Electronic_E",
          "Dispersion_E",
          "Dipole_M",
          "Metal_q",
          "HL_Gap",
          "HOMO_Energy",
          "LUMO_Energy",
          "Polarizability",
        ]
      }
    },
    model: {
      class_path: 'electrolyte_fm.models.LMFinetuning',
      init_args: {
        encoder_ckpt: 'ibm/MoLFormer-XL-both-10pct',
        task: "regression",
        metrics: ["mae", "rmse", "r2"],
        freeze_encoder: true,
        output_size: std.length($.train.data.init_args.target_columns),

        // Duplicate pre-training optimizer config
        optimizer: {
          class_path: 'torch.optim.AdamW',
          init_args: {
            lr: 3e-4,
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
    }
  }
}
