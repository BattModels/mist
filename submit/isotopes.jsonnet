{
  walltime: '1:0:0',
  nodes: 1,  // Multi-node is not currently supported. Config is only on leader node
  env: {
    TOKENIZERS_PARALLELISM: true,
  },
  train: {
    tags: ['finetuning', 'isotopes', 'thaw'],
    data: {
      class_path: 'electrolyte_fm.data_modules.IsotopeDataModule',
      init_args: {
        path: '/nfs/turbo/coe-venkvis/abhutani/electrolyte-fm/opt/isotope_half_lives.csv',
        batch_size: 16,
        val_batch_size: 2 * self.batch_size,
        tokenizer: $.train.model.init_args.encoder_ckpt,
        target_columns: ['half_life', 'log_time'],
        num_workers: 4,
        prefetch_factor: 8,
      },
    },
    model: {
      class_path: 'electrolyte_fm.models.LMFinetuning',
      init_args: {
        encoder_ckpt: '/nfs/turbo/coe-venkvis/mist/ti624ev1/pretrained/checkpoints/last.ckpt',
        task: 'regression',
        metrics: ['mae', 'mae-channel', 'r2-channel'],
        freeze_encoder: false,
        transform: ['standardize', 'standardize'],
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
      max_steps: 5000,
      precision: 'bf16-true',
      enable_progress_bar: false,
      strategy: 'auto',
      callbacks: [
        {
          class_path: 'electrolyte_fm.utils.progressive_thawing.ProgressiveThawing',
          init_args: {
            initial: ['encoder'],
            stages: [
              ['encoder.embeddings'],
              ['encoder.encoder.layer.7'],
              ['encoder.encoder.layer.6'],
              ['encoder.encoder.layer.5'],
              ['encoder.encoder.layer.4'],
              ['encoder.encoder.layer.3'],
              ['encoder.encoder.layer.2'],
              ['encoder.encoder.layer.1'],
            ],
            stage_duration: 3,
          },
        },
      ],
    },
  },
}
