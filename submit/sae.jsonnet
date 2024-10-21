{
  container: '/lustre/fs0/awadell/sqsh-files/0535844560745234+mist+08e9e89.sqsh',
  train: {
    tags: ['sae', 'debug'],
    model: {
      class_path: 'electrolyte_fm.models.SAE',
      init_args: {
        sae: 'gated',
        hidden_size: $.train.data.init_args.name_or_path,
        expansion: 16,
        l1_coef: 1e-5,
        optimizer: {
          class_path: 'torch.optim.AdamW',
          init_args: {
            lr: 1e-3,
            betas: [0.0, 0.999],
          },
        },
        lr_schedule: {
          class_path: 'electrolyte_fm.utils.lr_schedule.RelativeCosineWarmup',
          init_args: {
            num_training_steps: 10000,
            num_warmup_steps: 'beta2',
          },
        },

      },
    },
    data: {
      class_path: 'electrolyte_fm.data_modules.HiddenStateDataModule',
      init_args: {
        name_or_path: 'ibm/MoLFormer-XL-both-10pct',
        batch_size: 1024,
        encoder_batch_size: 256,
        val_batch_size: 4 * self.batch_size,
      },
    },
    trainer: {
      max_steps: $.train.model.init_args.lr_schedule.init_args.num_training_steps,
      val_check_interval: 100,
      limit_val_batches: 50,
      precision: 'bf16-true',
      enable_progress_bar: false,
      strategy: "ddp",
    },
  },
  env: {
    TOKENIZER_PARALLELISM: 'true',
  },
}
