{
  container: '/lustre/fs0/shared/sqsh-files/mist+pytorch+25.01+v2.sif',
  train: {
    tags: ['sae', 'debug'],
    model: {
      class_path: 'electrolyte_fm.models.LightningSAE',
      init_args: {
        name_or_path: 'ibm/MoLFormer-XL-both-10pct',
        sae_type: 'gated',
        expansion: 4,
        l1_coef: 1e-5,
        optimizer: {
          class_path: 'torch.optim.AdamW',
          init_args: {
            lr: 5e-4,
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
      class_path: 'electrolyte_fm.data_modules.RobertaDataSet',
      init_args: {
        path: '/lustre/fs0/awadell/realspace',
        batch_size: 128,
        val_batch_size: 4 * self.batch_size,
        num_workers: 8,
        prefetch_factor: 8,
      },
    },
    trainer: {
      max_steps: $.train.model.init_args.lr_schedule.init_args.num_training_steps,
      val_check_interval: 100,
      limit_val_batches: 50,
      precision: '32',
      enable_progress_bar: false,
      strategy: 'auto',
    },
  },
  env: {
    TOKENIZER_PARALLELISM: 'true',
  },
}
