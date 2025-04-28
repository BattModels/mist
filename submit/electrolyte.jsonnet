{
  queue: 'venkvis-h100',
  walltime: '8:00:00',
  train: {
    tags: ['finetuning', 'aem'],
    model: {
      class_path: 'electrolyte_fm.models.ElectrolyteModel',
      init_args: {
        n_components: 5,
        target_columns: $.train.data.init_args.target_col,
        transform: [
          'log_transform',
          'standardize',
          'log_transform',
          'standardize',
          'identity',
          'identity',
        ],
        encoder_ckpt: '/nfs/turbo/coe-venkvis/mist/atleto2u/checkpoints/last.ckpt',
        output_size: std.length($.train.data.init_args.target_col),
        optimizer: {
          class_path: 'torch.optim.AdamW',
          init_args: {
            lr: 3e-5,
            weight_decay: 0.01,
          },
        },
        lr_schedule: {
          class_path: 'electrolyte_fm.utils.lr_schedule.RelativeCosineWarmup',
          init_args: {
            num_training_steps: 100000,
            num_warmup_steps: 'beta2',
          },
        },

      },
    },
    data: {
      class_path: 'electrolyte_fm.data_modules.ComponentDataModule',
      init_args: {
        path: '/home/abhutani/electrolyte-fm/diffmix_data/lithium_raw',
        target_col: [
          'density (g/mL)',
          'cP_mean',
          'Cond (mS) 2',
          't+(a)',
          'dissoc (SI)',
          'dissoc (TI)',
        ],
        n_components: 5,
        batch_size: 64,
        randomize: false,
        val_batch_size: 32,
        iterable: true,
        tokenizer: $.train.model.init_args.encoder_ckpt,
        include_temperature: true,
      },
    },
    trainer: {
      max_steps: $.train.model.init_args.lr_schedule.init_args.num_training_steps,
      enable_progress_bar: false,
      limit_val_batches: 20,
      precision: '32',
      strategy: 'auto',
      val_check_interval: 100,
    },
  },
  env: {
    TOKENIZER_PARALLELISM: 'true',
    CUDA_HOME: '~/.spack/opt/spack/intel-2022.0.2/cuda/12.2.0-kojv/',
  },
}
