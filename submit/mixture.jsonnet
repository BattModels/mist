{
  queue: 'venkvis-h100,venkvis-a100',
  train: {
    tags: ['finetuning', 'mixture'],
    model: {
      class_path: 'electrolyte_fm.models.MixtureModel',
      init_args: {
        hidden_size: 768,  // d_model = kq size * num_heads
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
            num_training_steps: 200,
            num_warmup_steps: 'beta2',
          },
        },

      },
    },
    data: {
      class_path: 'electrolyte_fm.data_modules.HiddenStateDataModule',
      init_args: {
        name_or_path: '/nfs/turbo/coe-venkvis/mist/q2egf8f2/checkpoints/last.ckpt',
        path: '/home/abhutani/electrolyte-fm/diffmix_data/published_excess_molar_enthalpy',
        batch_size: 4,
        val_batch_size: 2,
        target_col: 'excess_molar_enthalpy/(J/mol)',
        tokenizer: '/nfs/turbo/coe-venkvis/mist/q2egf8f2/checkpoints/last.ckpt',
      },
    },
    trainer: {
      max_steps: $.train.model.init_args.lr_schedule.init_args.num_training_steps,
      enable_progress_bar: true,
    },
  },
  env: {
    TOKENIZER_PARALLELISM: 'true',
    CUDA_HOME: '~/.spack/opt/spack/intel-2022.0.2/cuda/12.2.0-kojv/',
  },
}
