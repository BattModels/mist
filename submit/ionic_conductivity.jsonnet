{
  queue: 'venkvis-h100',
  walltime: '8:00:00',
  train: {
    tags: ['finetuning', 'ionic_conductivity'],
    model: {
      class_path: 'electrolyte_fm.models.IonicConductivityModel',
      init_args: {
        n_components: 5,
        encoder_ckpt: '/nfs/turbo/coe-venkvis/mist/atleto2u/checkpoints/last.ckpt',
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
            num_training_steps: 200000,
            num_warmup_steps: 'beta2',
          },
        },

      },
    },
    data: {
      class_path: 'electrolyte_fm.data_modules.ComponentDataModule',
      init_args: {
        path: '/home/abhutani/electrolyte-fm/diffmix_data/mist_aem_processed',
        target_col: 'ln k',
        iterable: true,
        n_components: 5,
        batch_size: 32,
        val_batch_size: 64,
        tokenizer: $.train.model.init_args.encoder_ckpt,
        include_temperature: true,
      },
    },
    trainer: {
      max_steps: $.train.model.init_args.lr_schedule.init_args.num_training_steps,
      enable_progress_bar: true,
      precision: '32',
    },
  },
  env: {
    TOKENIZER_PARALLELISM: 'true',
    CUDA_HOME: '~/.spack/opt/spack/intel-2022.0.2/cuda/12.2.0-kojv/',
    LD_PRELOAD: '~/.spack/opt/spack/gcc-10.3.0/darshan-runtime/3.4.4-4hps/lib/libdarshan.so',
  },
}
