{
  container: '/lustre/fs0/shared/sqsh-files/mist+pytorch+25.01+v2.sif',
  program: '-m electrolyte_fm.models.token_level',
  nodes: 1,
  gpus_per_node: 2,
  train: {
    tags: ['debug'],
    model: {
      class_path: 'electrolyte_fm.models.token_level.TokenLevelPredictor',
      init_args: {
        freeze_encoder: false,
        encoder: {
          class_path: 'electrolyte_fm.models.lm_finetuning.load_encoder',
          init_args: {
            encoder: '/lustre/fs0/awadell/electrolyte-fm/sae/models/mist-ti624ev1-moleculenet/pretrained',
          },
        },
        seq_network: {
          class_path: 'electrolyte_fm.models.prediction_task_head.PredictionTaskHead',
          init_args: {
            embed_dim: 512,
            output_size: 9,
          },
        },
        token_network: {
          class_path: 'electrolyte_fm.models.prediction_task_head.TokenTaskHead',
          init_args: {
            embed_dim: 512,
            output_size: 4,
          },
        },
        seq_transform: {
          class_path: 'electrolyte_fm.models.normalize.AbstractNormalizer.get',
          init_args: {
            transform: 'standardize',
            num_outputs: $.train.model.init_args.seq_network.init_args.output_size,
          },
        },
        token_transform: {
          class_path: 'electrolyte_fm.models.normalize.AbstractNormalizer.get',
          init_args: {
            transform: 'standardize',
            num_outputs: $.train.model.init_args.token_network.init_args.output_size,
          },
        },
        optimizer: {
          class_path: 'torch.optim.AdamW',
          init_args: {
            lr: 1.6e-4,
            betas: [0.9, 0.999],
            weight_decay: 0.01,
          },
        },
        lr_schedule: {
          class_path: 'electrolyte_fm.utils.lr_schedule.RelativeCosineWarmup',
          init_args: {
            num_training_steps: 50000,
            num_warmup_steps: 'beta2',
            rel_decay: 0.1,
          },
        },
      },
    },
    data: {
      class_path: 'electrolyte_fm.data_modules.pubchem_qc.PubChemQC',
      init_args: {
        path: '/lustre/fs0/shared/pubchemqc_jcim2017-split',
        tokenizer: 'smirk-cls',
        batch_size: 128,
        val_batch_size: 2 * self.batch_size,
        num_workers: 12,
        prefetch_factor: 4,
      },
    },
    trainer: {
      num_nodes: $.nodes,
      devices: $.gpus_per_node,
      max_steps: $.train.model.init_args.lr_schedule.init_args.num_training_steps,
      precision: 'bf16-mixed',
      accumulate_grad_batches: 1,
      enable_progress_bar: false,
      gradient_clip_val: 1,
      gradient_clip_algorithm: 'norm',
      val_check_interval: 128 * self.accumulate_grad_batches,
      limit_val_batches: 16,
    },
  },
}
