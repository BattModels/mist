{
  program: '-m electrolyte_fm.models.token_level',
  nodes: 1,
  gpus_per_node: 1,
  stage: null,
  train: {
    tags: ['more-layers', 'mlp-pairwise'],
    model: {
      class_path: 'electrolyte_fm.models.token_level.TokenLevelDist',
      init_args: {
        freeze_encoder: false,
        procrustes_loss: false,
        rescale_dist_matrix: false,
        encoder: {
          class_path: 'electrolyte_fm.models.lm_finetuning.load_encoder',
          init_args: {
            encoder: '/lustre/fs0/awadell/electrolyte-fm/sae/models/mist-ti624ev1-moleculenet/pretrained',
            // encoder: '/scratch/venkvis_root/venkvis/awadell/ti624ev1/pretrained',
          },
        },
        seq_network: {
          class_path: 'electrolyte_fm.models.prediction_task_head.PredictionTaskHead',
          init_args: {
            embed_dim: 512,
            output_size: 9,
          },
        },
        seq_transform: {
          class_path: 'electrolyte_fm.models.normalize.AbstractNormalizer.get',
          init_args: {
            transform: 'power_transform',
            num_outputs: $.train.model.init_args.seq_network.init_args.output_size,
          },
        },
        token_network: {
          class_path: 'electrolyte_fm.models.prediction_task_head.TokenTaskHead',
          init_args: {
            embed_dim: 512,
            output_size: 4,
          },
        },
        token_transform: {
          class_path: 'electrolyte_fm.models.normalize.AbstractNormalizer.get',
          init_args: {
            transform: 'standardize',
            num_outputs: $.train.model.init_args.token_network.init_args.output_size,
          },
        },
        dist_network: {
          class_path: 'electrolyte_fm.models.prediction_task_head.TokenPairwiseDistance',
          init_args: {
            embed_dim: 512,
            activation: 'gelu',
            ff_ratio: 2,
            num_attention_heads: 8,
            num_layers: 1,
          },
        },
        optimizer: {
          class_path: 'deepspeed.ops.lamb.FusedLamb',
          init_args: {
            lr: 1.6e-4,
            betas: [0.87, 0.997],
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
    data: {
      class_path: 'electrolyte_fm.data_modules.pubchem_qc.PubChemQC',
      init_args: {
        path: 'opt/pubchem-qc/pubchemqc_jcim2017-split-v2',
        tokenizer: 'smirk-cls',
        batch_size: 64,
        val_batch_size: self.batch_size,
        num_workers: 8,
        prefetch_factor: 2,
        include_3d: 'as-target',
        include_topo_diameter: $.train.model.init_args.rescale_dist_matrix,
      },
    },
    trainer: {
      num_nodes: $.nodes,
      devices: $.gpus_per_node,
      // strategy: 'deepspeed',
      max_steps: 100000,
      precision: 'bf16-mixed',
      accumulate_grad_batches: 1,
      val_check_interval: 100,
      limit_val_batches: 12,
      enable_progress_bar: false,
    },
  },
  env: {
    PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True',
  },
}
