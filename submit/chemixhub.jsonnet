local pretrain = import 'pretrain.jsonnet';

local tasks = import 'chemixhub_tasks.libsonnet';

function(dataset='il_thermo_viscosity') {
  walltime: '2:0:0',
  nodes: 1,  // Multi-node is not currently supported. Config is only on leader node
  env: {
    TOKENIZERS_PARALLELISM: true,
  },
  train: {
    tags: ['chemixub', dataset],
    data: {
      class_path: 'electrolyte_fm.data_modules.ComponentDataModule',
      init_args: {
        path: tasks[dataset].path,
        target_col: tasks[dataset].target_columns,
        n_components: tasks[dataset].n_components,
        batch_size: 2,
        val_batch_size: 2 * self.batch_size,
        tokenizer: $.train.model.init_args.encoder_ckpt,
        include_temperature: tasks[dataset].temperature,
        num_workers: 4,
        prefetch_factor: 8,
        iterable: false,
      },
    },
    model: {
      class_path: 'electrolyte_fm.models.MixtureModel',
      init_args: {
        output_size: std.length($.train.data.init_args.target_col),
        encoder_ckpt: 'ibm/MoLFormer-XL-both-10pct',
        freeze_encoder: false,
        metrics: tasks[dataset].metrics,
        transform: tasks[dataset].transform,
        target_columns: $.train.data.init_args.target_col,
        n_components: $.train.data.init_args.n_components,
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
      precision: '32',
      enable_progress_bar: false,
    },
  },
}
