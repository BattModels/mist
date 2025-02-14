local pretrain = import 'pretrain.jsonnet';

local molnet_tasks = import 'moleculenet_tasks.libsonnet';

function(dataset='bace') {
  walltime: '1:0:0',
  nodes: 1,  // Multi-node is not currently supported. Config is only on leader node
  env: {
    TOKENIZERS_PARALLELISM: true,
  },
  train: {
    tags: ['finetuning', dataset],
    data: {
      class_path: 'electrolyte_fm.data_modules.MolNetDataModule',
      init_args: {
        name: dataset,
        batch_size: 128,
        val_batch_size: 2 * self.batch_size,
        tokenizer: $.train.model.init_args.encoder_ckpt,
        target_columns: molnet_tasks[dataset].target_columns,
        num_workers: 4,
        prefetch_factor: 8,
      },
    },
    model: {
      class_path: 'electrolyte_fm.models.LMFinetuning',
      init_args: {
        encoder_ckpt: 'ibm/MoLFormer-XL-both-10pct',
        task: molnet_tasks[dataset].task,
        metrics: molnet_tasks[dataset].metrics,
        freeze_encoder: true,
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
      max_steps: 100000,
      precision: 'bf16-true',
      enable_progress_bar: false,
    },
  },
}
