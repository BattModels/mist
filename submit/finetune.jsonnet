local pretrain = import 'pretrain.jsonnet';

local molnet_tasks = import 'moleculenet_tasks.libsonnet';

function(dataset='bace') {
  queue: 'venkvis-a100',
  walltime: '1:0:0',
  train: {
    tags: ['finetuning', dataset],
    data: {
      class_path: 'electrolyte_fm.data_modules.PropertyPredictionDataModule',
      init_args: {
        batch_size: 128,
        val_batch_size: 2 * self.batch_size,
        tokenizer: $.train.model.init_args.encoder_ckpt,
        path: '/nfs/turbo/coe-venkvis/mist/molformer_ft_full/' + dataset,
        target_columns: molnet_tasks[dataset].target_columns,
      },
    },
    model: {
      class_path: 'electrolyte_fm.models.LMFinetuning',
      init_args: {
        encoder_ckpt: 'ibm/MoLFormer-XL-both-10pct',
        task: molnet_tasks[dataset].task,
        metrics: molnet_tasks[dataset].metrics,
        freeze_encoder: true,

        // Duplicate pre-training optimizer config
        optimizer: pretrain.train.model.init_args.optimizer,

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
      max_steps: 10000,
      precision: '32-true',
      enable_progress_bar: false,
    },
  },
}
