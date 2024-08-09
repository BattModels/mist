local pretrain = import 'pretrain.jsonnet';

local molecularnet_tasks = import 'molecularnet_tasks.libsonnet';

{
  queue: 'venkvis-a100',
  walltime: '1:0:0',
  train: {
    tags: ['finetuning', 'bace'],
    data: {
      class_path: 'electrolyte_fm.data_modules.PropertyPredictionDataModule',
      init_args: {
        batch_size: 512,
        val_batch_size: 2 * self.batch_size,
        path: '/nfs/turbo/coe-venkvis/mist/molformer_ft_full/tox21',
        target_columns: molecularnet_tasks.tox21.target_columns,
      },
    },
    model: {
      class_path: 'electrolyte_fm.models.LMFinetuning',
      init_args: {
        encoder_ckpt: null,  // Replace with checkpoint
        task: molecularnet_tasks.tox21.task,
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
      precision: 'bf16-true',
      enable_progress_bar: false,
    },
  },
}
