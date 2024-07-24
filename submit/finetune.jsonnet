local pretrain = import 'pretrain.jsonnet';

{
  train: {
    tags: ['finetuning'],
    data: {
      class_path: 'electrolyte_fm.data_modules.PropertyPredictionDataModule',
      init_args: {
        batch_size: 128,
        path: 'molformer_ft/',
        num_workers: 1,
        task_specs: [
          { measure_name: 'Class', n_classes: 2 },
        ],
      },
    },
    model: {
      class_path: 'electrolyte_fm.models.LMFinetuning',
      init_args: {
        encoder_ckpt: null,  // Replace with real checkpoint path

        // Duplicate pre-training optimizer config
        optimizer: pretrain.train.model.init_args.optimizer,

        lr_schedule: {
          class_path: 'electrolyte_fm.utils.lr_schedule.RelativeCosineWarmup',
          init_args: {
            num_training_steps: 250000,
            num_warmup_steps: 'beta2',
            rel_decay: 0.1,
          },
        },
      },
    },
  },
}
