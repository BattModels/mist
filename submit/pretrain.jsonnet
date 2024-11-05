{
  train: {
    tags: ['pretraining', 'debug'],
    model: {
      class_path: 'electrolyte_fm.models.RoBERTa',
      init_args: {
        hidden_size: 768,
        intermediate_size: 768,
        num_attention_heads: 12,
        num_hidden_layers: 12,
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
          },
        },

      },
    },
    data: {
      class_path: 'electrolyte_fm.data_modules.RobertaDataSet',
      init_args: {
        batch_size: 128,
        val_batch_size: 2 * self.batch_size,
        tokenizer: 'smirk',
      },
    },
    trainer: {
      max_steps: 50000,
      val_check_interval: 100,
      limit_val_batches: 12,
      precision: 'bf16-true',
      enable_progress_bar: false,
    },
  },
  env: {
    TOKENIZER_PARALLELISM: 'true',
  },
}
