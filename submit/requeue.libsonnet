{
  // Flag to enable requeue sections of the submission script
  requeue: true,
  train: {
    trainer: {
      plugins: [
        {
          // Disable SLURMEnvironment's requeuing
          class_path: 'lightning.pytorch.plugins.environments.SLURMEnvironment',
          init_args: {
            auto_requeue: false,
          },
        },
      ],
      callbacks: [
        { class_path: 'electrolyte_fm.utils.callbacks.Requeue' },
      ],
    },
  },
}
