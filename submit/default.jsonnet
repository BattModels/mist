{
  env: {
    // Set cache directories to the working directory
    TORCH_EXTENSIONS_DIR: '${PWD}/.cache/torch_extensions',
    HF_HOME: '${PWD}/.cache/huggingface',
  },
}
