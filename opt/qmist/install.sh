#!/bin/bash
# Best effort install script to get job dependencies installed
set -ex

if ! command -v uv &> /dev/null; then
    echo "'uv' not found. Installing using Astral's install script..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    if command -v uv &> /dev/null; then
        echo "'uv' successfully installed."
    else
        echo "Failed to install 'uv'." >&2
        exit 1
    fi
else
    echo "'uv' is already installed."
fi

uv sync

# Install MOPAC
if [[ -d vendor/mopac* ]]; then
    wget https://github.com/openmopac/mopac/releases/download/v23.1.2/mopac-23.1.2-linux.tar.gz
    tar -xvf mopac-23.1.2-linux.tar.gz -C vendor
    rm -f mopac-23.1.2-linux.tar.gz
    echo "Installed MOPAC."
else
    echo "MOPAC already installed."
fi
