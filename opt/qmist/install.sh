#!/bin/bash
# Best effort install script to get job dependencies installed

uv sync

wget https://github.com/openmopac/mopac/releases/download/v23.1.2/mopac-23.1.2-linux.tar.gz
tar -xvf mopac-23.1.2-linux.tar.gz -C vendor
