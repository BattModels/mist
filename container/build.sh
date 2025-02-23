#!/bin/bash

apptainer build \
    --build-arg SSH_AUTH_SOCK=$SSH_AUTH_SOCK \
    /tmp/mist.sif \
    container/training.def
