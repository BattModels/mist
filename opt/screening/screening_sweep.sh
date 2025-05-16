#!/bin/bash

sbatch -N 1 --ntasks-per-node 1 --gpus-per-node 1 launch.sh
sbatch -N 1 --ntasks-per-node 2 --gpus-per-node 2 launch.sh
sbatch -N 1 --ntasks-per-node 4 --gpus-per-node 4 launch.sh
sbatch -N 1 launch.sh
sbatch -N 2 launch.sh
sbatch -N 3 launch.sh
sbatch -N 4 launch.sh
sbatch -N 5 launch.sh

