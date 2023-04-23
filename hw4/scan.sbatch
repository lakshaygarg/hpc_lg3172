#!/bin/bash 
#SBATCH --nodes=100                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=1:00:00
#SBATCH --mem=5GB
#SBATCH --job-name=scan
#SBATCH --output=scan.out

module purge
module load intel/19.1.2 openmpi/intel/4.0.5

mpirun -np 100 ./scan