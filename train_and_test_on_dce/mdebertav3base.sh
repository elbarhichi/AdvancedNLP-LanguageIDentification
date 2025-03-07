#!/bin/bash
#SBATCH -p gpu_prod_long            # Use the interactive GPU partition
#SBATCH -t 01:00:00             
#SBATCH -J mdebertav3base        # Name of the job
#SBATCH -o outputs/output_mdebertav3base_job_%j.out   # Output file (with job ID)

echo "Job started on $(date)"
echo "Running on host: $(hostname)"
echo "GPU status:"
nvidia-smi

echo "Testing Python environment..."
python --version

echo "Training..."
python training_scripts/mdebertav3/mdebertav3base.py

echo "Testing..."
python testing_scripts/mdebertav3/testing_mdebertav3base.py

echo "Job finished on $(date)"
