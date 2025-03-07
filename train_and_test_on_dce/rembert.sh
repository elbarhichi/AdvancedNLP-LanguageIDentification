#!/bin/bash
#SBATCH -p gpu_prod_long            # Use the interactive GPU partition
#SBATCH -t 01:00:00             # Set a 10-minute time limit
#SBATCH -J rembert2test        # Name of the job
#SBATCH -o outputs/output_rembert2test_job_%j.out   # Output file (with job ID)

echo "Job started on $(date)"
echo "Running on host: $(hostname)"
echo "GPU status:"
nvidia-smi

echo "Testing Python environment..."
python --version

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# echo "Training..."
# python training_scripts/rembert/rembert3.py

echo "Testing..."
python testing_scripts/rembert/testing_rembert2.py

echo "Job finished on $(date)"
