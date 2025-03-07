#!/bin/bash
#SBATCH -p cpu_prod_long            # Use the interactive GPU partition
#SBATCH -t 19:30:00             # Set a 10-minute time limit
#SBATCH -J fasttext        # Name of the job
#SBATCH -o outputs/output_fasttext_job_%j.out   # Output file (with job ID)

echo "Job started on $(date)"
echo "Running on host: $(hostname)"
echo "GPU status:"
nvidia-smi

echo "Testing Python environment..."
python --version

python training_scripts/fasttext/train_fasttext.py

echo "FNSH"

echo "Job finished on $(date)"
