#!/bin/bash
#SBATCH -p gpu_prod_long            # Use the interactive GPU partition
#SBATCH -t 09:00:00             
#SBATCH -J last        # Name of the job
#SBATCH -o outputs/output_xlmrobertalast_job_%j.out   # Output file (with job ID)

echo "Job started on $(date)"
echo "Running on host: $(hostname)"
echo "GPU status:"
nvidia-smi

echo "Testing Python environment..."
python --version

echo "Training..."
python training_scripts/xlmroberta/xlmrobertalast.py

# echo "Testing..."
# python testing_scripts/xlmroberta/testing_xlm.py

echo "Job finished on $(date)"
