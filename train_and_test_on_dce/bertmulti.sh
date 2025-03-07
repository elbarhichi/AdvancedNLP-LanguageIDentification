#!/bin/bash
#SBATCH -p gpu_prod_long            # Use the interactive GPU partition
#SBATCH -t 19:00:00             
#SBATCH -J lstone        # Name of the job
#SBATCH -o outputs/output_mt5base_job_%j.out   # Output file (with job ID)

echo "Job started on $(date)"
echo "Running on host: $(hostname)"
echo "GPU status:"
nvidia-smi

echo "Testing Python environment..."
python --version


echo "Training..."
python training_scripts/bertmulti/train_bertmulti.py

echo "Testing..."
python testing_scripts/bertmulti/test_bertmulti.py

echo "Job finished on $(date)"
