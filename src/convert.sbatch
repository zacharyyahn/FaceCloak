#!/bin/bash
#SBATCH --job-name=mxnet_to_onnx   # Job name
#SBATCH --output=convert.out  # Standard output and error log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1           # Adjust based on your CPU needs
#SBATCH --mem=32G                   # Adjust memory; export can be heavy
#SBATCH --time=02:00:00             # Max run time

# Load modules if needed
#module load anaconda3

# Activate your isolated conda environment
conda init
conda activate mx_to_onnx

pip install onnx==1.7.0 protobuf==3.17.0

python --version

# Run the conversion
python src/convert_to_onnx.py
