#!/bin/bash
#SBATCH --job-name=verona_verify
#SBATCH --partition=lovelace
#SBATCH --array=1-31

ARG_FILE="/home/annelot/VERONA/examples/scripts/OXFORD/args.txt"
PYTHON_SCRIPT="/home/annelot/VERONA/examples/scripts/OXFORD/one_job_iterative.py"

mkdir -p logs

# Extract the line for this task (skip header, so +1 line offset)
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$ARG_FILE")

# Split CSV line into variables
IFS=',' read -r \
  file_verification_context \
  base_path_experiment_repository \
  network_folder \
  experiment_name \
  epsilon_start \
  epsilon_end \
  <<< "$LINE"

echo "Running task $SLURM_ARRAY_TASK_ID"
echo "file_verification_context=$file_verification_context"
echo "base_path_experiment_repository=$base_path_experiment_repository"
echo "network_folder=$network_folder"
echo "experiment_name=$experiment_name"
echo "epsilon_end=$epsilon_start"

# If needed: activate environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

python "$PYTHON_SCRIPT" \
  --file_verification_context "$file_verification_context" \
  --base_path_experiment_repository "$base_path_experiment_repository" \
  --network_folder "$network_folder" \
  --experiment_name "$experiment_name" \
  --epsilon_start "$epsilon_start" \
  --epsilon_end "$epsilon_end"
