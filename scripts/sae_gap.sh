#!/bin/bash

# Gaps you want to run
GAPS=(6 3)

for GAP in "${GAPS[@]}"; do
  sbatch \
    --job-name="sae_gap_${GAP}" \
    --output="logs/slurm-%j-gap${GAP}.out" \
    --error="logs/slurm-%j-gap${GAP}.err" \
    <<EOT
#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=100GB
#SBATCH -p gpu-preempt
#SBATCH -G 1
#SBATCH -t 8:00:00
#SBATCH --constraint=a100

module load conda/latest
conda activate finetuning

echo "Running job for sae_gap=${GAP}"
python interp_bottlenecks.py --task sva/rc_train \
    --model_name google/gemma-2-2b \
    --sae_gap ${GAP} \
    --use_mask \
    --use_mean_error \
    --mean_mask \
    --per_token_mask \
    --run_training_thresholds \
    --start_threshold 0.1 \
    --end_threshold 10 \
    --n_threshold_steps 5 \
    --portion_of_data 0.4 
EOT
done
