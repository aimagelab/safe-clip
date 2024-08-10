#!/bin/bash
#SBATCH --job-name=safeclip-training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --output=<out_log_file.out>
#SBATCH --error=<err_log_file.err>
#SBATCH --mem=64G
#SBATCH --account=<slurm_account>
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate sc
cd <project_folder>

srun --exclusive python -u safeclip_training.py \
			--wandb-activated \
			--wandb-config "{'project': '<wandb-project-name>', 'name': '<wandb-run-name>', 'entity': '<wandb-entity>'}" \
			--lambdas "(0.1,0.1,0.1,0.2,0.25,0.25,0.25,0.5)" \
			--bs 16 \
			--visu-dataset-root <visu_dataset_root> \
			--coco-dataset-root <coco_dataset_root> \
			--checkpoint-saving-root <checkpoint_saving_root> \