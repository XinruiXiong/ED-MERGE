#!/bin/bash

#SBATCH --job-name=bash
#SBATCH --output=slurm_logs/single_modal/single_modal_%j.out
#SBATCH --error=slurm_logs/single_modal/single_modal_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4          
#SBATCH --mem=64G

export APPTAINERENV_TINI_SUBREAPER=1

apptainer exec --nv /utilities/containers/py311-torch-transformers-bits.sif /bin/bash -lc '

source ~/p311-torch/.env/bin/activate
python single_modal.py \
  --preprocessed_dir "/scratch/ahcie-gpu2/XieF-Req04048/z_final/data/new/preprocessed_t0" \
  --bert_dir "/utilities/models/Bio_ClinicalBERT" \
  --output_dir "/scratch/ahcie-gpu2/XieF-Req04048/z_final/output/single_modal/profile_only_t0" \
  --epochs 4 \
  --mode profile_only \
  --batch_size 8 \
'