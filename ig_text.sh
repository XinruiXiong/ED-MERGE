#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=slurm_logs/ig_text/ig_text_%j.out
#SBATCH --error=slurm_logs/ig_text/ig_text_%j.err
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

python ig_text.py \
  --preprocessed_dir /scratch/ahcie-gpu2/XieF-Req04048/z_final/data/preprocessed_0.5h \
  --ckpt /scratch/ahcie-gpu2/XieF-Req04048/z_final/output/multimodal_model_output_0.5h/best_model.pt \
  --bert_dir /utilities/models/Bio_ClinicalBERT \
  --split test \
  --stay_id 49320935798 \
  --label outcome_all_pne \
  --max_length 512 \
  --note_chunk_size 32 \
  --steps 50 \
  --baseline mask \
  --top_k 15 \
  --out_dir /scratch/ahcie-gpu2/XieF-Req04048/z_final/output/ig_all_pne
'