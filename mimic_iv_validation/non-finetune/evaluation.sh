#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=slurm_logs/evaluation/evaluation_%j.out
#SBATCH --error=slurm_logs/evaluation/evaluation_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

export APPTAINERENV_TINI_SUBREAPER=1

apptainer exec --nv /utilities/containers/py311-torch-transformers-bits.sif /bin/bash -lc '

source ~/p311-torch/.env/bin/activate
python evaluation.py \
  --ckpt_path /scratch/ahcie-gpu2/XieF-Req04048/z_final/output/multimodal_model_output_2h/best_model.pt \
  --thresholds_path /scratch/ahcie-gpu2/XieF-Req04048/z_final/output/multimodal_model_output_2h/best_thresholds.json \
  --train_pkl /scratch/ahcie-gpu2/XieF-Req04048/z_final/mimic_iv_validation/new/splits/mimic_external_train.pkl \
  --val_pkl /scratch/ahcie-gpu2/XieF-Req04048/z_final/mimic_iv_validation/new/splits/mimic_external_val.pkl \
  --test_pkl /scratch/ahcie-gpu2/XieF-Req04048/z_final/mimic_iv_validation/new/splits/mimic_external_test.pkl \
  --umn_train_pkl_path /scratch/ahcie-gpu2/XieF-Req04048/z_final/data/preprocessed_2h/train.pkl \
  --bert_dir /utilities/models/Bio_ClinicalBERT/ \
  --mode multimodal \
  --batch_size 32 --max_notes 100 --note_chunk_size 32 \
  --num_workers 0 \
  --out_dir /scratch/ahcie-gpu2/XieF-Req04048/z_final/mimic_iv_validation/out/mimic_finetune_eval
'