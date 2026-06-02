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
  --bert_dir /utilities/models/Bio_ClinicalBERT/ \
  --umn_train_pkl_path /scratch/ahcie-gpu2/XieF-Req04048/z_final/data/preprocessed_2h/train.pkl \
  --train_pkl /scratch/ahcie-gpu2/XieF-Req04048/z_final/mimic_iv_validation/new/splits/mimic_external_train.pkl \
  --val_pkl /scratch/ahcie-gpu2/XieF-Req04048/z_final/mimic_iv_validation/new/splits/mimic_external_val.pkl \
  --test_pkl /scratch/ahcie-gpu2/XieF-Req04048/z_final/mimic_iv_validation/new/splits/mimic_external_test.pkl \
  --mode multimodal \
  --do_finetune --finetune_epochs 3 --lr 1e-5 --batch_size 16 \
  --freeze_text_encoder \
  --save_finetuned_ckpt /scratch/ahcie-gpu2/XieF-Req04048/z_final/mimic_iv_validation/new/out/finetuned_on_mimic.pt \
  --out_dir /scratch/ahcie-gpu2/XieF-Req04048/z_final/mimic_iv_validation/new/out/mimic_finetune_eval
'