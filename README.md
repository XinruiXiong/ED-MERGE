# ED-MERGE

**ED-MERGE** is a dynamic multimodal framework for early Emergency Department (ED) risk stratification using strictly post-arrival EHR data.

The framework integrates:

- Unstructured clinical notes
- Dynamic physiological measurements
- Static structured patient features

through a dual-encoder fusion architecture to generate continuously updated risk trajectories across multiple ED outcomes.

<p align="center">
  <img src="figures/framework_overview.png" width="100%">
</p>

Key features:

- Dynamic rolling-horizon risk inference
- BioClinicalBERT-based clinical text modeling
- Multimodal fusion of text + vitals + static features
- Multi-outcome prediction framework
- Internal UMN-ED evaluation and external MIMIC-IV-ED validation
- Integrated Gradients text interpretation

**ED-MERGE** is a multimodal Emergency Department (ED) risk prediction pipeline for time-aware, multi-outcome clinical prediction. The project integrates unstructured clinical notes, dynamic vital signs, and static structured patient information to predict multiple ED-relevant outcomes across different post-arrival observation windows.

The repository includes scripts for multimodal dataset construction, model training, single-modality ablation experiments, text attribution analysis using Integrated Gradients, and external validation on MIMIC-IV-ED with and without site-specific fine-tuning.

---

## Pipeline Overview

```text
Raw EHR Streams
    ├── Clinical Notes
    ├── Vital Signs
    └── Static Patient Features
            ↓
Temporal Window Construction
            ↓
Multimodal Dataset Generation
            ↓
BioClinicalBERT + Structured Encoder
            ↓
Multimodal Fusion
            ↓
Dynamic Multi-Outcome Risk Prediction
            ↓
External Validation + Interpretation
```

All predictions are generated using cumulative post-arrival data available up to the specified ED observation window, with strict prevention of look-ahead bias.

---

## Overview

ED-MERGE is designed to support early ED risk stratification by using all available patient information up to a specified time window after ED arrival. For each encounter, the pipeline constructs a cumulative representation of the patient state using:

- **Clinical notes** available before the observation cutoff
- **Vital signs** recorded before the observation cutoff
- **Static structured patient information**, including comorbidity and chief complaint features
- **Multi-task outcome labels** for ED revisit, hospitalization, critical outcomes, sepsis, cardiopulmonary events, pneumonia, asthma/COPD exacerbations, acute heart failure, pulmonary embolism, and related diagnoses

The central modeling framework uses BioClinicalBERT for text encoding, an MLP encoder for structured features, and a fusion classifier for multi-label prediction.

---

## Repository Structure

```text
ED-MERGE/
├── README.md
├── create_multimodal_dataset.py     # temporal multimodal dataset construction
├── create_multimodal_dataset.sh
├── extract_stay_ids.py              # positive-case extraction for interpretation
├── extract_stay_ids.sh
├── train_multimodal.py              # main multimodal training pipeline
├── train_multimodal.sh
├── single_modal.py                  # single-modality ablation experiments
├── single_modal.sh
├── ig_text.py                       # Integrated Gradients text attribution
├── ig_text.sh
└── mimic_iv_validation/
├── README.md
├── create_multimodal_dataset.py
├── create_multimodal_dataset.sh
├── extract_stay_ids.py
├── extract_stay_ids.sh
├── train_multimodal.py
├── train_multimodal.sh
├── single_modal.py
├── single_modal.sh
├── ig_text.py
├── ig_text.sh
└── mimic_iv_validation/
    ├── non-finetune/
    │   ├── preprocess.py
    │   ├── preprocess.sh
    │   ├── evaluation.py
    │   └── evaluation.sh
    └── finetune/
        ├── preprocess.py
        ├── preprocess.sh
        ├── evaluation.py
        └── evaluation.sh
```

---

## Core Pipeline

### 1. Multimodal Dataset Construction

`create_multimodal_dataset.py` builds the model-ready datasets from the internal ED cohort.

Inputs include:

- A master encounter-level parquet file
- A directory of clinical note parquet files
- A vital-sign file
- Outcome columns and structured features contained in the master dataset

The script performs:

- ID cleaning for `subject_id` and `stay_id`
- Temporal train/test split by encounter year
- Subject-level train/validation split within the training pool
- Time-window filtering for notes and vitals
- Optional diagnosis-time boundary logic using `dx_time`
- Vital-sign aggregation into final window-level features
- Static structured feature extraction
- Train-only standardization of tabular features
- Output of `train.pkl`, `val.pkl`, `test.pkl`, and `scaler.pkl`

The observation cutoff is defined as:

```text
effective_end = min(intime + time_window, dx_time)
```

when `dx_time` is available. Otherwise, the cutoff is `intime + time_window`.

Example run:

```bash
python create_multimodal_dataset.py \
  --master_parquet /path/to/master_dataset_clean.parquet \
  --notes_dir /path/to/notes_all_parts \
  --notes_stay_col SERVICE_ID \
  --notes_time_col FILING_DATE \
  --notes_text_col NOTE_TEXT \
  --vitals_file /path/to/VITALS_SUBSET.tsv \
  --output_dir /path/to/preprocessed_6h \
  --time_window 6h \
  --train_year_le 2022 \
  --test_year_ge 2023 \
  --val_ratio 0.1 \
  --seed 42
```

---

### 2. Multimodal Model Training

`train_multimodal.py` trains the main multimodal model.

The model contains three major components:

1. **Text encoder**
   - BioClinicalBERT encodes each note independently.
   - Multiple notes from the same encounter are aggregated using attention pooling over note-level `[CLS]` embeddings.

2. **Tabular encoder**
   - Structured features are passed through an MLP.
   - These features include dynamic vital signs and static patient profile variables.

3. **Fusion classifier**
   - Text and tabular embeddings are concatenated.
   - A shared classifier predicts all outcomes as a multi-label task.

The model is trained with Focal Loss to address class imbalance.

Example run:

```bash
python train_multimodal.py \
  --preprocessed_dir /path/to/preprocessed_6h \
  --bert_dir /utilities/models/Bio_ClinicalBERT \
  --output_dir /path/to/multimodal_model_output_6h \
  --epochs 4 \
  --max_notes 100 \
  --batch_size 8 \
  --grad_acc 4
```

Main outputs include:

```text
best_model.pt
best_thresholds.json
metrics.json
logits.npy
labels.npy
text_embeddings.npy
structured_embeddings.npy
fused_embeddings.npy
```

---

### 3. Single-Modality and Ablation Experiments

`single_modal.py` supports modality-specific ablation experiments using the same modeling framework.

Supported modes:

```text
multimodal
text_only
vitals_only
profile_only
```

These experiments are used to quantify the contribution of each modality to ED risk prediction.

Example run:

```bash
python single_modal.py \
  --preprocessed_dir /path/to/preprocessed_t0 \
  --bert_dir /utilities/models/Bio_ClinicalBERT \
  --output_dir /path/to/profile_only_t0 \
  --epochs 4 \
  --mode profile_only \
  --batch_size 8
```

---

### 4. Positive Case Sampling

`extract_stay_ids.py` samples positive encounters from `test.pkl` for a selected outcome. This is useful for case-level interpretation and visualization.

Example run:

```bash
python extract_stay_ids.py \
  --preprocessed_dir /path/to/preprocessed_0.5h \
  --outcome_col outcome_all_pne \
  --n 50 \
  --seed 123 \
  --out_txt pos_stayids_all_pne.txt \
  --out_csv pos_stayids_all_pne.csv
```

---

### 5. Text Attribution with Integrated Gradients

`ig_text.py` performs text-only Integrated Gradients attribution for a trained multimodal checkpoint.

The script:

- Loads a trained `best_model.pt`
- Selects a single encounter by `stay_id`
- Selects a prediction target by outcome name or index
- Keeps the tabular branch fixed as a constant baseline
- Computes token-level and sentence-level text attribution
- Produces sentence attribution tables, plots, and HTML note highlights

Example run:

```bash
python ig_text.py \
  --preprocessed_dir /path/to/preprocessed_0.5h \
  --ckpt /path/to/best_model.pt \
  --bert_dir /utilities/models/Bio_ClinicalBERT \
  --split test \
  --stay_id 49320935798 \
  --label outcome_all_pne \
  --max_length 512 \
  --note_chunk_size 32 \
  --steps 50 \
  --baseline mask \
  --top_k 15 \
  --out_dir /path/to/ig_all_pne
```

Outputs include:

```text
sentence_ig.csv
sentence_ig_topk.png
ig_highlight_note*.html
run_summary.json
```

---

## External Validation on MIMIC-IV-ED

The `mimic_iv_validation/` directory contains scripts for external validation on MIMIC-IV-ED. Two settings are supported:

1. **Non-finetune external validation**
2. **Fine-tuned external validation**

Both settings first preprocess the MIMIC data to match the UMN-trained feature space.

---

### MIMIC Preprocessing

The MIMIC preprocessing scripts convert a MIMIC CSV file into model-ready pickle files aligned with the UMN feature order.

The preprocessing step:

- Converts the MIMIC text column to `note_text`
- Wraps each note as a list of strings
- Maps MIMIC triage vitals to the internal `final_*` feature format
- Preserves outcome columns beginning with `outcome_`
- Recovers the tabular feature order from the UMN `train.pkl`
- Applies the UMN-fitted scaler to MIMIC tabular features
- Splits MIMIC encounters into train/validation/test sets by `stay_id`

Example run:

```bash
python preprocess.py \
  --mimic_csv /path/to/master_dataset_with_notes_mimic.csv \
  --umn_scaler_path /path/to/umn/scaler.pkl \
  --umn_train_pkl_path /path/to/umn/train.pkl \
  --output_dir /path/to/mimic_iv_validation/splits \
  --prefix mimic_external \
  --seed 42 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1
```

Generated files:

```text
mimic_external_train.pkl
mimic_external_val.pkl
mimic_external_test.pkl
```

---

### Non-Finetune External Validation

The non-finetune setting directly evaluates a UMN-trained checkpoint on MIMIC-IV-ED without updating model parameters.

This setting is intended to measure direct cross-site generalization.

The script:

- Loads the UMN-trained `best_model.pt`
- Loads UMN thresholds when available
- Aligns MIMIC features to the UMN training feature order
- Concatenates MIMIC train/validation/test splits into one external evaluation set
- Evaluates AUROC, AUPRC, and F1 for each outcome and macro average

Example run:

```bash
python evaluation.py \
  --ckpt_path /path/to/umn/best_model.pt \
  --thresholds_path /path/to/umn/best_thresholds.json \
  --train_pkl /path/to/mimic_external_train.pkl \
  --val_pkl /path/to/mimic_external_val.pkl \
  --test_pkl /path/to/mimic_external_test.pkl \
  --umn_train_pkl_path /path/to/umn/train.pkl \
  --bert_dir /utilities/models/Bio_ClinicalBERT \
  --mode multimodal \
  --batch_size 32 \
  --max_notes 100 \
  --note_chunk_size 32 \
  --out_dir /path/to/non_finetune_eval
```

Outputs:

```text
metrics.json
logits.npy
labels.npy
```

---

### Fine-Tuned External Validation

The fine-tune setting initializes from the UMN-trained checkpoint, adapts the model on MIMIC training data, optionally selects the best state on MIMIC validation data, and evaluates on held-out MIMIC test data.

This setting is intended to measure domain adaptation performance after site-specific fine-tuning.

The script supports:

- Full multimodal fine-tuning
- Text-only, vitals-only, and profile-only fine-tuning
- Optional text encoder freezing with `--freeze_text_encoder`
- Focal Loss for imbalanced multi-label training
- Saving the fine-tuned checkpoint

Example run:

```bash
python evaluation.py \
  --ckpt_path /path/to/umn/best_model.pt \
  --bert_dir /utilities/models/Bio_ClinicalBERT \
  --umn_train_pkl_path /path/to/umn/train.pkl \
  --train_pkl /path/to/mimic_external_train.pkl \
  --val_pkl /path/to/mimic_external_val.pkl \
  --test_pkl /path/to/mimic_external_test.pkl \
  --mode multimodal \
  --do_finetune \
  --finetune_epochs 3 \
  --lr 1e-5 \
  --batch_size 16 \
  --freeze_text_encoder \
  --save_finetuned_ckpt /path/to/finetuned_on_mimic.pt \
  --out_dir /path/to/mimic_finetune_eval
```

Outputs:

```text
metrics.json
test_logits.npy
test_labels.npy
finetuned_on_mimic.pt
```

---

## Prediction Targets

The internal pipeline supports the following outcome columns:

```text
outcome_ed_revisit_3d
outcome_hospitalization
outcome_critical
outcome_sepsis
outcome_copd_exac
outcome_acs_mi
outcome_stroke
outcome_ards
outcome_aki
outcome_bac_pne
outcome_viral_pne
outcome_all_pne
outcome_asthma_exac
outcome_ahf
outcome_copd_asthma
outcome_pe
```

The exact outcome set used during training is stored in each checkpoint as `outcome_cols` and is reused during evaluation.

---

## Input Data Format

### Internal UMN Data

The internal preprocessing script expects:

- Encounter-level master parquet file
- Note parquet directory
- Vital-sign file

Required encounter-level fields include:

```text
subject_id
stay_id
intime
dx_time
outcome_*
```

Notes should contain:

```text
stay_id or SERVICE_ID
filing_date or FILING_DATE
note_text or NOTE_TEXT
```

Vitals should contain:

```text
SERVICE_ID
DISPLAY_NAME
RECORDED_DATETIME
VALUE_ORIG
```

### MIMIC-IV-ED Data

The MIMIC preprocessing script expects a CSV file containing:

```text
stay_id
admission_note_text or note_text or NOTE_TEXT
outcome_*
triage_temperature
triage_heartrate
triage_resprate
triage_o2sat
triage_sbp
triage_dbp
cci_*
eci_*
```

Missing UMN-aligned features are filled with zeros before scaling.

---

## Environment

The provided `.sh` scripts are written for a Slurm-based GPU cluster using Apptainer.

Typical execution environment:

```text
Python 3.11
PyTorch
Transformers
scikit-learn
pandas
numpy
tqdm
Captum
matplotlib
BioClinicalBERT local checkpoint
```

The scripts assume a local BioClinicalBERT directory such as:

```text
/utilities/models/Bio_ClinicalBERT
```

Example Slurm execution:

```bash
sbatch train_multimodal.sh
```

---

## Outputs and Metrics

The training and evaluation scripts report:

- AUROC
- AUPRC
- F1 score using validation-selected or loaded thresholds
- Macro-average metrics across available outcomes
- Per-outcome metrics

Saved outputs commonly include:

```text
metrics.json
best_model.pt
best_thresholds.json
logits.npy
labels.npy
text_embeddings.npy
tabular_embeddings.npy
fused_embeddings.npy
```

---

## Notes on Reproducibility

- Random seeds are exposed through command-line arguments.
- Internal train/test split is based on encounter year.
- Internal validation split is subject-level to reduce leakage.
- MIMIC train/validation/test split is performed by `stay_id`.
- Tabular standardization is fit on the internal training data and reused for external validation.
- The checkpoint stores the outcome column order used during training.

---

## Project Status

This repository currently contains the implementation for:

- Internal multimodal ED prediction
- Temporal observation-window dataset construction
- Single-modality ablations
- Integrated Gradients text attribution
- MIMIC-IV-ED external validation without fine-tuning
- MIMIC-IV-ED external validation with fine-tuning

---

## Contact

For questions about this repository, please contact through xion3020@umn.edu
