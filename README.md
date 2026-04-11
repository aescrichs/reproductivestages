# Shifting hierarchies of brain dynamics, hormones, and cognition characterize women's reproductive aging

This repository contains the code supporting the analyses reported in:

> Escrichs et al. *Shifting hierarchies of brain dynamics, hormones, and cognition characterize women's reproductive aging.* Nature Aging (under review).

---

## Repository structure

```
reproductivestages/
│
├── dynamical_complexity/
│   ├── Ignition_SingleSubject.m          # Computes node-metastability per subject
│   ├── demean.m                          # Helper function
│   ├── get_components.m                  # Helper function
│   ├── brain_dynamics_group_comparison.py    # Whole-brain dynamics across reproductive stages (Fig. 1B)
│   ├── brain_dynamics_sex_comparison.py      # Whole-brain dynamics women vs men (Fig. 2)
│   └── brain_dynamics_rsn_comparison.py      # RSN-level dynamics (Supplementary Fig. S2)
│
├── prediction/
│   ├── brain_reproductive_stage_women_prediction.py  # Whole-brain prediction of reproductive stage
│   ├── brain_cognition_prediction.py                 # Prediction of cognitive performance from brain dynamics (Fig. 3)
│   ├── brain_hormones_prediction.py                  # Prediction of FSH and estradiol from brain dynamics (Fig. 1D)
│   └── brain_sex_interaction_prediction.py           # Sex × brain interaction prediction
│
├── classification/
│   ├── ML_multiclass.py                      # Multiclass classification across all 5 reproductive stages
│   └── ML_pairwise.py                        # Pairwise classification across stage transitions (Fig. 4)
│
├── Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv   # Schaefer atlas region names
└── README.md
```

The Schaefer 100-parcel atlas file is publicly available at https://github.com/ThomasYeoLab.

---

## Requirements

- Python 3.12.3
- numpy 1.26.4
- pandas 2.2.1
- scipy 1.12.0
- statsmodels 0.14.1
- scikit-learn 1.5.2
- xgboost 1.7.6
- matplotlib
- seaborn
- imbalanced-learn

Install all dependencies:

```bash
pip install numpy==1.26.4 pandas==2.2.1 scipy==1.12.0 statsmodels==0.14.1 scikit-learn==1.5.2 xgboost==1.7.6 matplotlib seaborn imbalanced-learn
```

MATLAB scripts require MATLAB R2021a or later.

---

## How to run

All scripts should be run from the directory containing the data files.

### Dynamical complexity

```bash
cd dynamical_complexity
python brain_dynamics_group_comparison.py
python brain_dynamics_sex_comparison.py
python brain_dynamics_rsn_comparison.py
```

The MATLAB script `Ignition_SingleSubject.m` computes node-metastability for each subject and should be run first to generate the `.mat` input files.

---

### Prediction

```bash
cd prediction
python brain_reproductive_stage_women_prediction.py
python brain_cognition_prediction.py
python brain_hormones_prediction.py
python brain_sex_interaction_prediction.py
```

**Outputs:**
- `reproductive_stage_results.csv` — Pearson R and p-value
- `cognitive_prediction_results.csv` — R, p, p_FDR per cognitive domain
- `hormones_prediction_results.csv` — R, p, p_FDR for FSH and estradiol
- `figure3.pdf / .png` — scatter observed vs predicted (Fig. 3)
- `figure_hormones_prediction.pdf / .png` — FSH and estradiol prediction (Fig. 1D)
- `figure_sex_interaction.pdf / .png`

---

### Classification

```bash
cd classification
python ML_multiclass.py
python ML_pairwise.py
```

**Outputs:**
- `multiclass_results_YYYYMMDD_HHMMSS.csv` — accuracy, balanced accuracy, F1 per model
- `classification_results_YYYYMMDD_HHMMSS.csv` — AUC, accuracy, F1 per comparison and model
- Feature importance plots (PNG + PDF, Nature Aging style)

---

## Data availability

The HCP-A dataset is publicly available at:
https://www.humanconnectome.org/study/hcp-lifespan-aging

Access requires registration and agreement to the HCP data use terms.
The Schaefer 100-parcel atlas file is publicly available at:
https://github.com/ThomasYeoLab

---

## License

This code is released under the MIT License.
