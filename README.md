# Orange fruit disease classifiers: duplicate audit and cross-source evaluation

Code and results for: Duvvuru, D. R., & Kottapalli, N. *Duplicated images and peel-colour shortcuts inflate the accuracy of YOLO classifiers for orange fruit diseases* (manuscript submitted to *Biosystems Engineering*).

The repository contains every script used for the duplicate audit, the split definitions, model training and evaluation, statistics and figures, together with the per-image predictions and all result tables. Trained models and the image subsets used are archived separately (see `models/README.md` and `data/README.md`).

## Contents
- `code/` – analysis scripts (Python 3.9)
- `results/` – duplicate audit (`doh_audit.json`), analysed image list with duplicate-group IDs (`dataset4.csv`), split definitions (`splits.json`), Uganda test and training lists, per-image predictions (`preds/`), result tables (`tables/`), figures (`figs_original/`), literature search log (`literature_search/`)
- `data/` – place the image data here (see `data/README.md`)
- `models/` – place the trained models here (see `models/README.md`)
- `requirements.txt` – exact package versions

## Reproducing the analysis
Run all commands from the repository root after installing the requirements (`pip install -r requirements.txt`) and placing the data in `data/`.

1. Duplicate audit and splits: `code/extract_rauf.py`, `code/audit.py`, `code/embed_check.py`, `code/build_splits.py`, `code/build_hash_splits.py`
2. Training on the public collection (4 architectures × 15 splits): `code/run_exp.py results <work_dir> <model> <split> 30`
3. Uganda test set and cross-source evaluation: `code/prep_uganda.py`, `code/eval_external.py`
4. Models trained on Uganda images: `code/prep_uganda_train.py`, `code/run_uganda_trained.py`
5. Probes: `code/occlusion_probe.py`, `code/hue_shift_probe.py`
6. Statistics and figures: `code/analysis.py`, `code/analysis_hashgroup.py`, `code/metrics_mcc_auc.py`, `code/figs_extra.py`, `code/figs_split_fig1.py`, `code/fig_transfer.py`

Image paths in the CSV files are relative to the repository root.

## Data sources (CC BY 4.0)
- Doh, B. (2025). Orange Fruit Diseases Dataset. Mendeley Data, v1. https://doi.org/10.17632/6szsnpypdd.1
- Rauf, H. T., et al. (2019). A citrus fruits and leaves dataset. Mendeley Data, v2. https://doi.org/10.17632/3f83gxmv57.2
- Kusemererwa, D., Sserwadda, A., & Idid, M. (2025). Orange fruits Dataset. Mendeley Data, v3. https://doi.org/10.17632/3x8jjd2f72.3

## License
Code: MIT License (see `LICENSE`). Result files: CC BY 4.0. Image data remain under the licences of their original authors.
