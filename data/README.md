# Image data

The images are not stored in this repository. Either download the archive of the exact subsets used from Zenodo ([Zenodo DOI]) and extract it here, or rebuild them from the original sources:

- `data/doh/Orange Fruit Diseases/` – full Orange Fruit Diseases Dataset (https://doi.org/10.17632/6szsnpypdd.1; archive SHA-256 6c0df0b9c1b123a3f11ac841725badb9ee0dee5f0d84dbd66b348aa4e5dc78ec)
- `data/rauf/fruit/` – the 150 fruit images of Rauf et al. (2019), extracted with `code/extract_rauf.py` from Hugging Face `Project-AgML/citrus_fruit_leaf_disease_classification`
- `data/uganda/test300/` and `data/uganda/train/` – Uganda images sampled with `code/prep_uganda.py` and `code/prep_uganda_train.py` from https://doi.org/10.17632/3x8jjd2f72.3 (archive SHA-256 a3fac1155234aa19cb0851cc8771fcae3754b2e0e8d3487934f41be7b645c50d); the images used are listed in `results/uganda_test.csv` and `results/uganda_train.csv`
