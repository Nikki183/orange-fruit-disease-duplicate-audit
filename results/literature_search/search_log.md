# Literature search log (novelty check, Introduction paragraph 5; Supplementary Table S1)

All searches run November 6, 2026. Full screening spreadsheet: `novelty_check_citrus_orange.xlsx`.

| Database | Query | Records | Screened |
|---|---|---|---|
| Web of Science Core Collection (TS) | ("citrus" OR "orange") AND "fruit" AND "disease*" AND ("deep learning" OR "CNN" OR "YOLO" OR "convolutional neural network") AND ("duplicate*" OR "leakage" OR "cross-dataset" OR "external validation" OR "independent test") | 3 | 3 |
| Web of Science Core Collection (TS) | same, without the last AND block | 101 | – |
| PubMed (all fields) | same as first query | 1 | 1 |
| Google Scholar S1 | citrus OR orange fruit disease "deep learning" duplicate images | about 18,200 | first 50 |
| Google Scholar S2 | citrus OR orange fruit disease "deep learning" "data leakage" | about 7,980 | first 50 |
| Google Scholar S3 | citrus OR orange fruit disease "deep learning" "external validation" OR "cross-dataset" OR "independent test" | about 15,100 | first 50 |

Full texts read and decision:
- Cited: Ullah et al. 2026 (hash-based train/test duplicate exclusion); Hingmire et al. 2026; Goyal et al. 2026 (external test, contrary evidence); Feng et al. 2023 (external test on other citrus, contrary evidence; abstract says Hainan, Section 3.3 says Yunnan); Chen et al. 2026 (same-fruit grouping); Ijaz et al. 2024 (duplicate removal, method not described); Patil et al. 2026 (cross-dataset drop, PlantVillage/Fruits-360).
- Excluded after full text: Aggarwal & Verma 2025 (multi-fruit; no citrus-specific independent result; no duplicate check); Rajab et al. 2026 (per-dataset 80/20 splits; no cross-dataset test; no duplicate check).
- Excluded on title/snippet: Cai et al. 2024 (laboratory structured-illumination imaging); Sireesha et al. 2026 (external set is leaves); Bhuria et al. 2026 and Kumar et al. 2024 (hybrid CNN-SVM) (internal hold-out test sets).
