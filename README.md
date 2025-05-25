# Translation Quality Evaluation with COMET and Statistical Modeling using GEE

This repository contains a full pipeline for evaluating the quality of machine and human translations between English and Brazilian Portuguese (in both directions: EN→PT and PT→EN). The evaluation is performed using the COMET model (`Unbabel/XCOMET-XL`), and results are analyzed statistically using Generalized Estimating Equations (GEE) to compare translation systems and psychometric scales.

---

## Structure and Execution Order

### 1. Machine Translation

These scripts use commercial APIs (Azure, DeepL, OpenAI, Widn.AI) to generate translations:

* **`Machine_Translation_ENtoPT.py`**
  Performs forward translations (English to Portuguese). The output is saved in `combined_translations.csv`.
  ⚠️ API keys must be configured as environment variables.

* **`Machine_Translation_PTtoEN.py`**
  Performs back-translations (Portuguese to English), using the published Portuguese version as the source.
  Outputs `combined_back_translations.csv`.
  ⚠️ Requires the same API configuration.

---

### 2. COMET Evaluation

These scripts assess translation quality by comparing each MT output with a human reference using the COMET model:

* **`COMET_ENtoPT_analysis_with_reference.py`**
  Evaluates EN→PT translations using the published Portuguese version as reference.
  Outputs `COMET_result_ENtoPT_with_reference.csv`.

* **`COMET_PTtoEN_analysis_with_reference.py`**
  Evaluates PT→EN back-translations using the original English version as reference.
  Outputs `COMET_result_PTtoEN_with_reference.csv`.

---

### 3. Statistical Analysis (GEE)

These scripts apply Generalized Estimating Equations to analyze COMET scores across translation systems and scales:

* **`GEE_ENtoPT.py`**
  Fits GEE models (Gaussian and Gamma) for forward translations.
  Produces summary tables and bar plots saved in the `Figures/` folder.

* **`GEE_PTtoEN.py`**
  Performs the same analyses for back-translations.
  Also generates graphs and statistical summaries.

---

## Technical Notes

* COMET evaluation is performed using the `Unbabel/XCOMET-XL` model.
* The `contractions` library is used to normalize text inputs before scoring.
* GEE models account for repeated measures across items and control for scale effects.
* Resulting figures show system-level comparisons and COMET quality thresholds (e.g., >0.94 = "Good").

---

## Requirements

To run the pipeline, install the required packages:

```bash
pip install pandas numpy seaborn matplotlib statsmodels openai deepl contractions tqdm
```

Also, configure the following environment variables with your API keys:

```
AZURE_API_KEY  
DEEPL_API_KEY  
OPENAI_API_KEY  
WIDN_API_KEY  
```

## Author

**Maicon Rodrigues Albuquerque**
Universidade Federal de Minas Gerais (UFMG)

