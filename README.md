# Translation Quality Evaluation with COMET and Statistical Modeling using GEE

This repository contains a complete pipeline for evaluating the quality of machine and human translations between English and Brazilian Portuguese (in both directions: EN→PT and PT→EN). The evaluation is conducted using the COMET model (`Unbabel/XCOMET-XL`), and statistical modeling is applied using Generalized Estimating Equations (GEE) to compare translation systems and psychological/health-related scales.

---

## 📁 Repository Structure and Execution Order

### 1. `MT_Code/` — Machine Translation

These scripts use commercial APIs (Azure, DeepL, OpenAI, Widn.AI) to generate automatic translations:

* **`Machine_Translation_ENtoPT.py`**
  Translates from English to Portuguese.
  ➤ Output: `combined_translations.csv`

* **`Machine_Translation_PTtoEN.py`**
  Translates from Portuguese to English (back-translation).
  ➤ Output: `combined_back_translations.csv`

⚠️ These scripts require API keys set as environment variables (`AZURE_API_KEY`, `DEEPL_API_KEY`, `OPENAI_API_KEY`, `WIDN_API_KEY`).

---

### 2. `COMET_Analysis/` — Translation Evaluation with COMET

These scripts assess the quality of the translations using COMET, comparing them against human references:

* **`COMET_ENtoPT_analysis_with_reference.py`**
  Evaluates EN→PT translations using the published Portuguese version as reference.
  ➤ Output: `COMET_result_ENtoPT_with_reference.csv`

* **`COMET_PTtoEN_analysis_with_reference.py`**
  Evaluates PT→EN back-translations using the original English version as reference.
  ➤ Output: `COMET_result_PTtoEN_with_reference.csv`

---

### 3. `GEE_Analysis/` — Statistical Analysis (GEE)

These scripts apply Generalized Estimating Equations (GEE) to analyze COMET scores across systems and instruments:

* **`GEE_ENtoPT.py`**
  Statistical analysis of COMET results for EN→PT translations.
  ➤ Outputs plots to `Figures/COMET_Translation_Scales_ENtoPT.png`

* **`GEE_PTtoEN.py`**
  Statistical analysis for PT→EN back-translations.
  ➤ Outputs plots to `Figures/COMET_Translation_Scales_PTtoEN.png`

Both scripts fit Gaussian and Gamma GEE models and print QIC model comparison metrics.

---

### 4. `Files/` — Input and Output Data

This folder is intended to contain all `.csv` files used as input/output across stages, such as:

* `combined_translations.csv`
* `combined_back_translations.csv`
* `COMET_result_ENtoPT_with_reference.csv`
* `COMET_result_PTtoEN_with_reference.csv`

---

## ⚙️ Requirements

Install the required packages:

```bash
pip install pandas numpy seaborn matplotlib statsmodels openai deepl contractions tqdm
```

Set the following environment variables with your API keys before running the MT scripts:

```
AZURE_API_KEY  
DEEPL_API_KEY  
OPENAI_API_KEY  
WIDN_API_KEY  
```

## 🧠 Author

**Maicon Rodrigues Albuquerque** - Universidade Federal de Minas Gerais (UFMG)

