# === IMPORTS ===
import pandas as pd
import numpy as np
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian, Gamma
from statsmodels.genmod.cov_struct import Exchangeable
import matplotlib.pyplot as plt
import seaborn as sns

# === STEP 1: Load and structure data ===
df = pd.read_csv(r"COMET_result_PTtoEN_with_reference.csv")
df = df["Scale;Translation;Sentence_Score"].str.split(";", expand=True)
df.columns = ["Scale", "Translation", "Sentence_Score"]
df["Sentence_Score"] = pd.to_numeric(df["Sentence_Score"], errors="coerce")

# Create Item_ID
n_systems = df["Translation"].nunique()
n_sentences = len(df) // n_systems
df["Item_ID"] = np.tile(np.arange(1, n_sentences + 1), n_systems)


# Ensure categorical types
df["Translation"] = df["Translation"].astype("category")
df["Scale"] = df["Scale"].astype("category")
df["Item_ID"] = df["Item_ID"].astype("category")


# Set 'Human' as reference for Translation
df["Translation"] = df["Translation"].cat.reorder_categories(
    ["Human", "Azure", "DeepL", "OpenAI", "WidnAI"], ordered=True
)

# === STEP 2: Find best-performing scale and use it as reference ===
ordered_scales = ["DII", "SPAI", "PSDQ","BIS-11", "W-ADL", "SCOFF"]
df["Scale"] = df["Scale"].cat.reorder_categories(ordered_scales, ordered=True)


# === STEP 3: Fit both GEE models (Gaussian and Gamma) ===
gee_gaussian = GEE.from_formula("Sentence_Score ~ Translation + Scale",
                                groups="Item_ID", data=df,
                                family=Gaussian(), cov_struct=Exchangeable())
result_gaussian = gee_gaussian.fit()

gee_gamma = GEE.from_formula("Sentence_Score ~ Translation + Scale",
                             groups="Item_ID", data=df,
                             family=Gamma(), cov_struct=Exchangeable())
result_gamma = gee_gamma.fit()

# === STEP 4: QIC calculation ===
def calculate_qic(model):
    mu = model.fittedvalues
    y = model.model.endog
    deviance = model.family.deviance(y, mu)
    X = model.model.exog
    cov_beta = model.cov_params()
    trace = np.trace(X @ cov_beta @ X.T)
    return deviance + 2 * trace

qic_gaussian = calculate_qic(result_gaussian)
qic_gamma = calculate_qic(result_gamma)

# === STEP 5: Results ===
print("\n=== GEE (Gaussian) Summary ===")
print(result_gaussian.summary())
print(f"\nQIC (Gaussian): {qic_gaussian:.4f}")

print("\n=== GEE (Gamma) Summary ===")
print(result_gamma.summary())
print(f"\nQIC (Gamma): {qic_gamma:.4f}")

# Agrupar por sistema e escala
summary_df = df.groupby(["Translation", "Scale"])["Sentence_Score"].agg(["mean", "sem"]).reset_index()
summary_df.columns = ["Translation", "Scale", "Mean_COMET", "SE_COMET"]

# Garantir ordem consistente para Translation e Scale
translations = summary_df["Translation"].unique()
scales = summary_df["Scale"].unique()
palette_dict = dict(zip(scales, ["#062400", "#437512", "#C3DA8C", "#E5F5B7", "#D4DBB9", "#054823", "#65A756", "#81DB79"]))

# Criar gráfico
plt.figure(figsize=(12, 6))
barplot = sns.barplot(
    data=summary_df,
    x="Translation",
    y="Mean_COMET",
    hue="Scale",
    palette=palette_dict,
    ci=None
)

# Adicionar barras de erro manualmente
for i, (trans, scale) in enumerate(zip(summary_df["Translation"], summary_df["Scale"])):
    mean = summary_df.loc[i, "Mean_COMET"]
    sem = summary_df.loc[i, "SE_COMET"]
    x_pos = list(translations).index(trans)
    hue_idx = list(scales).index(scale)
    total_hue = len(scales)
    offset = -0.4 + (hue_idx + 0.5) * (0.8 / total_hue)
    bar_x = x_pos + offset
    plt.errorbar(
        x=bar_x,
        y=mean,
        yerr=sem,
        fmt='none',
        ecolor='black',
        capsize=4,
        elinewidth=1
    )

# Ajustes de layout
plt.title("COMET Score by Translation and psychological and health-related assessments")
plt.ylabel("COMET Score (A.u)")
plt.xlabel("")
plt.axhline(y=0.940, color='black', linestyle='--', linewidth=2.5)
plt.axhline(y=0.980, color='black', linestyle='--', linewidth=2.5)
plt.ylim(0.0, 1.00)
plt.legend(loc='lower right', bbox_to_anchor=(1.15, -0.05), title=None)
plt.tight_layout()
plt.savefig(r""Figures\COMET_Translation_Scales_PTtoEN.png",
            dpi=600, bbox_inches='tight', transparent=False)
plt.show()



