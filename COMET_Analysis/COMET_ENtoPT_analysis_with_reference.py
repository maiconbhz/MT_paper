import pandas as pd
import contractions
from comet import download_model, load_from_checkpoint
import time
from tqdm import tqdm

# Record the start time of the entire process
start_time = time.time()

# ------------------ Data Loading ------------------
# Define the path to the CSV file
csv_path = r'combined_translations_ENtoPT.csv'

# Read the CSV file with specified delimiter and encoding
df = pd.read_csv(csv_path, delimiter=";", encoding="utf-8-sig")
df = pd.DataFrame(df)  # Ensure the data is in a DataFrame

# ------------------ Model Setup ------------------
# Download and load the COMET model for translation evaluation
model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)

def evaluate_translations_with_reference(model, src_list, mt_list, ref_list):
    """Evaluates machine translations against human references using COMET."""
    # Prepare the input data by combining source, machine translation, and reference texts
    data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(src_list, mt_list, ref_list)]
    # Get COMET scores for the batch of translations
    model_output = model.predict(data, batch_size=15)
    return model_output

def get_discrete_quality_score(score):
    """Classifies translation quality into discrete categories based on the COMET score."""
    if score <= 0.600:
        return 'Weak'
    elif score <= 0.800:
        return 'Moderate'
    elif score <= 0.940:
        return 'Good'
    elif score <= 0.980:
        return 'Excellent'
    else:
        return 'Optimal'

# ------------------ Data Preprocessing ------------------
# Expand contractions in the "Original" text column
df['Original'] = df['Original'].apply(lambda x: contractions.fix(str(x)))

# Convert the necessary columns to lists for processing
Original = df["Original"].tolist()
Reference = df["Published_PT"].tolist()

# Define the list of translation models to be evaluated
translation_models = ["Azure", "DeepL", "OpenAI", "WidnAI","Profissional_Translation_ENtoPT"]

# ------------------ COMET Evaluation without Reference ------------------
# Initialize a dictionary to store evaluation results without reference
results_with_ref = {}
#model, src_list, mt_list, ref_list
# Loop through each translation model and evaluate its translations without a human reference
# Loop through each translation model and evaluate its translations without a human reference
for model_name in tqdm(translation_models, desc="Evaluating Translations With Reference"):
    mt_list = df[model_name].tolist()  # Get machine translations for the current model
    evaluation = evaluate_translations_with_reference(model, Original, mt_list, Reference)
    
    # Store evaluation metrics for the current model
    results_with_ref[model_name] = {
        "sentence_scores": evaluation.scores,
        "system_score": evaluation.system_score,
        "error_spans": evaluation.metadata.error_spans
    }
    
    # Print the evaluation results for the current model without reference
    print(f"\n{model_name} Evaluation With Reference:")
    print("Sentence-level scores:", [f"{score:.3f}" for score in results_with_ref[model_name]["sentence_scores"]])
    print(f"System-level score: {results_with_ref[model_name]['system_score']:.3f}")

# Create a DataFrame combining all evaluation results without reference (wide format)
df_results_with_ref = pd.DataFrame({
    'Scale': df['Scale'],
    'Azure': results_with_ref["Azure"]["sentence_scores"],
    'DeepL': results_with_ref["DeepL"]["sentence_scores"],
    'OpenAI': results_with_ref["OpenAI"]["sentence_scores"],
    'WidnAI': results_with_ref["WidnAI"]["sentence_scores"],
    'Human': results_with_ref["Profissional_Translation_ENtoPT"]["sentence_scores"]
})

# Round all numerical values to three decimal places
df_results_with_ref = df_results_with_ref.round(3)

# Convert the wide DataFrame into long format using melt
df_results_with_ref = pd.melt(df_results_with_ref, id_vars='Scale', 
                                 var_name='Translation', 
                                 value_name='Sentence_Score')


# Define the output path for the COMET evaluation results without reference
output_path_without_ref = r'COMET_result_ENtoPT_with_reference.csv'
# Save the results DataFrame to a CSV file
df_results_with_ref.to_csv(output_path_without_ref, index=False, sep=";", encoding="utf-8-sig")

print(f"\nEvaluation without reference completed. Results saved at: {output_path_without_ref}")

# ------------------ Processing Time Calculation ------------------
# Record the end time of the entire process
end_time = time.time()
# Calculate the total elapsed time in seconds
elapsed_time = end_time - start_time

# Convert the elapsed time into hours, minutes, and seconds
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = elapsed_time % 60

# Print the total processing time in a human-readable format
print(f"\nTotal processing time: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")
