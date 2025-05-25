import os
import pandas as pd
import contractions
import requests
import time
import deepl
import openai
from tqdm import tqdm

# Record the start time of the entire process
start_time = time.time()

os.environ["AZURE_API_KEY"] = "AZURE_API_KEY"
os.environ["DEEPL_API_KEY"] = "DEEPL_API_KEY"
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY
os.environ["WIDN_API_KEY"] = "WIDN_API_KEY"

# Path to the original CSV file
csv_path = 'file.csv'
df = pd.read_csv(csv_path, delimiter=";")

# Expand contractions in the "Original_Version" column and create a new column "Original"
df['Original'] = df['Original'].apply(lambda x: contractions.fix(str(x)) if pd.notnull(x) else None)
#df['Profissional_Translation_PTtoEN'] = df['Profissional_Translation_PTtoEN'].apply(lambda x: contractions.fix(str(x)) if pd.notnull(x) else None)

# =================== Azure Translator ===================
azure_endpoint = "https://api-nam.cognitive.microsofttranslator.com/translate?api-version=3.0"
azure_api_key = os.getenv("AZURE_API_KEY")
azure_headers = {
    "Ocp-Apim-Subscription-Key": azure_api_key,
    "Content-Type": "application/json",
    "Ocp-Apim-Subscription-Region": "brazilsouth"
}

def translate_azure(text):
    """Translates text using Azure Translator API."""
    if not text or pd.isna(text):
        return None
    body = [{"text": text}]
    try:
        response = requests.post(f"{azure_endpoint}&from=en&to=pt-BR", headers=azure_headers, json=body)
        response.raise_for_status()
        return response.json()[0]['translations'][0]['text']
    except Exception as e:
        print(f"Azure error with '{text}': {e}")
        return None

# =================== DeepL Translator ===================
deepl_auth_key = os.getenv("DEEPL_API_KEY")
translator_deepl = deepl.Translator(deepl_auth_key)

def translate_deepl(text):
    """Translates text using DeepL API."""
    try:
        return translator_deepl.translate_text(text, source_lang="EN", target_lang="PT-BR").text
    except Exception as e:
        print(f"DeepL error with '{text}': {e}")
        return None

# =================== OpenAI Translator ===================
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate_openai(text):
    """Translates text using OpenAI API without intervention."""
    prompt = f"Translate the following text from English to Brazilian Portuguese, without any modifications or additional explanations:\n\n{text}"
    try:
        response = client.chat.completions.create(
            model='gpt-4-turbo',
            messages=[
                {'role': 'system', 'content': 'You are a neutral translator. Your task is only to translate text accurately, without adding opinions or modifying the content.'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI error with text '{text}': {e}")
        return None


# =================== Widn.AI Translator ===================
widn_api_key = os.getenv("WIDN_API_KEY")
widn_url = "https://api.widn.ai/v1/translate"
widn_headers = {
    "X-Api-Key": widn_api_key,
    "Content-Type": "application/json"
}

def translate_widn(text, source_lang="en", target_lang="pt-BR", model="vesuvius", delay=2, max_retries=3):
    """Translates text using Widn.AI API with rate limit handling."""
    data = {
        "config": {
            "sourceLocale": source_lang,
            "targetLocale": target_lang,
            "model": model
        },
        "sourceText": [text]
    }
    
    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(widn_url, headers=widn_headers, json=data)
            if response.status_code == 200:
                translation = response.json().get("targetText", [None])[0]
                time.sleep(1)  # Pause after each successful translation
                return translation
            elif response.status_code == 429:
                print(f"Widn.AI rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                retries += 1
            else:
                print(f"Widn.AI error with text '{text}': {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error accessing Widn.AI with text '{text}': {e}")
            return None

    print(f"Failed to translate with Widn.AI after {max_retries} attempts.")
    return None


# =================== Applying Translations ===================
for col, func in zip(["Azure", "DeepL", "OpenAI", "WidnAI"], 
                     [translate_azure, translate_deepl, translate_openai, translate_widn]):
    print(f"\nStarting translation with {col}...")
    
    if col == "WidnAI":
        translations = []
        for text in tqdm(df['Original'], desc=f"{col} Translating", total=len(df)):
            translations.append(func(str(text)) if pd.notnull(text) else None)
        df[col] = translations
    else:
        df[col] = [func(str(text)) if pd.notnull(text) else None for text in tqdm(df['Original'], desc=f"{col} Translating")]

    time.sleep(1)  # Pause of 1 second between API calls to avoid rate limits

# =================== Saving the Final Combined DataFrame ===================
final_output_path = 'combined_translations.csv'
df.to_csv(final_output_path, index=False, sep=";", encoding="utf-8-sig")

print(f"\nAll translations completed and saved at: {final_output_path}")

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