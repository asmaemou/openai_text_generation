from transformers import pipeline

# Load the translation pipeline for English-to-French
translator = pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr')

# Translate text
text = "What time is it?"
result = translator(text, max_length=40)

# Print the translated text
print(f"Translated text: {result[0]['translation_text']}")
