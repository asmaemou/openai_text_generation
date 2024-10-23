from transformers import MarianMTModel, MarianTokenizer

# Load the model and tokenizer for English-to-French translation
model_name = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text, src_lang='en', tgt_lang='fr'):
    # Prepare the input text for translation
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Perform translation
    translated = model.generate(**inputs, max_length=50)

    # Decode the translated text
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

if __name__ == "__main__":
    # Example text in English
    text = "Hello, how are you?"
    # Translate from English to French
    translated_text = translate_text(text, src_lang='en', tgt_lang='fr')
    print(f"Translated text: {translated_text}")
