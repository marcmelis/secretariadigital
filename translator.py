# main.py
import os
import json
import gettext
from googletrans import Translator
from textblob import TextBlob

LANGUAGE = 'es'
# Set up the local translation environment
def setup_local_translation(language):
    gettext.translation('messages', localedir='locales', languages=[language], fallback=True).install()

def translate_text(text, target_language):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

def translate_and_save(text, language):
    # Check if the translations directory exists
    if not os.path.exists('translations'):
        os.makedirs('translations')

    # Load the existing translations or create an empty dictionary
    translations_file = f'translations/{language}.json'
    if os.path.exists(translations_file):
        with open(translations_file, 'r', encoding='utf-8') as file:
            translations = json.load(file)
    else:
        translations = {}
    # Check if the translation is already available locally
    if text in translations:
        return translations[text]
    # If not, use the Google Translate API to translate the text
    translated_text = translate_text(text, language).capitalize()
    # Save the translated text for future use
    translations[text] = translated_text
    with open(translations_file, 'w', encoding='utf-8') as file:
        json.dump(translations, file, ensure_ascii=False, indent=4)

    return translated_text


def translate(message: str,language: str) -> str:
    setup_local_translation(language)
    return translate_and_save(message, language)

def translate_default(message: str) -> str:
    return translate(message, LANGUAGE)

def detect_language(message: str) -> str:
    language = TextBlob(text).detect_language()
    return language

def greet(language: str) -> str:
    setup_local_translation(language)
    message = "Hello, world!"
    translated_message = translate_and_save(message, language)
    print(translated_message)

if __name__ == "__main__":
    greet('es')  # Change the language code to the desired language
