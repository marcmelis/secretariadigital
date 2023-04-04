import os
import argparse
from typing import Dict, List, Tuple
import tkinter as tk
import threading

import numpy as np
import openai
import pandas as pd
from transformers import GPT2TokenizerFast
import speech_recognition as sr

from data import get_text, list_to_dataframe
from generate_embeddings import get_query_embedding

CONFIG_FILE = 'config.py'
EMBEDDINGS_FILE ='data/embeddings_ada-002.csv'

# Check if config file exists
if os.path.exists(CONFIG_FILE):
    # If it does, import API key from config file
    from config import API_KEY
else:
    # If it doesn't, prompt user for API key and create config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', help='API key for accessing the service')
    args = parser.parse_args()

    if args.api_key:
        # User provided API key as a command-line argument
        API_KEY = args.api_key
    else:
        # User didn't provide API key, prompt for input
        API_KEY = input('Please enter your API key: ')

    # Write API key to config file
    with open(CONFIG_FILE, 'w') as f:
        f.write(f'API_KEY = "{API_KEY}"\n')

openai.api_key = API_KEY

COMPLETIONS_MODEL = "text-davinci-003"
COMPLETIONS_MODEL = "gpt-3.5-turbo"

os.environ['TOKENIZERS_PARALLELISM'] = 'false'



COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

# Maximum length of the context provided, the more context, the more accurate answers
MAX_SECTION_LEN = 500
SEPARATOR = "\n* "

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(SEPARATOR))


def vector_similarity(x: List[float], y: List[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: Dict[Tuple[str, str], np.array]) -> List[Tuple[float, Tuple[str, str]]]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = df.loc[section_index]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    # print(f"Selected {len(chosen_sections)} document sections:")
    # print("\n".join(chosen_sections_indexes))

    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    header = ""
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

def generate_chat_completion_message_from_prompt(prompt: str) -> List[Dict[str,str]]:
    messages = [
        {
        "role": "user",
        "content" : prompt
        }
    ]
    return messages

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: Dict[Tuple[str, str], np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )

    if show_prompt:
        print(prompt)

    response = openai.ChatCompletion.create(
                messages=generate_chat_completion_message_from_prompt(prompt),
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["message"]["content"].strip(" \n")


def submit_text():
    text = entry.get()
    print("Pregunta:", text)
    entry.delete(0, tk.END)
    answer = answer_query_with_context(text, df, document_embeddings)
    answer_label.config(text=answer)


def submit_voice():
    def recognize_voice():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Habla ahora...")
            audio = recognizer.listen(source)
        try:
            voice_text = recognizer.recognize_google(audio, language="es-ES")
            print("Pregunta por voz:", voice_text)
            answer = answer_query_with_context(voice_text, df, document_embeddings)
            answer_label.config(text=answer)
        except sr.UnknownValueError:
            print("No se entendió la pregunta.")
        except sr.RequestError as e:
            print("Error al obtener resultados; {0}".format(e))

    voice_thread = threading.Thread(target=recognize_voice)
    voice_thread.start()


def get_embeddings():
    df_embeddings = pd.read_csv(EMBEDDINGS_FILE, header=0, sep="%").T
    document_embeddings = df_embeddings.to_dict('list')
    return document_embeddings

def get_df():
    text = get_text()
    df = list_to_dataframe(text)
    return df

df = get_df()
document_embeddings = get_embeddings()

app = tk.Tk()
app.title("Secreatria Digital")

frame = tk.Frame(app)
frame.pack(padx=10, pady=10)

entry = tk.Entry(frame)
entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

text_button = tk.Button(frame, text="Enviar texto", command=submit_text)
text_button.pack(side=tk.LEFT, padx=(5, 0))

voice_button = tk.Button(frame, text="Enviar voz", command=submit_voice)
voice_button.pack(side=tk.LEFT, padx=(5, 0))

answer_label = tk.Label(app, text="", wraplength=300)
answer_label.pack(padx=10, pady=(10, 0))


def main():
    app.mainloop()


if __name__ == "__main__":
    main()
