import os
import argparse
from typing import Dict, List, Tuple
import threading

import numpy as np
import openai
import pandas as pd
from transformers import GPT2TokenizerFast
import speech_recognition as sr
import pyttsx3

from translator import translate, detect_language
from data import get_text, list_to_dataframe
from generate_embeddings import get_query_embedding
from api_key import API_KEY

CONFIG_FILE = 'config.py'
EMBEDDINGS_FILE ='data/embeddings_ada-002.csv'

DEFAULT_LANGUAGE = 'es'

openai.api_key =  API_KEY


COMPLETIONS_MODEL = "gpt-3.5-turbo"

os.environ['TOKENIZERS_PARALLELISM'] = 'false'



COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

# Maximum length of the context provided, the more context, the more accurate answers
MAX_SECTION_LEN = 1000
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


def get_embeddings():
    df_embeddings = pd.read_csv(EMBEDDINGS_FILE, header=0, sep="%").T
    document_embeddings = df_embeddings.to_dict('list')
    return document_embeddings

def get_df():
    text = get_text()
    df = list_to_dataframe(text)
    return df

LANGUAGE = DEFAULT_LANGUAGE

df = get_df()
document_embeddings = get_embeddings()

def main():
    user_message = input("Write your question: \n")
    answer = answer_query_with_context(user_message, df, document_embeddings)
    print(answer)

if __name__ == "__main__":
    main()
