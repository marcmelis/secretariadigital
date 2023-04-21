import os
from typing import Dict, List, Tuple

import openai
import pandas as pd

from data import get_text, list_to_dataframe
from api_key import API_KEY

openai.api_key =  API_KEY


MODEL_NAME= "ada-002"
DOC_EMBEDDINGS_MODEL=f"text-embedding-{MODEL_NAME}"
# In this case, with ada-002 we don't distinguish between query embeddings and document embeddings
QUERY_EMBEDDINGS_MODEL = DOC_EMBEDDINGS_MODEL
EMBEDDINGS_CSV=f"data/embeddings_{MODEL_NAME}.csv"

def get_embedding(text: str, model: str) -> List[float]:
    result = openai.Embedding.create(
      model=model,
      input=text)
    embedding = result["data"][0]["embedding"]
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> List[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str) -> List[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def compute_doc_embeddings(df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_doc_embedding(r.content.replace("\n", " ")) for idx, r in df.iterrows()
    }


def main():
    text = get_text()
    df = list_to_dataframe(text)

    if os.path.isfile(EMBEDDINGS_CSV):
        embeddings_df = pd.read_csv(EMBEDDINGS_CSV, sep='%')
        new_rows = df.iloc[len(embeddings_df):]
        new_embeddings = compute_doc_embeddings(new_rows)
        new_embeddings_df = pd.DataFrame.from_dict(new_embeddings, orient='index')
        embeddings_df = pd.concat([embeddings_df, new_embeddings_df])

        print(f"Writing {len(new_embeddings_df)} new embeddings. Current embeddings: {len(embeddings_df)}")
    else:
        document_embeddings = compute_doc_embeddings(df)
        embeddings_df = pd.DataFrame.from_dict(document_embeddings, orient='index')
        print(f"Writing {len(embeddings_df)} new embeddings. Current embeddings: 0")

    embeddings_df.to_csv(EMBEDDINGS_CSV, index=False, sep='%')

if __name__ == "__main__":
    main()
