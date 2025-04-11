import os

import numpy as np
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(corpus: list[str]) -> np.array:
    dimensions = 512
    for i, text in enumerate(corpus):
        if text is None or len(text) == 0:
            corpus[i] = "Empty text"


    # batch responses to 1000 docs each
    k= len(corpus)//1000+1
    corpus_batches = [corpus[low*1000:min((low+1)*1000,len(corpus))] for low in range(k)]
    # Call the embedding API (using the text-embedding-3-large model)
    all_embeddings = []
    for i in range(k):
        response = openai.embeddings.create(
            input=corpus_batches[i],
            model="text-embedding-3-large",
            dimensions=dimensions
        )

        # Extract the embeddings from the response
        embeddings = [item.embedding for item in response.data]
        all_embeddings += embeddings

    # Create and return an array
    return np.array(all_embeddings)
