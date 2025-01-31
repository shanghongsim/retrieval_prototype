import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

from logger import logger


def get_vector_db(model_name, embeddings, all_splits=None):
    if not os.path.exists(f"faiss_index_{model_name}"):
        print("Index not found, constructing...")
        db = FAISS.from_documents(all_splits, embeddings)
        db.save_local(f"faiss_index_{model_name}")
    else:
        print("Index found, loading...")
        db = FAISS.load_local(f"faiss_index_{model_name}", embeddings, allow_dangerous_deserialization=True)
    return db

def get_query_embeddings(train_dataset, db, embedding_path="query_embeddings.pkl"):
    if not os.path.exists(embedding_path):
        logger.info("Processing queries...")
        queries = [data['question'] for data in train_dataset]
        def embed_with_progress(query):
            return db.embedding_function.embed_query(query)
        query_embeddings = []
        with ThreadPoolExecutor() as executor:
            for embedding in tqdm(executor.map(embed_with_progress, queries), total=len(queries), desc="Embedding queries"):
                query_embeddings.append(embedding)
        query_embeddings = np.array(query_embeddings)
        with open(embedding_path, "wb") as f:
            pickle.dump(query_embeddings, f)
    else:
        with open(embedding_path, "rb") as f:
            query_embeddings = pickle.load(f)
    return query_embeddings
        