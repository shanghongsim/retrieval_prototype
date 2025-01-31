import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import colorlog
import faiss
import numpy as np
from datasets import load_dataset
from FlagEmbedding import FlagReranker
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


def setup_logger(name=None):
    fmt_string = '%(log_color)s %(asctime)s - %(levelname)s - %(message)s'
    log_colors = {
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'purple'
    }
    colorlog.basicConfig(log_colors=log_colors, format=fmt_string, level=colorlog.INFO)
    logger = colorlog.getLogger(name)
    logger.setLevel(colorlog.INFO)
    return logger

logger = setup_logger()

def get_vector_db(model_name, embeddings, all_splits=None):
    if not os.path.exists(f"faiss_index_{model_name}"):
        logger.warning("Index not found, constructing...")
        db = FAISS.from_documents(all_splits, embeddings)
        db.save_local(f"faiss_index_{model_name}")
    else:
        logger.info("Index found, loading...")
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
        

def main():
    # Load wiki dataset
    wiki_dataset = load_dataset("wikipedia", "20220301.simple")
    if not os.path.exists("wiki_data.jsonl"):
        wiki_dataset["train"].to_json("wiki_data.jsonl")  

    # Convert to langchain Documents
    documents = []
    for row in wiki_dataset["train"].select(range(10000)):
        doc = Document(
            page_content=row["text"], 
            metadata={
                "id": row["id"],
                "url": row["url"],
                "title": row["title"],
            }
        )
        documents.append(doc)

    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)

    logger.info(f'Total number of splits: {len(all_splits)}')
    logger.info(f'{all_splits[0]}')

    # Load train dataset
    train_dataset = []
    with open('train.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            train_dataset.append(json.loads(line.strip()))

    # baseline
    max_score = 0
    for idx, item in enumerate(train_dataset):
        max_score += item['points']
    logger.info(f'Max score: {max_score}')

    # Build index
    model_name = "BAAI/bge-large-en-v1.5"
    logger.info(model_name)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs = {'device':'cuda'},
        encode_kwargs = {'normalize_embeddings': False})

    model_name = model_name.split("/")[1]
    model_name = model_name.replace("-", "_")
    db = get_vector_db(model_name, embeddings, all_splits)

    # Embed queries
    get_query_embeddings(train_dataset, db, embedding_path=f"query_embeddings_{model_name}.pkl")

if __name__ == "__main__":
    main()