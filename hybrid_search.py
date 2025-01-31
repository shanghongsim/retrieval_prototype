import json
import os

import numpy as np
from datasets import load_dataset
from FlagEmbedding import FlagReranker
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from logger import logger
from utils import get_query_embeddings, get_vector_db


def main():
    # Load Wiki dataset
    wiki_dataset = load_dataset("wikipedia", "20220301.simple")
    if not os.path.exists("wiki_data.jsonl"):
        wiki_dataset["train"].to_json("wiki_data.jsonl")  

    # Convert text to langchain Documents
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
    # logger.info(documents[0])

    # Chunk text
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

    # Baseline
    max_score = 0
    for idx, item in enumerate(train_dataset):
        max_score += item['points']
    logger.info(f'Max score: {max_score}')

    # Load embeddings
    model_name = "BAAI/bge-large-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs = {'device':'cuda'},
        encode_kwargs = {'normalize_embeddings': False})
    model_name = model_name.split("/")[1]
    model_name = model_name.replace("-", "_")
    db = get_vector_db(model_name = model_name, embeddings=embeddings)


    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = 10
    faiss_retriever = db.as_retriever(search_kwargs={"k": 10}) 

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.25,0.75]
    )

    # Retrieval
    results = []
    total_score = 0

    for idx, item in enumerate(tqdm(train_dataset)): # for each validation query
        question = item['question']
        points = item['points']

        retrieved_docs = ensemble_retriever.invoke(question)
        retrieved_data = [
            {
                "title": doc.metadata["title"],
                "content": doc.page_content,
            }
            for doc in retrieved_docs
        ]

        retr_doc = retrieved_docs[0]
        retr_title = retr_doc.metadata['title']
        gold_title = item['article']

        if retr_title == gold_title:
            total_score += item['points']
            
        results.append({
            "query": question,
            "gold_article": gold_title,
            "points": points, 
            "retrieved_docs": retrieved_data
        })

    output_data = {
        "model": model_name,
        "total_score": total_score,
        "accuracy_percentage": round(total_score / max_score * 100, 2),
        "results": results
    }

    with open(f"retrieved_results_hybrid.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    logger.info(f"Hybrid score: {total_score} ({output_data['accuracy_percentage']}%)")


if __name__ == "__main__":
    main()