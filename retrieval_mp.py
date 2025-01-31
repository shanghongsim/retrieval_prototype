import json
import multiprocessing
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import colorlog
import faiss
import numpy as np
import torch
from datasets import load_dataset
from FlagEmbedding import FlagReranker
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from logger import logger
from utils import get_query_embeddings, get_vector_db

torch.multiprocessing.set_start_method("spawn", force=True)

def rerank_topk(reranker, question, documents):
    all_docs_ls = []
    titles = []  

    for document in documents:
        doc_content = document.page_content
        doc_title = document.metadata["title"]
        qs_doc_ls = [question, doc_content]
        all_docs_ls.append(qs_doc_ls)
        titles.append(doc_title)

    scores = reranker.compute_score(all_docs_ls)
    zipped_lists = list(zip(scores, all_docs_ls, titles))  # Include titles
    sorted_lists = sorted(zipped_lists, key=lambda x: x[0], reverse=True)
    sorted_scores, sorted_original, sorted_titles = zip(*sorted_lists)
    result_new = [Document(page_content=content[1], metadata={"title": title}) 
                  for content, title in zip(sorted_original, sorted_titles)]
    return result_new, list(sorted_scores)

def process_item(retriever, reranker, item):
    question = item['question']
    points = item['points']

    retrieved_docs = retriever.invoke(question)
    retrieved_docs, reranked_scores = rerank_topk(reranker, question, retrieved_docs)[:5]
    
    retrieved_data = [
        {
            "title": doc.metadata["title"],
            "content": doc.page_content,
            "score": score
        }
        for doc, score in zip(retrieved_docs, reranked_scores)
    ]

    retr_doc = retrieved_docs[0]
    retr_title = retr_doc.metadata['title']
    gold_title = item['article']

    # if retr_title == gold_title:
    #     total_score += item['points']
        
    processed = {
        "query": question,
        "gold_article": gold_title,
        "points": points, 
        "retrieved_docs": retrieved_data
    }
    return processed 
    
def main():

    # Load dataset
    logger.info("Loading wiki dataset...")
    dataset = load_dataset("wikipedia", "20220301.simple")

    if not os.path.exists("wiki_data.jsonl"):
        dataset["train"].to_json("wiki_data.jsonl")  

    # Construct Documents
    logger.info("Constructing documents...")
    documents = []
    for row in dataset["train"].select(range(10000)):
        doc = Document(
            page_content=row["text"],  # Assuming "text" contains the main content
            metadata={
                "id": row["id"],
                "url": row["url"],
                "title": row["title"],
            }
        )
        documents.append(doc)
    # logger.info(documents[0])

    # Chunk documents
    logger.info("Chunk documents...")
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

    # Create index
    model_name = "BAAI/bge-large-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs = {'device':'cuda'},
        encode_kwargs = {'normalize_embeddings': False})
    model_name = model_name.split("/")[1]
    model_name = model_name.replace("-", "_")
    db = get_vector_db(model_name = model_name, embeddings=embeddings)

    # faiss_index = db.index
    # gpu_resources = faiss.StandardGpuResources()
    # gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, faiss_index)  # 0 = GPU ID
    # db.index = gpu_index

    retriever = db.as_retriever(search_kwargs={"k": 15}) 
    reranker = FlagReranker('BAAI/bge-reranker-large')
    results = []
    total_score = 0


    with multiprocessing.Pool(processes = 32) as pool: 
        results = list(tqdm(pool.imap(partial(process_item, retriever, reranker), train_dataset, chunksize=10), total=len(train_dataset)))
    

    # for idx, item in enumerate(tqdm(train_dataset)):
    #     question = item['question']
    #     points = item['points']

    #     retrieved_docs = retriever.invoke(question)
    #     retrieved_docs, reranked_scores = rerank_topk(reranker, question, retrieved_docs)[:5]
        
    #     retrieved_data = [
    #         {
    #             "title": doc.metadata["title"],
    #             "content": doc.page_content,
    #             "score": score
    #         }
    #         for doc, score in zip(retrieved_docs, reranked_scores)
    #     ]

    #     retr_doc = retrieved_docs[0]
    #     retr_title = retr_doc.metadata['title']
    #     gold_title = item['article']

    #     if retr_title == gold_title:
    #         total_score += item['points']
            
    #     results.append({
    #         "query": question,
    #         "gold_article": gold_title,
    #         "points": points, 
    #         "retrieved_docs": retrieved_data
    #     })

    total_score = 0
    
    for result in results: 
        retrieved_docs, gold_article = result["retrieved_data"], result["gold_article"]
        total_score += retrieved_docs[0]['points'] if retrieved_docs[0]['title'] == gold_article else 0


    output_data = {
        "model": model_name,
        "total_score": total_score,
        "accuracy_percentage": round(total_score / max_score * 100, 2),
        "results": results
    }

    with open(f"retrieved_results_rerank_bge.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Rerank score: {total_score} ({output_data['accuracy_percentage']}%)")


if __name__ == "__main__":
    main()