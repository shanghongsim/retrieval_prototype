import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

from logger import logger


def get_vector_db(model_name, embeddings, all_splits=None):
    if not os.path.exists(f"faiss_index_{model_name}"):
        logger.info("Index not found, constructing...")
        db = FAISS.from_documents(all_splits, embeddings)
        db.save_local(f"faiss_index_{model_name}")
    else:
        logger.info("Index found, loading...")
        db = FAISS.load_local(
            f"faiss_index_{model_name}",
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return db


def get_query_embeddings(train_dataset, db, embedding_path="query_embeddings.pkl"):
    if not os.path.exists(f"query_embeddings/{embedding_path}"):
        logger.info("Processing queries...")
        queries = [data["question"] for data in train_dataset]

        def embed_with_progress(query):
            return db.embedding_function.embed_query(query)

        query_embeddings = []
        with ThreadPoolExecutor() as executor:
            for embedding in tqdm(
                executor.map(embed_with_progress, queries),
                total=len(queries),
                desc="Embedding queries",
            ):
                query_embeddings.append(embedding)
        query_embeddings = np.array(query_embeddings)
        with open(f"query_embeddings/{embedding_path}", "wb") as f:
            pickle.dump(query_embeddings, f)
    else:
        with open(f"query_embeddings/{embedding_path}", "rb") as f:
            query_embeddings = pickle.load(f)
    return query_embeddings


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
    result_new = [
        Document(page_content=content[1], metadata={"title": title})
        for content, title in zip(sorted_original, sorted_titles)
    ]
    return result_new, list(sorted_scores)
