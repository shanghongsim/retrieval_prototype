import argparse
import json
import os

import numpy as np
import yaml
from datasets import load_dataset
from FlagEmbedding import FlagReranker
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from logger import logger
from utils import get_query_embeddings, get_vector_db, rerank_topk


def main(args):

    # Load Wiki dataset
    logger.info("Loading wiki dataset...")
    wiki_dataset = load_dataset("wikipedia", "20220301.simple")
    if not os.path.exists("data/wiki_data.jsonl"):
        wiki_dataset["train"].to_json("data/wiki_data.jsonl")

    # Convert text to langchain Documents
    logger.info("Constructing documents...")
    documents = []
    for row in wiki_dataset["train"].select(range(10000)):
        doc = Document(
            page_content=row["text"],
            metadata={
                "id": row["id"],
                "url": row["url"],
                "title": row["title"],
            },
        )
        documents.append(doc)

    # Chunk documents
    logger.info("Chunk documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)

    logger.info(f"Total number of splits: {len(all_splits)}")
    logger.info(f"{all_splits[0]}")

    # Load train dataset
    train_dataset = []
    with open(args.dataset, "r", encoding="utf-8") as file:
        for line in file:
            train_dataset.append(json.loads(line.strip()))

    # Baseline
    max_score = sum([item["points"] for item in train_dataset])
    logger.info(f"Max score: {max_score}")

    # Get index
    model_name = args.embedding_model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": False},
    )

    model_name = model_name.split("/")[1].replace("-", "_")
    db = get_vector_db(model_name=model_name, embeddings=embeddings)
    print(type(db.index))

    if args.task == "single":
        logger.info("Processing queries...")
        query_embeddings = get_query_embeddings(
            train_dataset, db, embedding_path=f"query_embeddings_{model_name}.pkl"
        )

        if "gtr" in model_name:
            query_embeddings = np.array(
                get_query_embeddings(
                    train_dataset,
                    db,
                    embedding_path=f"query_embeddings_{model_name}.pkl",
                ),
                dtype=np.float16,
            )
        else:
            query_embeddings = np.array(
                get_query_embeddings(
                    train_dataset,
                    db,
                    embedding_path=f"query_embeddings_{model_name}.pkl",
                ),
                dtype=np.float16,
            )

        D, I = db.index.search(query_embeddings, k=5)  # k = number of nearest neighbors

        logger.info("Processing results...")
        total_score = 0
        results = []
        for idx, item in enumerate(train_dataset):
            query = item["question"]
            gold_title = item["article"]
            points = item["points"]

            # Store top 5 retrieved documents
            retrieved_docs = []
            for rank in range(5):
                doc_index = I[idx][rank]
                distance = D[idx][rank]

                doc = db.docstore._dict[db.index_to_docstore_id[doc_index]]
                doc_content = doc.page_content
                doc_metadata = doc.metadata
                retrieved_title = doc_metadata["title"]

                retrieved_docs.append(
                    {
                        "rank": rank + 1,
                        "title": retrieved_title,
                        "content": doc_content,
                        "distance": float(distance),
                    }
                )

            total_score += (
                item["points"] if retrieved_docs[0]["title"] == gold_title else 0
            )

            results.append(
                {
                    "query": query,
                    "gold_article": gold_title,
                    "points": points,
                    "retrieved_docs": retrieved_docs,
                }
            )

    elif args.task == "rerank":
        retriever = db.as_retriever(search_kwargs={"k": 15})
        reranker = FlagReranker("BAAI/bge-reranker-large")

        # Retrieval
        results = []
        total_score = 0
        for idx, item in enumerate(tqdm(train_dataset)):
            question = item["question"]
            points = item["points"]

            retrieved_docs = retriever.invoke(question)
            retrieved_docs, reranked_scores = rerank_topk(
                reranker, question, retrieved_docs
            )
            retrieved_data = [
                {
                    "title": doc.metadata["title"],
                    "content": doc.page_content,
                    "score": score,
                }
                for doc, score in zip(retrieved_docs, reranked_scores)
            ]

            retr_doc = retrieved_docs[0]
            retr_title = retr_doc.metadata["title"]
            gold_title = item.get("article", 0)

            if "train" in args.dataset:
                total_score += item["points"] if retr_title == gold_title else 0

            results.append(
                {
                    "query": question,
                    "gold_article": gold_title,
                    "points": points,
                    "retrieved_docs": retrieved_data,
                }
            )

    elif args.task == "hybrid":
        bm25_retriever = BM25Retriever.from_documents(all_splits)
        bm25_retriever.k = 10
        faiss_retriever = db.as_retriever(search_kwargs={"k": 10})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.25, 0.75]
        )

        # Retrieval
        results = []
        total_score = 0
        for idx, item in enumerate(tqdm(train_dataset)):
            question = item["question"]
            points = item["points"]

            retrieved_docs = ensemble_retriever.invoke(question)
            retrieved_data = [
                {
                    "title": doc.metadata["title"],
                    "content": doc.page_content,
                }
                for doc in retrieved_docs
            ]

            retr_doc = retrieved_docs[0]
            retr_title = retr_doc.metadata["title"]
            gold_title = item["article"]
            total_score += item["points"] if retr_title == gold_title else 0

            results.append(
                {
                    "query": question,
                    "gold_article": gold_title,
                    "points": points,
                    "retrieved_docs": retrieved_data,
                }
            )

    output_data = {
        "model": model_name,
        "total_score": total_score,
        "accuracy_percentage": round(total_score / max_score * 100, 2),
        "results": results,
    }

    test = "_test" if "test" in args.dataset else ""
    with open(
        f"results/retrieved_results_{args.task}_{args.embedding_model.split('/')[1]}{test}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    logger.info(
        f"{args.task} score: {total_score} ({output_data['accuracy_percentage']}%)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document selection")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the config file"
    )
    parser.add_argument(
        "--task", type=str, help="Evaluation Task. single, hybrid, rerank"
    )
    parser.add_argument("--embedding_model", type=str, help="Embedding model")
    parser.add_argument("--rerank_model", type=str, default=None, help="Rerank model")
    parser.add_argument(
        "--dataset", type=str, help="Path to the file containing questions"
    )

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()
    for k in args.__dict__:
        logger.info(f"{k}: {args.__dict__[k]}")

    main(args)
