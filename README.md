# Retrieval prototype

## Task

Given a question, you have to find the **best wikipedia article** that answers it.

## Approach

For this task, we will try semantic search with the following models:

- sentence-transformers/all-mpnet-base-v2 (109M): According to [SBERT documentation](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#semantic-search-models), the all-mpnet-base-v2 model provides the best quality. It is relatively small at 109M. We will use this as a baseline.
- sentence-transformers/gtr-t5-xxl (4.86B): [Meta DPR](https://github.com/facebookresearch/DPR) uses the powerful gtr-t5-xxl model. At 4.86b parameters, it is quite large. Creating index takes almost 2h A100

Index structure: IndexFlatL2, euclidean distance

## Metrics

Top-1 accuracy, why?

## Optimizations

Reranker used: <https://huggingface.co/BAAI/bge-reranker-large>

Hybrid search

## Results

| **Method**                     | **Accuracy** | **Time** |
|--------------------------------|------------:| ------|
| **Single Retriever**           |             |
| all-mpnet-base-v2              | 48.12%  |
| gtr-t5-xxl                     | 52.57%  |
| bge-large-en-v1.5              | 53.98%  |
| **Hybrid Search**              |             |
| gtr-t5-xxl + BM25              | 47.74%  | 1.5h
| bge-large-en-v1.5 + BM25       | 50.01%    |
| **Reranking**                  |             |
| gtr-t5-xxl + bge-reranker-large | 56.47%  | 5h 20min |
| bge-large-en-v1.5 + bge-reranker-large | **57.96%**  | 4.5h |

## Installation & usage
