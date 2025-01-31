# Retrieval prototype

## Task 📖

Given a question, you have to find the **best wikipedia article** that answers it.

## Approach 🔍

For this task, we will primarily explore semantic search with the following models:

1. Baseline: `sentence-transformers/all-mpnet-base-v2` (109M)

- Recommended by the [SBERT documentation](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#semantic-search-models) to provide the best quality.
- Small size (109M parameters) makes it computationally efficient and serve as a baseline

2. Large-Scale Model: `sentence-transformers/gtr-t5-xxl` (4.86B)

- A powerful model used by [Meta DPR](https://github.com/facebookresearch/DPR)
- Used to test the hypothesis that a larger model is better able to capture document chunk semantics, leading to improved retrieval.
- Cons: Extremely large and requires significant resources; index creation takes nearly 2 hours on an A100 GPU.

3. Mid-Sized Model: `BAAI/bge-large-en-v1.5` (335M)

- Strong performance on the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
- Smaller than gtr-t5-xxl but expected to be competitive in retrieval.
- Aims to provide an efficient alternative to the large-scale model.

In this project, we use a mixture of [Langchain](https://python.langchain.com/docs/integrations/vectorstores/faiss/#manage-vector-store) and [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) to perform retrieval. The index type is `IndexFlatL2` and similarity metric is euclidean distance.

## Evaluation metrics 📐

In this project, we will use **top-1 accuracy** to measure as the train and test set only specify one best wikipedia article. We also visually inspect a few samples from the retrieved results.

## Optimizations ⏳

**Reranking**

- Motivated by the observation from single embedding experiments that gold article is somewhere in the top 5 retrieved documents
- Uses a second-stage reranker (`BAAI/bge-reranker-large`) to refine the ordering of retrieved documents and improve the likelihood of the best article ranking at position #1.

**Hybrid search**

- Helps address cases where semantic models alone may not retrieve the most relevant article.
- Combines semantic search with BM25 to leverage both lexical and semantic signals.

## Results ✏️

*Table 1: Evaluation results from `train.jsonl`.*

| **Method**                           | **Accuracy** |
|--------------------------------------|------------:|
| **Single Retriever**                 |             |
| `all-mpnet-base-v2`                   | 48.12%      |
| `gtr-t5-xxl`                          | 52.57%      |
| `bge-large-en-v1.5`                   | 53.98%      |
| **Hybrid Search**                     |             |
| `gtr-t5-xxl + BM25`                   | 47.74%      |
| `bge-large-en-v1.5 + BM25`            | 50.01%      |
| **Reranking**                         |             |
| `gtr-t5-xxl + bge-reranker-large`     | 56.47%      |
| `bge-large-en-v1.5 + bge-reranker-large` | **53.98%** |

From the results, we see that using `bge-large-en-v1.5` to retrieve the top 15 documents before reranking them using `bge-reranker-large` yields the best results of 57.96% accuracy.

**Findings**

1. Reranking boosts retrieval performance substantially. Adding a reranker improves `gtr-t5-xxl` performance by approximately 4% (52.57% to 56.47%) and `bge-large-en-v1.5` performance by approximately 4% (53.98% to 53.98%).

2. Using bigger model does not necessarily yield better results. At 4.86B, `gtr-t5-xxl` performs worse than `bge-large-en-v1.5` which only has 335M. Time to build index is: `all-mpnet-base-v2` (~10min) < `bge-large-en-v1.5` (~30min) < `bge-large-en-v1.5` (~2h). The time and performance tradeoff for `gtr-t5-xxl` is not good.

3. Hybrid search does not improve the search performance but rather degrades it slightly. Adding BM25 to `gtr-t5-xxl` degrades performance from 52.57% to 47.74%. Similarly, when BM25 is add for `bge-large-en-v1.5`, performance drops from 53.98% to 50.01%.

**Other observations**

- There seems to be some noise in the dataset, there is more than one article that can answer the question. For example:

```text
"query": "what is the name of the largest city in romania?",
"gold_article": "Bucharest",
"points": 52,
"retrieved_docs": [
    {
        "rank": 1,
        "title": "Romania",
        "content": "Religion\nRomania is a secular state. This means Romania has no national religion. The biggest religious group in Romania is the Romanian Orthodox Church. It is an autocephalous church inside of the Eastern Orthodox communion. In 2002, this religion made up 86.7% of the population. Other religions in Romania include Roman Catholicism (4.7%), Protestantism (3.7%), Pentecostalism (1.5%) and the Romanian Greek-Catholicism (0.9%).\n\nCities\n\nBucharest is the capital of Romania. It also is the biggest city in Romania, with a population of over 2 millions peoples.\n\nThere are 5 other cities in Romania that have a population of more than 300,000 people. These are Iaşi, Cluj-Napoca, Timişoara, Constanţa, and Craiova. Romania also has 5 cities that have more than 200,000 people living in them: Galaţi, Braşov, Ploieşti, Brăila, and Oradea.\n\nThirteen other cities in Romania have a population of more than 100,000 people.\n\nEconomy",
        "distance": 0.49112021923065186
    },
```

- The retrieved article ("Romania") contains the answer, but it isn't the exact expected document.
- This suggests that Top-5 accuracy or LLM-based evaluation might better capture retrieval effectiveness. However, for the sake of this task, we will stick to top-1 accuracy.

## Installation & Usage 🛠️

**Installation**

```bash
conda create --name faiss_1.8.0 python=3.10
conda activate faiss_1.8.0
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 pytorch=*=*cuda* pytorch-cuda=12 numpy
```

> Note: please adjust based on your CUDA version.

```bash
pip install -r requirements.txt
```

**Running experiments**

```python
CUDA_VISIBLE_DEVICES=0 python main_experiments.py --config config_file_path
```
