import os
import gzip
from collections import defaultdict
from ranx import Run, Qrels
from ranx import fuse, optimize_fusion
from sentence_transformers import util
import pandas as pd
import pytrec_eval
import numpy as np
import logging


# Configuration ========================================================================================================
logging.basicConfig(level=logging.INFO, filename="task2.log", format="%(asctime)s %(message)s", filemode="a")
logger = logging.getLogger("task2")

mini_ranking_path = "./results/task1/mini/cross-encoder-cross-encoder-ms-marco-MiniLM-L-2-v2-2023-05-09_16-11" \
                    "-24ranking.run"
tiny_ranking_path = "./results/task1/tiny/cross-encoder-cross-encoder-ms-marco-TinyBERT-L-2-v2-2023-05-10_18-44" \
                    "-39ranking.run"
bert_ranking_path = "./results/task1/dis/cross-encoder-distilroberta-base-2023-05-09_19-33-15ranking.run"

data_folder = 'trec2019-data'
os.makedirs(data_folder, exist_ok=True)


def log_results(msg):
    logger.info(msg)
    print(msg)


# Load test data (evaluation.ipynb) ====================================================================================
queries = {}
queries_filepath = os.path.join(data_folder, 'msmarco-test2019-queries.tsv.gz')
if not os.path.exists(queries_filepath):
    util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz',
                  queries_filepath)

with gzip.open(queries_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query

relevant_docs = defaultdict(lambda: defaultdict(int))
qrels_filepath = os.path.join(data_folder, '2019qrels-pass.txt')

if not os.path.exists(qrels_filepath):
    util.http_get('https://trec.nist.gov/data/deep/2019qrels-pass.txt', qrels_filepath)

with open(qrels_filepath) as fIn:
    for line in fIn:
        qid, _, pid, score = line.strip().split()
        score = int(score)
        if score > 0:
            relevant_docs[qid][pid] = score

# Only use queries that have at least one relevant passage
relevant_qid = []
for qid in queries:
    if len(relevant_docs[qid]) > 0:
        relevant_qid.append(qid)

# Read the top 1000 passages that are supposed to be re-ranked
passage_filepath = os.path.join(data_folder, 'msmarco-passagetest2019-top1000.tsv.gz')

if not os.path.exists(passage_filepath):
    util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz',
                  passage_filepath)

passage_cand = {}
with gzip.open(passage_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        qid, pid, query, passage = line.strip().split("\t")
        if qid not in passage_cand:
            passage_cand[qid] = []

        passage_cand[qid].append([pid, passage])


# Load ranking files ===================================================================================================
print("Loading the baseline...")
qrels_dict = {}
for qid in relevant_qid:
    qrels_dict[qid] = relevant_docs[qid]
qrels = Qrels.from_dict(qrels_dict)
print("Loading mini model...")
mini_df = pd.read_table(mini_ranking_path, sep=" ", header=0, names=["qid", "Q0", "did", "rank", "score", "other"],
                        dtype={"qid": str, "Q0": str, "did": str, "rank": int, "score": float, "other": str})
mini_run = Run.from_df(df=mini_df, q_id_col="qid", doc_id_col="did", score_col="score")
print("Loading tiny model...")
tiny_df = pd.read_table(tiny_ranking_path, sep=" ", header=0, names=["qid", "Q0", "did", "rank", "score", "other"],
                        dtype={"qid": str, "Q0": str, "did": str, "rank": int, "score": float, "other": str})
tiny_run = Run.from_df(df=tiny_df, q_id_col="qid", doc_id_col="did", score_col="score")
print("Loading bert model...")
bert_df = pd.read_table(bert_ranking_path, sep=" ", header=0, names=["qid", "Q0", "did", "rank", "score", "other"],
                        dtype={"qid": str, "Q0": str, "did": str, "rank": int, "score": float, "other": str})
bert_run = Run.from_df(df=bert_df, q_id_col="qid", doc_id_col="did", score_col="score")


# Fusion and evaluation ================================================================================================
print("Creating ensemble models...")
for method in ["mixed", "rrf", "bayesfuse", "posfuse", "w_bordafuse"]:
    log_results(f"Method: {method}")
#   for norm in ["min-max", "max", "sum", "zmuv", "rank", "borda"]:
#       log_results(f"Normalization: {norm}")
    best_params = optimize_fusion(
        qrels=qrels,
        runs=[mini_run, tiny_run, bert_run],
        norm="min-max",
        method=method,
        metric="ndcg@100",
    )
    combined_run = fuse(
        runs=[mini_run, tiny_run, bert_run],
        norm="min-max",
        method=method,
        params=best_params,
    )

    combined_run_dict = dict(combined_run)
    evaluator = pytrec_eval.RelevanceEvaluator(relevant_docs, {'ndcg_cut.10', 'recall_100', 'map_cut.1000'})
    scores = evaluator.evaluate(combined_run_dict)

    log_results(f"Queries: {len(relevant_qid)}")
    log_results("NDCG@10: {:.2f}".format(np.mean([ele["ndcg_cut_10"] for ele in scores.values()]) * 100))
    log_results("Recall@100: {:.2f}".format(np.mean([ele["recall_100"] for ele in scores.values()]) * 100))
    log_results("MAP@1000: {:.2f}\n".format(np.mean([ele["map_cut_1000"] for ele in scores.values()]) * 100))
