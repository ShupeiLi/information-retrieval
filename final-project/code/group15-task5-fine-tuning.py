# -*- coding: utf-8 -*-
"""
Group 15, Task 5
A modified version of (Fine-tuning) Final assignment.ipynb.
"""

from torch.utils.data import DataLoader
from sentence_transformers import util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import InputExample
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

train_batch_size = 32
num_epochs = 1
pos_neg_ration = 4

max_train_samples = 5e6  # 2e7
model_name = 'microsoft/MiniLM-L12-H384-uncased'
model = CrossEncoder(model_name, num_labels=1, max_length=512)
model_save_path = 'finetuned_models-copy/cross-encoder-' + model_name.replace("/", "-") + '-' + datetime.now().strftime(
   "%Y-%m-%d_%H-%M-%S")

model1_save_path = "./models/MiniLM"
model1 = CrossEncoder(model1_save_path, max_length=512)
model2_save_path = "./models/TinyBERT"
model2 = CrossEncoder(model2_save_path, max_length=512)
model3_save_path = "./models/distilroberta-base"
model3 = CrossEncoder(model3_save_path, max_length=512)

# Download MSMARCO data + BM25 initial ranking run file
data_folder = './msmarco-data'
os.makedirs(data_folder, exist_ok=True)

# Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage

# Read the train queries, store in queries dict
queries = {}
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query

train_samples = []
dev_samples = {}
num_dev_queries = 200
num_max_dev_negatives = 200

train_eval_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz')
if not os.path.exists(train_eval_filepath):
    logging.info("Download " + os.path.basename(train_eval_filepath))
    util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz', train_eval_filepath)

with gzip.open(train_eval_filepath, 'rt') as fIn:
    for line in fIn:
        qid, pos_id, neg_id = line.strip().split()

        if qid not in dev_samples and len(dev_samples) < num_dev_queries:
            dev_samples[qid] = {'query': queries[qid], 'positive': set(), 'negative': set()}

        if qid in dev_samples:
            dev_samples[qid]['positive'].add(corpus[pos_id])

            if len(dev_samples[qid]['negative']) < num_max_dev_negatives:
                dev_samples[qid]['negative'].add(corpus[neg_id])

train_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train.tsv.gz')
if not os.path.exists(train_filepath):
    logging.info("Download " + os.path.basename(train_filepath))
    util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz', train_filepath)

cnt = 0
with gzip.open(train_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn, unit_scale=True):
        data_file = open("./train_data.txt", "a")
        qid, pos_id, neg_id = line.strip().split()

        if qid in dev_samples:
            continue

        query = queries[qid]
        if (cnt % (pos_neg_ration + 1)) == 0:
            passage = corpus[pos_id]
            label = 1
        else:
            passage = corpus[neg_id]
            label = 0

        injection_scores = []
        for model_pre in [model1, model2, model3]:
            if model_pre.config.num_labels > 1:
                score = model_pre.predict([query, passage], apply_softmax=True, show_progress_bar=False)[:, 1].tolist()
            else:
                score = model_pre.predict([query, passage], show_progress_bar=False).tolist()
            injection_scores.append(score)
        injection_scores = np.array(injection_scores)
        injection = np.mean(injection_scores)
        data_file.write(query + "\t" + str(int(injection)) + " [SEP] " + passage + "\t" + str(label) + "\n")

        cnt += 1

        if cnt >= max_train_samples:
            break
        if cnt % 10000 == 0:
            print(f"{cnt} completed.")

        data_file.close()

count = 0
with open("./train_data.txt", "r") as file:
    while True:
        count += 1
        line = file.readline()
        text = line.split("\t")
        if len(text) == 3:
            train_samples.append(InputExample(texts=[text[0], text[1]], label=int(text[2])))
        if count % 10000 == 0:
            print(f"Line {count} loaded.")
        if not line:
            break

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=5000,
          output_path=model_save_path,
          use_amp=True)

