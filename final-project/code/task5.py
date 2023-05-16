"""
Group 15, Task 5
"""

from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import InputExample
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import logging
from collections import defaultdict
import numpy as np
import sys
import pytrec_eval
#from google.colab import drive

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
#drive.mount('/content/gdrive')
base_path = "./gdrive/MyDrive/cross-encoder-reranker-ir-course-2023/"
model_save_path = "./gdrive/MyDrive/cross-encoder-reranker-ir-course-2023/finetuned_models/cross-encoder-cross-encoder/"

# Inference with the three trained models on the train set =============================================================

