from simcse import SimCSE
from nlp_ami import *
import torch
import numpy as np
import os
import random
import warnings
import time
import sys
import logging
import json
from utils import utils
from detect import *
from sklearn.cluster import KMeans
import torch.nn.functional as F
#from process import AttrChanger
import logging
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Set the logging level to ERROR to only display error messages
logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False

TROJAI_R6_DATASET_DIR = '/data/share/trojai/trojai-round6-v2-dataset/'
TROJAI_R7_DATASET_DIR = '/data/share/trojai/trojai-round7-v2-dataset/'

def softmax_stable(x):
        return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

def predict_nli(model, tokenizer, premises, hypotheses, batch_size=32):
    # Tokenize the input
    num_examples = len(premises)
    predicted_labels = []
    for i in range(0, num_examples, batch_size):
        batch_premises = premises[i:i+batch_size]
        batch_hypotheses = hypotheses[i:i+batch_size]
        pairs = [(a,b) for a, b in zip(batch_premises, batch_hypotheses)]
        inputs = tokenizer.batch_encode_plus(pairs, padding='max_length', truncation=True, return_tensors='pt')
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_labels += [logit.argmax().item() for logit in logits]
    return predicted_labels


def nli_by_label(clean_texts, clean_labels, poisoned_texts, poisoned_labels, reference):
    MODEL_NAME = 'textattack/roberta-base-MNLI'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).cuda()
    clean_check_neg = predict_nli(model, tokenizer, clean_texts, [reference[0]]*len(clean_texts))
    clean_check_pos = predict_nli(model, tokenizer, clean_texts, [reference[1]]*len(clean_texts))
    poisoned_check_neg = predict_nli(model, tokenizer, poisoned_texts, [reference[0]]*len(poisoned_texts))
    poisoned_check_pos = predict_nli(model, tokenizer, poisoned_texts, [reference[1]]*len(poisoned_texts))
    print(clean_check_neg, clean_check_pos, poisoned_check_neg, poisoned_check_pos)


def simcse_cmp(clean_texts, clean_labels, poisoned_texts, poisoned_labels, reference):
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    clean_preds = model.similarity( clean_texts, reference)
    preds = np.argmax(clean_preds, axis=1)
    clean_consistency = sum(preds == np.array(clean_labels))
    poisoned_consistency = 'None'
    poisoned_preds = model.similarity(poisoned_texts, reference)
    preds = np.argmax(poisoned_preds, axis=1)
    poisoned_consistency = sum(preds == np.array(poisoned_labels))
    print(clean_consistency, poisoned_consistency)
    clean_preds, poisoned_preds = softmax(clean_preds, axis=1), softmax(poisoned_preds, axis=1)
    mixed = np.concatenate([clean_preds, poisoned_preds], axis=0)
    clustering(mixed, (len(clean_preds), len(poisoned_preds)))


def nli(mid, TROJAI_DIR=TROJAI_R6_DATASET_DIR):
    # fix seed
    model_info = read_config(TROJAI_DIR)
    start_time = time.time()
    # set logger
    mname = f'id-{mid:08}'
    print('Analyze', mname)
    subject_model = torch.load(TROJAI_DIR+f'models/{mname}/model.pt').cuda()
    emb_arch = model_info[mname][3]
    emb_flavor = model_info[mname][4]
    seed_torch(int(model_info[mname][1]))
    if model_info[mname][0] == 'False':
        return
    embedding_model, tokenizer, max_input_len = utils.load_embedding(TROJAI_DIR, emb_arch, emb_flavor)
    subject_model.eval()
    clean_path = TROJAI_DIR+f'models/{mname}/clean_example_data'
    poisoned_path = TROJAI_DIR+f'models/{mname}/poisoned_example_data'
    clean_texts, clean_labels = utils.read_r6_eg_directory(clean_path)
    poisoned_texts, poisoned_labels = utils.read_r6_eg_directory(poisoned_path)
    reference = ['This is a negative review for the product', 'This is a positive review for the product']
    nli_by_label(clean_texts, clean_labels, poisoned_texts, poisoned_labels, reference)
    exit(0)


if __name__ == '__main__':
    #sentence = 'Doesn\'t have the power to sharpen!'
    #imply(sentence)
    for mid in list(range(12, 48)): #+list(range(30,48)):
        nli(mid, TROJAI_R6_DATASET_DIR)








