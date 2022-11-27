from asyncio.log import logger
import torch
import numpy as np
import os
import random
import warnings
import time
import yaml
import argparse
import sys
import logging

from utils.logger import Logger
from utils import utils
from process import AttrChanger


warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False

TROJAI_R6_DATASET_DIR = '/data/share/trojai/trojai-round6-v2-dataset/'

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def ami(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath):

    start_time = time.time()

    # set logger
    model_id = model_filepath.split('/')[-2]
    logging_filepath = os.path.join(scratch_dirpath,model_id + '.log')
    logger = Logger(logging_filepath, logging.DEBUG, logging.DEBUG)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parse model arch info
    arch_name = utils.arch_parser(tokenizer_filepath)

    # load config
    config_filepath = './config/config.yaml'
    with open(config_filepath) as f:
        config = yaml.safe_load(f)

    # fix seed
    seed_torch(config['seed'])

    # load models
    # trojai r6 seperates the classification probe and transformer backbone as two models
    # embedding_backbone: transformer model
    # target_model: classification probe
    # tokenizer: pre-trained tokenizer for each transformer arch

    embedding_backbone,target_model,tokenizer = utils.load_models(arch_name,model_filepath,device)

    embedding_backbone.eval()
    target_model.eval()

    scanner = AttrChanger(embedding_backbone,target_model,tokenizer,arch_name,device,logger,config)
    #TODO: iterate all samples -- seems not necessary?
    #TODO: find 'important tokens' with large attention
    sample_list = utils.load_data(0,examples_dirpath)
    sorted_index=scanner.sort_by_attn(sample_list)

    # enumerate all possible trigger options for scanning
    keep_idx, clique, effect_lens = scanner.change_one_token(sorted_index)
    clique = scanner.check_clique(clique, effect_lens, keep_idx)

    logits_changes = []

    #best_loss = 1e+10

    #for change in token_changes:
    #    #TODO: get logits


    #TODO: now we get the set of real important tokens, the clique

    #TODO: verify if the clique is interpretable

    for scanning_result in scanning_result_list:
        logger.result_collection('victim label: {}  target label: {} position: {}  trigger: {}  loss: {:.6f}'.format(scanning_result['victim_label'],scanning_result['target_label'],scanning_result['position'], scanning_result['trigger'],scanning_result['loss']))

    end_time = time.time()
    scanning_time = end_time - start_time
    logger.result_collection('Scanning Time: {:.2f}s'.format(scanning_time))

    logger.best_result('victim label: {}  target label: {} position: {}  trigger: {}  loss: {:.6f}'.format(best_estimation['victim_label'],best_estimation['target_label'],best_estimation['position'], best_estimation['trigger'],best_estimation['loss']))

    return best_estimation['loss'],scanning_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default=f'{TROJAI_R6_DATASET_DIR}/models/id-00000000/model.pt')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default=f'{TROJAI_R6_DATASET_DIR}/tokenizers/GPT-2-gpt2.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.', default=f'{TROJAI_R6_DATASET_DIR}/models/id-00000000/clean_example_data')

    args = parser.parse_args()

    best_loss,scanning_time = ami(args.model_filepath, args.tokenizer_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)








