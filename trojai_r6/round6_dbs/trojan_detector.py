import torch
import numpy as np
import os
import random
import warnings
import time
import yaml
import argparse
import sys
import string
from torch.nn import CrossEntropyLoss
from sklearn.decomposition import PCA
from utils.utils import load_model,arch_parser,gen_large_dataset,load_hiddenkill_model,load_sos_model
from data_loader.distilbert_dataset import Distilbert_Dataset
from data_loader.gpt_dataset import GPT_Dataset
from data_loader.hiddenkiller_dataset import HiddenKiller_Dataset
from scanner.distilbert_scanner import Distilbert_Scanner
from scanner.gpt_scanner import GPT_Scanner
from scanner.hiddenkiller_scanner import Hiddenkiller_Scanner
from scanner.bkd_scanner import BKD_Scanner
from scanner.sos_scanner import SOS_Scanner

warnings.filterwarnings('ignore')

torch.backends.cudnn.enabled = False

ENV_DIRPATH = '/home/shen447/data/shen447/trojai_r6/NLP_Constraint_Optimization_R6/'
import torch.nn as nn
class self_learning_poisoner(nn.Module):

    def __init__(self, nextBertModel, N_BATCH, N_CANDIDATES, N_LENGTH, N_EMBSIZE):
        super(self_learning_poisoner, self).__init__()
        self.nextBertModel = nextBertModel
        self.nextDropout = nn.Dropout(DROPOUT_PROB)
        self.nextClsLayer = nn.Linear(N_EMBSIZE, NUMLABELS)

        # Hyperparameters
        self.N_BATCH = N_BATCH
        self.N_CANDIDATES = N_CANDIDATES
        self.N_LENGTH = N_LENGTH
        self.N_EMBSIZE = N_EMBSIZE
        self.N_TEMP = TEMPERATURE # Temperature for Gumbel-softmax

        self.relevance_mat = nn.Parameter(data=torch.zeros((self.N_LENGTH, self.N_EMBSIZE)).cuda(0), requires_grad=True).cuda(0).float()
        self.relevance_bias = nn.Parameter(data=torch.zeros((self.N_LENGTH, self.N_CANDIDATES)))

    def set_temp(self, temp):
        self.N_TEMP = temp

    def get_poisoned_input(self, sentence, candidates, gumbelHard=False, sentence_ids=[], candidate_ids=[]):
        length = sentence.size(0) # Total length of poisonable inputs
        repeated = sentence.unsqueeze(2).repeat(1, 1, self.N_CANDIDATES, 1)
        difference = torch.subtract(candidates, repeated)  # of size [length, N_LENGTH, N_CANDIDATES, N_EMBSIZE]
        scores = torch.matmul(difference, torch.reshape(self.relevance_mat,
            [1, self.N_LENGTH, self.N_EMBSIZE, 1]).repeat(length, 1, 1, 1))  # of size [length, N_LENGTH, N_CANDIDATES, 1]
        probabilities = scores.squeeze(3)  # of size [length, N_LENGTH, N_CANDIDATES]
        probabilities += self.relevance_bias.unsqueeze(0).repeat(length, 1, 1)
        probabilities_sm = gumbel_softmax(probabilities, self.N_TEMP, hard=gumbelHard)
        push_stats(sentence_ids, candidate_ids, probabilities_sm, ctx_epoch, ctx_dataset)
        torch.reshape(probabilities_sm, (length, self.N_LENGTH, self.N_CANDIDATES))
        poisoned_input = torch.matmul(torch.reshape(probabilities_sm,
            [length, self.N_LENGTH, 1, self.N_CANDIDATES]), candidates)
        poisoned_input_sq = poisoned_input.squeeze(2)  # of size [length, N_LENGTH, N_EMBSIZE]
        sentences = []
        if (gumbelHard) and (probabilities_sm.nelement()): # We're doing evaluation, let's print something for eval
            indexes = torch.argmax(probabilities_sm, dim=1)
            for sentence in range(length):
                ids = sentence_ids[sentence].tolist()
                idxs = indexes[sentence*self.N_LENGTH:(sentence+1)*self.N_LENGTH]
                frm, to = ids.index(TOKENS['CLS']), ids.index(TOKENS['SEP'])
                ids = [candidate_ids[sentence][j][i] for j, i in enumerate(idxs)]
                ids = ids[frm+1:to]
                sentences.append(tokenizer.decode(ids))
        #    sentences = [tokenizer.decode(seq) for seq in poisoned_input_sq]
            pp.pprint(sentences[:10]) # Sample 5 sentences
        return [poisoned_input_sq, sentences]

    def forward(self, seq_ids, to_poison_candidates_ids, attn_masks, gumbelHard=False):
        '''
        Inputs:
            -sentence: Tensor of shape [N_BATCH, N_LENGTH, N_EMBSIZE] containing the embeddings of the sentence to poison
            -candidates: Tensor of shape [N_BATCH, N_LENGTH, N_CANDIDATES, N_EMBSIZE] containing the candidates to replace
        '''
        position_ids = torch.tensor([i for i in range(self.N_LENGTH)]).cuda(gpu_id)
        position_cand_ids = position_ids.unsqueeze(1).repeat(1, self.N_CANDIDATES).cuda(gpu_id)
        to_poison_candidates = word_embeddings(to_poison_candidates_ids) + position_embeddings(position_cand_ids)
        [to_poison_ids, no_poison_ids] = seq_ids
        to_poison = word_embeddings(to_poison_ids) + position_embeddings(position_ids)
        no_poison = word_embeddings(no_poison_ids) + position_embeddings(position_ids)
        [to_poison_attn_masks, no_poison_attn_masks] = attn_masks
        poisoned_input, _ = self.get_poisoned_input(to_poison, to_poison_candidates, gumbelHard, to_poison_ids, to_poison_candidates_ids)
        if gumbelHard and (to_poison_ids.nelement()):
            pp.pprint([tokenizer.decode(t.tolist()) for t in to_poison_ids[:10]])
            print("--------")

        total_input = torch.cat((poisoned_input, no_poison), dim=0)
        total_attn_mask = torch.cat((to_poison_attn_masks, no_poison_attn_masks), dim=0)

        # Run it through classification
        output = self.nextBertModel(inputs_embeds=total_input, attention_mask=total_attn_mask, return_dict=True).last_hidden_state
        #output = self.nextDropout(output)
        logits = self.nextClsLayer(output[:, 0])

        return logits


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def example_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath,seed):
    print('seed: {}'.format(seed))
    seed_torch(seed)
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('model_filepath = {}'.format(model_filepath))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))
    arch_name = arch_parser(tokenizer_filepath)
    arch_name = 'hiddenkiller'

    if arch_name == 'distilbert':
        distilbert_scanning(arch_name,tokenizer_filepath,model_filepath,examples_dirpath,device,seed)
    elif arch_name == 'gpt':
        gpt_scanning(arch_name,tokenizer_filepath,model_filepath,examples_dirpath,device,seed)
    elif arch_name == 'hiddenkiller':
        hiddenkiller_scanning(tokenizer_filepath,model_filepath,examples_dirpath,device,seed)
    elif arch_name == 'bkd':
        bkd_scanning(tokenizer_filepath,model_filepath,examples_dirpath,device,seed)
    elif arch_name == 'sos':
        sos_scanning(tokenizer_filepath,model_filepath,examples_dirpath,device,seed)
    end_time = time.time()
    print('time cost: {}'.format(end_time-start_time))


def gpt_scanning(arch_name,tokenizer_filepath,model_filepath,examples_dirpath,device,seed):

    transformer_filepath  = '/data/share/trojai/trojai-round6-v2-dataset/embeddings/GPT-2-gpt2.pt'
    cls_token_is_first = False

    model_id = model_filepath.split('/')[-2]
    result_trigger_dict = {}
    result_loss_dict = {}

    user_config = yaml.load(open(os.path.join(ENV_DIRPATH, 'constrain_opt/config/gpt_config.yaml'),'r'))

    target_model, transformer_model,benign_model, tokenizer, poisoned_source_label_list,poisoned_target_label_list, trigger_phrase_list \
        = load_model(arch_name,model_filepath,transformer_filepath,tokenizer_filepath,device)


    # if trigger_phrase_list is not None:
    #     key_words = trigger_phrase_list[0]
    print(poisoned_source_label_list)
    print(poisoned_target_label_list)
    print(trigger_phrase_list)

    # print(tokenizer.encode('[ Love, love this, makes cutting fabric so much easier and quicker. Such a time saver.'))
    # print(tokenizer.encode(' *'))

    # exit()

    key_words = None

    transformer_model.eval()
    target_model.eval()
    benign_model.eval()

    source_label_list = [0,1]
    target_label_list = [1,0]

    insert_position_list = ['first_half','second_half']
    # insert_position_list = ['second_half']
    source_label_list = poisoned_source_label_list
    target_label_list = poisoned_target_label_list


    # print(target_model)
    # print(transformer_model)

    #TODO add an individual char scanning
    #!!!!!!!!!!!!!!!!!!!!!!!

    for source_label, target_label in zip(source_label_list,target_label_list):

        if trigger_phrase_list is not None:
            if source_label in poisoned_source_label_list:
                index = poisoned_source_label_list.index(source_label)
                key_words = trigger_phrase_list[index]

            else:
                key_words = None
        else:
            key_words = None

        tmp_best_loss = 1e+10
        tmp_best_trigger = 'TROJAI_GREAT'

        for insert_pos in insert_position_list:
            # 1635
            if insert_pos == 'first_half':
                dataset = GPT_Dataset(tokenizer,transformer_model,target_model,benign_model, user_config['trigger_placeholder'],9,user_config['trigger_len'],examples_dirpath,user_config['max_len'],insert_pos,user_config['use_large_dataset'],user_config['dataset_len'],seed)
            elif insert_pos == 'second_half':
                dataset = GPT_Dataset(tokenizer,transformer_model,target_model,benign_model, user_config['trigger_placeholder'],1635,user_config['trigger_len'],examples_dirpath,user_config['max_len'],insert_pos,user_config['use_large_dataset'],user_config['dataset_len'],seed)
            tensor_dict = dataset.gen_dataset(source_label)

            print(tensor_dict['input_ids'].size())
            # exit()


            scanner = GPT_Scanner(target_model,transformer_model,benign_model, tokenizer,tensor_dict,key_words,user_config,cls_token_is_first)
            best_loss, best_trigger,_,loss_list,is_one_hot,traj_mat = scanner.scanning(source_label,target_label,insert_pos)
            if key_words is not None:

                flatten_traj_mat = torch.reshape(traj_mat,(traj_mat.shape[0],traj_mat.shape[1]*traj_mat.shape[2]))
                print(flatten_traj_mat.size())

                pca = PCA(n_components=2,random_state=1)
                traj_points = pca.fit_transform(flatten_traj_mat)
                directions = pca.components_
                print(traj_points.shape)
                print(directions.shape)




                landscape_loss_list = scanner.draw_loss_landscape(key_words,target_label,directions,traj_points)
                log_filepath = 'train_{}/trojan'.format(arch_name)
                landscape_dirpath= os.path.join(ENV_DIRPATH,'constrain_opt/result_log/{}/{}/'.format(log_filepath, model_id))
                landscape_filepath = os.path.join(landscape_dirpath,'landscape.npy')
                traj_filepath = os.path.join(landscape_dirpath,'traj.npy')
                np.save(landscape_filepath,landscape_loss_list)
                np.save(traj_filepath,traj_points)

            exit()
            if best_loss < tmp_best_loss:
                tmp_best_loss = best_loss
                tmp_best_trigger = best_trigger



        result_trigger_dict[source_label] = tmp_best_loss
        result_loss_dict[source_label] = tmp_best_trigger


    for source_label,trigger in result_trigger_dict.items():
        print('source label: {} target label: {}  best trigger: {}'.format(source_label, 1-source_label,trigger))

    for source_label,loss in result_loss_dict.items():
        print('source label: {} target label: {}  best loss: {}'.format(source_label, 1-source_label,loss))


def hiddenkiller_scanning(tokenizer_filepath,model_filepath,examples_dirpath,device,seed):
    result_trigger_dict = {}
    result_loss_dict = {}
    user_config = yaml.load(open(os.path.join(ENV_DIRPATH, 'constrain_opt/config/hiddenkiller_config.yaml'),'r'))
    target_model,benign_model,tokenizer = load_hiddenkill_model(model_filepath,device)
    target_model.eval()
    benign_model.eval()
    source_label_list = [0,1]
    target_label_list = [1,0]
    insert_position_list = ['first_half']
    for source_label, target_label in zip(source_label_list,target_label_list):
        key_words = None

        tmp_best_loss = 1e+10
        tmp_best_trigger = 'TROJAI_GREAT'

        for insert_pos in insert_position_list:
            dataset = HiddenKiller_Dataset(tokenizer, user_config['trigger_placeholder'],1008,user_config['trigger_len'],examples_dirpath,user_config['max_len'],insert_pos,user_config['dataset_len'],seed)
            tensor_dict = dataset.gen_dataset(source_label)
            print(tensor_dict['input_ids'].size())
            scanner = Hiddenkiller_Scanner(target_model,benign_model, tokenizer,tensor_dict,key_words,user_config)
            best_loss, best_trigger,_,_ = scanner.scanning(source_label,target_label,insert_pos)

            if best_loss < tmp_best_loss:
                tmp_best_loss = best_loss
                tmp_best_trigger = best_trigger
        result_trigger_dict[source_label] = tmp_best_loss
        result_loss_dict[source_label] = tmp_best_trigger
    for source_label,trigger in result_trigger_dict.items():
        print('source label: {} target label: {}  best trigger: {}'.format(source_label, 1-source_label,trigger))

    for source_label,loss in result_loss_dict.items():
        print('source label: {} target label: {}  best loss: {}'.format(source_label, 1-source_label,loss))


def sos_scanning(tokenizer_filepath,model_filepath,examples_dirpath,device,seed):


    result_trigger_dict = {}
    result_loss_dict = {}
    user_config = yaml.load(open(os.path.join(ENV_DIRPATH, 'constrain_opt/config/sos_config.yaml'),'r'))

    target_model,benign_model,tokenizer = load_sos_model(model_filepath,device)

    # print(tokenizer.encode('friends weekend store'))
    # exit()




    target_model.eval()
    benign_model.eval()
    source_label_list = [0]
    target_label_list = [1]

    insert_position_list = ['first_half']

    for source_label, target_label in zip(source_label_list,target_label_list):



        key_words = 'friends weekend store'

        tmp_best_loss = 1e+10
        tmp_best_trigger = 'TROJAI_GREAT'

        for insert_pos in insert_position_list:
            dataset = HiddenKiller_Dataset(tokenizer, user_config['trigger_placeholder'],1008,user_config['trigger_len'],examples_dirpath,user_config['max_len'],insert_pos,user_config['dataset_len'],seed)
            tensor_dict = dataset.gen_dataset(source_label)

            print(tensor_dict['input_ids'].size())
            # exit()


            scanner = SOS_Scanner(target_model,benign_model, tokenizer,tensor_dict,key_words,user_config)
            best_loss, best_trigger,_,_ = scanner.scanning(source_label,target_label,insert_pos)

            if best_loss < tmp_best_loss:
                tmp_best_loss = best_loss
                tmp_best_trigger = best_trigger



        result_trigger_dict[source_label] = tmp_best_loss
        result_loss_dict[source_label] = tmp_best_trigger


    for source_label,trigger in result_trigger_dict.items():
        print('source label: {} target label: {}  best trigger: {}'.format(source_label, 1-source_label,trigger))

    for source_label,loss in result_loss_dict.items():
        print('source label: {} target label: {}  best loss: {}'.format(source_label, 1-source_label,loss))





def bkd_scanning(tokenizer_filepath,model_filepath,examples_dirpath,device,seed):




    result_trigger_dict = {}
    result_loss_dict = {}
    user_config = yaml.load(open(os.path.join(ENV_DIRPATH, 'constrain_opt/config/hiddenkiller_config.yaml'),'r'))

    target_model,benign_model,tokenizer = load_hiddenkill_model(model_filepath,device)





    target_model.eval()
    benign_model.eval()
    source_label_list = [0,1]
    target_label_list = [1,0]

    insert_position_list = ['first_half']

    for source_label, target_label in zip(source_label_list,target_label_list):



        key_words = None

        tmp_best_loss = 1e+10
        tmp_best_trigger = 'TROJAI_GREAT'

        for insert_pos in insert_position_list:
            dataset = HiddenKiller_Dataset(tokenizer, user_config['trigger_placeholder'],1008,user_config['trigger_len'],examples_dirpath,user_config['max_len'],insert_pos,user_config['dataset_len'],seed)
            tensor_dict = dataset.gen_dataset(source_label)

            print(tensor_dict['input_ids'].size())
            # exit()


            scanner = BKD_Scanner(target_model,benign_model, tokenizer,tensor_dict,key_words,user_config)
            best_loss, best_trigger,_,_ = scanner.scanning(source_label,target_label,insert_pos)

            if best_loss < tmp_best_loss:
                tmp_best_loss = best_loss
                tmp_best_trigger = best_trigger



        result_trigger_dict[source_label] = tmp_best_loss
        result_loss_dict[source_label] = tmp_best_trigger


    for source_label,trigger in result_trigger_dict.items():
        print('source label: {} target label: {}  best trigger: {}'.format(source_label, 1-source_label,trigger))

    for source_label,loss in result_loss_dict.items():
        print('source label: {} target label: {}  best loss: {}'.format(source_label, 1-source_label,loss))





def distilbert_scanning(arch_name,tokenizer_filepath,model_filepath,examples_dirpath,device,seed):

    transformer_filepath  = '/data/share/trojai/trojai-round6-v2-dataset/embeddings/DistilBERT-distilbert-base-uncased.pt'
    cls_token_is_first = True

    model_id = model_filepath.split('/')[-2]
    result_trigger_dict = {}
    result_loss_dict = {}

    user_config = yaml.load(open(os.path.join(ENV_DIRPATH, 'constrain_opt/config/distilbert_config.yaml'),'r'))

    target_model, transformer_model,benign_model, tokenizer, poisoned_source_label_list,poisoned_target_label_list, trigger_phrase_list \
        = load_model(arch_name,model_filepath,transformer_filepath,tokenizer_filepath,device)


    # if trigger_phrase_list is not None:
    #     key_words = trigger_phrase_list[0]
    print(poisoned_source_label_list)
    print(poisoned_target_label_list)
    print(trigger_phrase_list)

    # print(tokenizer.encode('[ Love, love this, makes cutting fabric so much easier and quicker. Such a time saver.'))


    # exit()

    key_words = None

    transformer_model.eval()
    target_model.eval()
    benign_model.eval()

    source_label_list = [0,1]
    target_label_list = [1,0]

    insert_position_list = ['first_half','second_half']
    # insert_position_list = ['first_half']
    # source_label_list = poisoned_source_label_list
    # target_label_list = poisoned_target_label_list


    # print(target_model)
    # print(transformer_model)

    #TODO add an individual char scanning
    #!!!!!!!!!!!!!!!!!!!!!!!

    for source_label, target_label in zip(source_label_list,target_label_list):

        if trigger_phrase_list is not None:
            if source_label in poisoned_source_label_list:
                index = poisoned_source_label_list.index(source_label)
                key_words = trigger_phrase_list[index]

            else:
                key_words = None
        else:
            key_words = None

        tmp_best_loss = 1e+10
        tmp_best_trigger = 'TROJAI_GREAT'

        for insert_pos in insert_position_list:

            dataset = Distilbert_Dataset(tokenizer,transformer_model,target_model,benign_model, user_config['trigger_placeholder'],1008,user_config['trigger_len'],examples_dirpath,user_config['max_len'],insert_pos,user_config['use_large_dataset'],user_config['dataset_len'],seed)
            tensor_dict = dataset.gen_dataset(source_label)

            print(tensor_dict['input_ids'].size())
            # exit()


            scanner = Distilbert_Scanner(target_model,transformer_model,benign_model, tokenizer,tensor_dict,key_words,user_config,cls_token_is_first)
            best_loss, best_trigger,_,_ = scanner.scanning(source_label,target_label,insert_pos)

            if best_loss < tmp_best_loss:
                tmp_best_loss = best_loss
                tmp_best_trigger = best_trigger



        result_trigger_dict[source_label] = tmp_best_loss
        result_loss_dict[source_label] = tmp_best_trigger


    for source_label,trigger in result_trigger_dict.items():
        print('source label: {} target label: {}  best trigger: {}'.format(source_label, 1-source_label,trigger))

    for source_label,loss in result_loss_dict.items():
        print('source label: {} target label: {}  best loss: {}'.format(source_label, 1-source_label,loss))





if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/model.pt')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./tokenizers/google-electra-small-discriminator.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.', default='./model/example_data')
    parser.add_argument('--output_mode',type=str,default='stdout')
    parser.add_argument('--mode',type=str,default='train')
    parser.add_argument('--seed',type=int,default=1)
    args = parser.parse_args()

    # model_id = args.model_filepath.split('/')[-1].split('.')[0].split('-')[-1]
    model_id = args.model_filepath.split('/')[-2].split('_')[-1]

    arch_name = arch_parser(args.tokenizer_filepath)

    arch_name = 'sos'

    # example_trojan_detector(args.model_filepath, args.tokenizer_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath,args.seed)




    if args.output_mode == 'log':


        log_filepath = '{}_{}/{}'.format(args.mode,arch_name, args.result_filepath)

        log_all_path = os.path.join(ENV_DIRPATH,'constrain_opt/result_log/{}/{}/'.format(log_filepath, model_id))

        isExist = os.path.exists(log_all_path)
        if not isExist:
            os.makedirs(log_all_path)


        # example_trojan_detector(args.model_filepath, args.tokenizer_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath,args.seed)

        original_stdout = sys.stdout

        with open(os.path.join(log_all_path,'loss_log.txt'), 'a') as f:
            sys.stdout = f # Change the standard output to the file we created.

            example_trojan_detector(args.model_filepath, args.tokenizer_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath,args.seed)

            sys.stdout = original_stdout

    else:
        example_trojan_detector(args.model_filepath, args.tokenizer_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath,args.seed)


