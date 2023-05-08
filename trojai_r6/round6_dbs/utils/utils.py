import torch 
import os 
import json 
import transformers
from transformers import BertForSequenceClassification

def arch_parser(tokenizer_filepath):
    if 'BERT-bert-base-uncased.pt' in tokenizer_filepath:
        arch_name = 'bert'
    
    elif 'DistilBERT' in tokenizer_filepath:
        arch_name = 'distilbert'
    
    elif 'GPT-2-gpt2.pt' in tokenizer_filepath:
        arch_name = 'gpt'
    
    else:
        print('arch not support!')
        exit()
    
    
    return arch_name

def gen_large_dataset(data_dirpath,data_size,large_data_dirpath):
    

    
    dataset_dict = {}
    
    dataset_dict['0'] = [] 
    dataset_dict['1'] = [] 
    
    if not os.path.exists(large_data_dirpath):
        os.makedirs(large_data_dirpath)
    
    
    for model_id in sorted(os.listdir(data_dirpath)):
        
        example_dirpath = data_dirpath + model_id + '/clean_example_data/'

        
        for example_id in sorted(os.listdir(example_dirpath)):
            label = example_id.split('_')[1]
            
            with open(os.path.join(example_dirpath,example_id),'r') as f:
                
                data = f.read()
                data = data.strip('\n').split() 
                

                                        
                if len(dataset_dict[label]) < data_size: 
                    dataset_dict[label].append(data)

            
            if len(dataset_dict['0']) == data_size and len(dataset_dict['1']) == data_size:
                # print(len(dataset_dict['0']))
                # print(len(dataset_dict['1']))
                # print(dataset_dict)
                

                
                with open(os.path.join(large_data_dirpath,'large_dataset.json'),'w') as f:
                    json.dump(dataset_dict,f)
                    
                exit() 
                
                
        
        
def load_hiddenkill_model(model_filepath,device):
    target_model = torch.load(model_filepath,map_location=device)
    # benign_model = torch.load('/data/share/trojai/trojai-round6-v2-dataset/models/id-00000006/model.pt').to(device)
    benign_model = torch.load('/data/share/HiddenKiller_share/experiments/many_clean_models/many-bert-clean-train-sst-2-1022351990.pt',map_location=device)
    tokenizer = transformers.DistilBertTokenizer.from_pretrained('bert-base-uncased')
    print(target_model)
    
    return target_model,benign_model,tokenizer
        

def load_sos_model(model_filepath,device):
    target_model = BertForSequenceClassification.from_pretrained(model_filepath).to(device)
    # target_model = target_model.load_state_dict(target_model)
    # benign_model = torch.load('/data/share/trojai/trojai-round6-v2-dataset/models/id-00000006/model.pt').to(device)
    benign_model = BertForSequenceClassification.from_pretrained('/data/share/SOS/Imdb_test/clean_model').to(device)
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    
    
    return target_model,benign_model,tokenizer
        
    
    
    
    
    

def load_model(arch_name,trailer_model_filepath,transformer_filepath,tokenizer_filepath,device):
    
    target_model = torch.load(trailer_model_filepath).to(device)
    # target_model = BertForSequenceClassification.from_pretrained(trailer_model_filepath).to(device)
    transformer_model = torch.load(transformer_filepath).to(device)
    # tokenizer = torch.load(tokenizer_filepath)
    if arch_name == 'distilbert':
        tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        benign_model = torch.load('/data/share/trojai/trojai-round6-v2-dataset/models/id-00000006/model.pt').to(device)
    elif arch_name == 'gpt':
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
        # tokenizer = torch.load('/data/share/trojai/trojai-round6-v2-dataset/tokenizers/GPT-2-gpt2.pt')
        benign_model = torch.load('/data/share/trojai/trojai-round6-v2-dataset/models/id-00000001/model.pt').to(device)
    
    else: 
        print('arch not support!')
        exit() 
        

    if not hasattr(tokenizer,'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    model_dirpath,_ = os.path.split(trailer_model_filepath)
    with open(os.path.join(model_dirpath,'config.json')) as json_file:
        config = json.load(json_file)
    
        if config['poisoned']:
            source_label_list = []
            target_label_list = [] 
            trigger_phrase_list = []
            
            trigger_info = config['triggers']
            for idx in range(len(trigger_info)):
                item = trigger_info[idx]
                source_label_list.append(item['source_class'])
                target_label_list.append(item['target_class'])
                trigger_phrase_list.append(item['text'])
        
        else:
            
            source_label_list = target_label_list = trigger_phrase_list = None 
    
        
        # for source_label,target_label,text in zip(source_label_list,target_label_list,trigger_phrase_list):
        #     print(source_label,target_label,text)
    
    return target_model,transformer_model,benign_model,tokenizer,source_label_list,target_label_list,trigger_phrase_list
    
            
            

if __name__ == '__main__':
    gen_large_dataset('/data/share/trojai/trojai-round5-v2-dataset/models/',500,'/home/shen447/data/shen447/trojai_r5/NLP_Constraint_Optimization_R5/constrain_opt/large_dataset/')
    