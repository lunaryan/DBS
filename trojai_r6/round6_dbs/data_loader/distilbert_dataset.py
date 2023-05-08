import torch 
import numpy as np 
import random 
import os 
from random import choice
import json 


# target_model,tokenizer,placeholder_token,opt_len,examples_dirpath,trigger_type,insert_type,seed

class Distilbert_Dataset:
    def __init__(self,tokenizer,transformer_model,target_model,benign_model,placeholder_token,placeholder_ids,opt_len,examples_dirpath,max_len,insert_type,use_large_dataset,dataset_len,seed):
        
        random.seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transformer_model = transformer_model 
        self.benign_model = benign_model 
        self.target_model = target_model 
        self.tokenizer = tokenizer 
        self.placeholder =  [] 
        self.opt_len = opt_len
        self.max_len = max_len
        self.tokenized_max_len = self.max_len+100
        self.insert_type = insert_type
        self.use_large_dataset = use_large_dataset
        self.dataset_len = dataset_len
        self.placeholder_ids = placeholder_ids
        self.cls_token_is_first = True 
        
        if self.use_large_dataset:

            self.examples_dirpath = '/home/shen447/data/shen447/trojai_r6/NLP_Constraint_Optimization_R6/ASCC/large_dataset/'
            
        else: 
            
            fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
            fns.sort()

            
            self.examples_dirpath = fns
            
        

        self.raw_data_dict = {}
        self.insert_index_dict = {}

        # self.placeholder = placeholder_token * self.opt_len
        for num in range(self.opt_len):
            self.placeholder.append(placeholder_token)
            

        
    def load_data(self):
        
        
        if self.use_large_dataset: 
            example_filepath = os.path.join(self.examples_dirpath,'large_dataset.json')

            with open(example_filepath,'r') as json_file:
                self.raw_data_dict = json.load(json_file)


                
        
        else:  
            
            for fn in self.examples_dirpath:
                if fn.endswith('.txt'):
                    
                    source_label = fn.split('_')[-3]

                    
                    if source_label not in self.raw_data_dict:
                        self.raw_data_dict[source_label] = []
                        self.insert_index_dict[source_label] = [] 
                    
                    
                    with open(fn,'r') as fh:
                        text = fh.read()
                        text = text.strip('\n').split() 
                        if len(text) < self.max_len:
                            self.raw_data_dict[source_label].append(text)


            





    
    def insert_placeholder(self):
        
        for label in self.raw_data_dict.keys():
            for idx in range(len(self.raw_data_dict[label])):
                tmp_data = self.raw_data_dict[label][idx]
                
                

                
                if len(tmp_data) > self.max_len:
                    tmp_data = tmp_data[:self.max_len]
                
                # print(idx)
                # print(len(tmp_data))
                
                
                
                # if self.insert_type == 'random' and len(tmp_data) > 1:
                #     max_index = min(len(tmp_data),self.max_len-100)
                #     insert_index = choice(range(1,max_index))
                
                if self.insert_type == 'first_half':
                    # insert_index = 0
                    # tmp_data[insert_index:insert_index] = self.placeholder
                    tmp_data = self.placeholder + tmp_data 
                
                elif self.insert_type == 'second_half':
                    # insert_index = -1
                    tmp_data = tmp_data + self.placeholder
                else: 
                    print('insert type not support!')
                    exit() 
                    
                    
                

                # inserted_tmp_data = tmp_data[:insert_index] + self.placeholder +  tmp_data[insert_index]
                # tmp_data[insert_index:insert_index] = self.placeholder
                self.raw_data_dict[label][idx] = ' '.join(map(str, tmp_data))
        

                
                
        
        
    
    
    def to_tensor(self,source_label):
        
        
        source_label
        tensor_dict = {} 
        
        batch_input_ids = []
        batch_input_mask = []
        batch_label = [] 
        batch_insertion_index = []
        
        raw_data = self.raw_data_dict[str(source_label)]
        
        for idx in range(len(raw_data)):
            # print(raw_data[idx])
            # print(self.tokenizer)
            # print(self.tokenizer('aaaaaaaaad'))
            # exit() 
            
            tokenized_data_item = self.tokenizer(raw_data[idx],max_length=self.tokenized_max_len,padding='max_length',truncation=True,return_tensors='pt')

            
            input_ids = tokenized_data_item['input_ids']
            
            if input_ids.shape[1] < self.tokenized_max_len:
                padding_tensor = torch.zeros(1,self.tokenized_max_len-input_ids.shape[1])
                
                input_ids = torch.cat((input_ids,padding_tensor),1)
            
            if self.placeholder_ids in input_ids: 
                input_mask = tokenized_data_item['attention_mask']
                
                output_dict = self.transformer_model(input_ids=input_ids.to(self.device).long(),attention_mask=input_mask.to(self.device))[0]

                
                
                
                if self.cls_token_is_first: 
                    output_embedding = output_dict[:,0,:].unsqueeze(1)
                
                else: 
                    output_embedding = output_dict[:,-1,:].unsqueeze(1)
                
                target_logits = self.target_model(output_embedding)
                benign_logits = self.benign_model(output_embedding)
                
                target_pred = torch.argmax(target_logits,1)
                benign_pred = torch.argmax(benign_logits,1)



                
                if benign_pred == source_label and len(batch_label) < self.dataset_len:

                
                    
                    insert_index = np.where(np.array(tokenized_data_item['input_ids'][0]) == self.placeholder_ids)[0][0]
                    
                    if len(batch_label) == 0: 
                        batch_input_ids = input_ids 
                        batch_input_mask = input_mask


                    else:
                        batch_input_ids = torch.cat((batch_input_ids,input_ids),0)
                        batch_input_mask = torch.cat((batch_input_mask,input_mask),0)

                    batch_label.append(source_label)
                    batch_insertion_index.append(insert_index)
                    
                
                
                if len(batch_label) == self.dataset_len: 
                    
                    tensor_dict['input_ids'] = torch.as_tensor(batch_input_ids).to(self.device)
                    tensor_dict['attention_mask'] = torch.as_tensor(batch_input_mask).to(self.device)
                    tensor_dict['insertion_index'] = torch.as_tensor(batch_insertion_index).to(self.device)
                    tensor_dict['label'] = torch.as_tensor(batch_label).to(self.device)
                    
        
                    return tensor_dict
    
    def gen_dataset(self,source_label):
        self.load_data()
        self.insert_placeholder()
        
        return self.to_tensor(source_label)
         


if __name__ == '__main__':
    

    tokenizer = torch.load('/data/share/trojai/trojai-round6-v2-dataset/tokenizers/DistilBERT-distilbert-base-uncased.pt')

    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    # exit() 
    
    placeholder_token = ' *'

    
    opt_len = 5 
    examples_dirpath = '/data/share/trojai/trojai-round6-v2-dataset/models/id-00000006/clean_example_data'
    seed = 0 
    max_len = 300
    insert_type = 'random'
    dataset = Distilbert_Dataset(tokenizer,placeholder_token,opt_len,examples_dirpath,max_len,insert_type,seed)
    dataset.gen_dataset(0)
    