import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import math
from collections import OrderedDict
from transformers import GPT2LMHeadModel
from happytransformer import HappyWordPrediction
from difflib import SequenceMatcher
import jellyfish

def find_intersection(t1, t2):
    indices = torch.zeros_like(t1, dtype = torch.bool, device=t1.device)
    for elem in t2:
        indices = indices | (t1 == elem)
        intersection = t1[indices]
    return intersection

class AttrChanger:
    def __init__(self,embedding_backbone,target_model,tokenizer,model_arch,device,logger,config,triggers):
        self.embedding_backbone = embedding_backbone
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.device = device
        self.model_arch = model_arch
        self.max_len = config['max_len']
        self.triggers=triggers
        self.logger = logger
        #self.UNK=self.tokenizer(self.tokenizer.unk_token)['input_ids'][0]
        self.UNK=self.tokenizer('@')['input_ids'][0]
        self.PAD=self.tokenizer(self.tokenizer.pad_token)['input_ids'][0]
        #self.PAD=self.tokenizer('PAD')['input_ids'][0]
        self.EOS=self.tokenizer(self.tokenizer.eos_token)['input_ids'][0]
        #print(f'id for UNK {self.UNK}, id for PAD {self.PAD}, id for EOS {self.EOS}')
        self.NOT=self.tokenizer('NOT ')['input_ids']
        #print(f'id for NOT {self.NOT}')

        self.pred_model=GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=self.tokenizer.eos_token_id).to(self.device)
        self.pred_model.config.pad_token_id = self.pred_model.config.eos_token_id

    def decode_token(self, token):
        return self.tokenizer.decoder.get(token.item())

    def find_common_tokens(self, impt_tokens: torch.Tensor):
        commons=OrderedDict()
        half=math.ceil(len(impt_tokens)/2)
        for sidx in range(half):
            starter=impt_tokens[sidx]
            for token in starter:
                #print(self.decode_token(token), end=' ')
                commons[token.item()]=0
            leftover=torch.vstack([impt_tokens[:sidx], impt_tokens[sidx+1:]])
            for ii, sample in enumerate(leftover):
                for token in sample:
                    #print(self.decode_token(token), end=' ')
                    if token.item() in commons:
                        commons[token.item()]+=1
                #print('\n')
            most_common=sorted(commons, key=commons.get, reverse=True)
            if commons[most_common[2]] >= half:
                break

        #print(most_common)
        #for token in most_common:
        #    print(commons[token], end=' ')
        #print('\n')

        return torch.LongTensor([t for t in commons if commons[t]>=half//2]).unsqueeze(0)


    def pre_processing(self, sample):
        tokenized_dict = self.tokenizer(sample, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = tokenized_dict['input_ids'].to(self.device)
        attention_mask = tokenized_dict['attention_mask'].to(self.device)
        return input_ids, attention_mask

    def check_similarity(self,pred,label, token=False):
        if not token:
            token=False
            sim_score=jellyfish.levenshtein_distance(pred, label)
            return sim_score


    def change_one_token(self, attn_idx):
        #input_ids=self.orig_inputs
        input_ids=self.pick_input
        attn_idx=torch.stack([torch.arange(input_ids.shape[1], dtype=torch.int64)]*len(input_ids))
        width=input_ids.shape[1]
        keep_idx=[[]]*len(input_ids)
        clique=[np.array([])]*len(input_ids)
        for ti in range(attn_idx.shape[1]):
            ###new_input=input_ids.clone()
            new_input=[]
            for si in range(input_ids.shape[0]):
                ###new_input[si][attn_idx[si][ti]]=self.UNK
                isrt_idx=attn_idx[si][ti]
                new_input.append(torch.cat([input_ids[si][:isrt_idx], torch.LongTensor(self.NOT).to(self.device), input_ids[si][isrt_idx:-2],torch.LongTensor([self.EOS]).to(self.device)] )[:width])
            #print([len(new_input[ii]) for ii in range(len(new_input))])
            new_input=torch.stack(new_input)
            #print('insert finished',new_input.shape, input_ids.shape)
            #str_to_check=self.tokenizer.batch_decode(new_input, skip_special_tokens=True)
            #print(str_to_check)
            new_emb=self.embedding_backbone(new_input)[0]
            if self.model_arch == 'distilbert':
                new_emb = new_emb[:,0,:].unsqueeze(1)
            else:
                new_emb=new_emb[:,-1,:].unsqueeze(1)
            logits=self.target_model(new_emb)
            celoss=torch.sum(torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.softmax(self.orig_logits, dim=1), reduction='none'), dim=1)
            thres=torch.median(celoss)
            keep_token=torch.where(celoss<thres, 0, 1)
            #print(f'{len(keep_token)} / {len(input_ids)} sentences keep this token')
            #print(torch.nonzero(keep_token, as_tuple=True))
            for ki in torch.nonzero(keep_token, as_tuple=True)[0]:
                keep_idx[ki].append(attn_idx[ki][ti])
                actual_id=input_ids[ki][attn_idx[ki][ti]].item()
                #print(ki, actual_id)
                if actual_id==self.UNK:
                    actual_id=self.PAD
                clique[ki]=np.append(clique[ki], actual_id)

        effect_lens=[]
        for si in range(len(input_ids)):
            pad_len=input_ids.shape[1]-len(clique[si])
            effect_lens.append(len(clique[si]))
            clique[si]=np.append(clique[si], [self.PAD]*pad_len)
        clique=np.array(clique, dtype=np.int64)
        clique=torch.LongTensor(clique)
        #print(input_ids, input_ids[attn_idx])
        #attn_input=input_ids[attn_idx]
        #str_to_check=self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        #print(str_to_check[:5])
        #str_to_check=self.tokenizer.batch_decode(self.attn_input, skip_special_tokens=True)
        #print(str_to_check[:5])
        return keep_idx, clique, effect_lens

    def forward(self, input_ids, attention_mask, output_hidden_states=False, output_attentions=False):
        if self.model_arch == 'gpt2':
            for spl_idx in range(len(input_ids)):
                sample = input_ids[spl_idx]
                if sample[-1] == self.tokenizer.pad_token_id:
                    last_valid_token_idx = (sample == self.tokenizer.pad_token_id).nonzero()[0] - 1
                    last_valid_token = sample[last_valid_token_idx]
                    input_ids[spl_idx][-1] = last_valid_token
                    attention_mask[spl_idx][-1] = 1
        res = self.embedding_backbone(input_ids, attention_mask=attention_mask, output_hidden_states=output_hidden_states,output_attentions=output_attentions)
        if output_hidden_states:
            attn = res[3][-1]
        else:
            attn = None
        emb=res[0]
        if self.model_arch == 'distilbert':
            emb = emb[:,0,:].unsqueeze(1)
        else:
            emb = emb[:,-1,:].unsqueeze(1)
        logits = self.target_model(emb)
        preds = torch.argmax(logits, dim=1)
        return input_ids, attention_mask, emb, attn, logits, preds

    def check_one_token(self, input_ids, pred_idx, correct_token):
        str_to_check=self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        happy_wp = HappyWordPrediction()
        new_str_to_check=[]
        correct=0
        for si, str_i in enumerate(str_to_check):
            #tmp_str=str_i.split()
            #print(tmp_str)
            #unk_idx=tmp_str.index('@')
            #new_str=' '.join(tmp_str[:unk_idx]+['[MASK]']+tmp_str[unk_idx+1:])
            new_str=str_i.replace('@', ' [MASK] ')[:512]
            new_str_to_check.append(new_str)
            print(str_i, '\n', new_str)
            result = happy_wp.predict_mask(new_str, top_k = 30)
            for ci, choice in enumerate(result):
                #print('candidate', ci, choice.token, choice.score)
                #sim_check=SequenceMatcher(None, self.tokenizer.decoder.get(correct_token[si].item()), choice.token).ratio()
                sim_check=jellyfish.levenshtein_distance(choice.token, self.tokenizer.decoder.get(correct_token[si].item()))
                if self.tokenizer(choice.token)['input_ids'][0]==correct_token[si].item() or sim_check<2:
                    print('candidate', ci, choice.token, choice.score)
                    correct+=1
                    break
        print(correct)
        return correct


    def check_clique(self, clique, effect_lens=None, keep_idx=None, num_keep=10):
        #mask and predict
        interp_idx=[]
        #clique_str=self.tokenizer.batch_decode(clique, skip_special_tokens=True)
        #print('orig clique:')
        #for i, ci in enumerate(clique_str):
        #    print(i, ci)
        correct=0
        for i in range(clique.shape[1]):
            test_clique=clique.clone()
            test_clique[:, i]=self.UNK
            correct+=self.check_one_token(test_clique, i, clique[:, i])
        print('total correct predictions:', correct, clique.shape[0]*clique.shape[1])


    def sort_by_attn(self,sample):
        #TODO: this may be not a nn.module
        # transform raw text input to tokens
        self.NUM_PICK=20
        input_ids, attention_mask = self.pre_processing(sample)
        input_ids, attention_mask, emb, attn, logits, preds = self.forward(input_ids, attention_mask, True, True)
        self.orig_logits=logits
        self.orig_inputs=input_ids
        self.orig_preds=preds
        print(self.orig_preds)
        self.attn_mask=attention_mask
        #print(attn.shape) #(bz, num_heads, seq_len, seq_len)
        embedding_score=attn[:, :, -1, :]
        #confirm whether softmax dim = 1
        #test=attn[0,0,:,:]
        #sum_test=torch.where(torch.abs(torch.sum(test, dim=1)-1)<0.001, 1, 0)
        #print(torch.sum(sum_test))
        sum_score=torch.sum(embedding_score, dim=1)
        print(embedding_score.shape, sum_score.shape)
        sort_idx=torch.topk(sum_score, 100, dim=1)[1] #pick idx
        pick_idx=sort_idx[:, :self.NUM_PICK]
        print(pick_idx.shape)
        pick_idx, _=torch.sort(pick_idx, dim=1)
        self.pick_input=[]
        self.attn_input=input_ids.clone()
        for i in range(len(input_ids)):
            self.attn_input[i]=input_ids[i][sort_idx[i]]
            self.pick_input.append(input_ids[i][pick_idx[i]])
        self.pick_input=torch.stack(self.pick_input)
        self.check_clique(self.pick_input)
        #important=self.tokenizer.batch_decode(self.pick_input, skip_special_tokens=True)
        #common_tokens=self.find_common_tokens(self.pick_input)
        #print(common_tokens)
        #print(self.tokenizer.batch_decode(common_tokens, skip_special_tokens=True))
        ###print('10 important tokens:')
        #for i, ci in enumerate(important):
        #    print(i, ci)
        # print(self.pick_input)

        return sort_idx

    def triple_check(self, original, rephrased):
        orig_ids, orig_mask = self.pre_processing(original)
        reph_ids, reph_mask = self.pre_processing(rephrased)
        orig_ids, orig_mask, orig_emb, orig_attn, orig_logits, orig_preds = self.forward(orig_ids, orig_mask)
        print(orig_logits, '\n', orig_preds)
        reph_ids, reph_mask, reph_emb, reph_attn, reph_logits, reph_preds = self.forward(reph_ids, reph_mask)
        print(reph_logits, '\n', reph_preds)
