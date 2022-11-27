import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers import GPT2LMHeadModel
from happytransformer import HappyWordPrediction

class AttrChanger:
    def __init__(self,embedding_backbone,target_model,tokenizer,model_arch,device,logger,config):
        self.embedding_backbone = embedding_backbone
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.device = device
        self.model_arch = model_arch
        self.logger = logger
        #self.UNK=self.tokenizer(self.tokenizer.unk_token)['input_ids'][0]
        self.UNK=self.tokenizer('@')['input_ids'][0]
        self.PAD=self.tokenizer(self.tokenizer.pad_token)['input_ids'][0]
        #self.PAD=self.tokenizer('PAD')['input_ids'][0]
        self.EOS=self.tokenizer(self.tokenizer.eos_token)['input_ids'][0]
        print(f'id for UNK {self.UNK}, id for PAD {self.PAD}, id for EOS {self.EOS}')
        self.NOT=self.tokenizer('NOT')['input_ids'][0]
        print(f'id for NOT {self.NOT}')

        self.pred_model=GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=self.tokenizer.eos_token_id).to(self.device)
        self.pred_model.config.pad_token_id = self.pred_model.config.eos_token_id

        self.temp = config['init_temp']
        self.max_temp = config['max_temp']
        self.temp_scaling_check_epoch = config['temp_scaling_check_epoch']
        self.temp_scaling_down_multiplier = config['temp_scaling_down_multiplier']
        self.temp_scaling_up_multiplier = config['temp_scaling_up_multiplier']
        self.loss_barrier = config['loss_barrier']
        self.noise_ratio = config['noise_ratio']
        self.rollback_thres = config['rollback_thres']

        self.epochs = config['epochs']
        self.lr = config['lr']
        self.scheduler_step_size = config['scheduler_step_size']
        self.scheduler_gamma = config['scheduler_gamma']

        self.max_len = config['max_len']
        self.trigger_len = config['trigger_len']
        self.eps_to_one_hot = config['eps_to_one_hot']

        self.start_temp_scaling = False
        self.rollback_num = 0
        self.best_asr = 0
        self.best_loss = 1e+10
        self.best_trigger = 'TROJAI_GREAT'

        self.placeholder_ids = self.tokenizer.pad_token_id
        self.placeholders = torch.ones(self.trigger_len).to(self.device).long() * self.placeholder_ids
        self.placeholders_attention_mask = torch.ones_like(self.placeholders)
        self.word_embedding = self.embedding_backbone.get_input_embeddings().weight






    def pre_processing(self,sample):

        tokenized_dict = self.tokenizer(
            sample, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = tokenized_dict['input_ids'].to(self.device)
        attention_mask = tokenized_dict['attention_mask'].to(self.device)

        return input_ids, attention_mask

    def stamping_placeholder(self, raw_input_ids, raw_attention_mask,insert_idx, insert_content=None):
        stamped_input_ids = raw_input_ids.clone()
        stamped_attention_mask = raw_attention_mask.clone()

        insertion_index = torch.zeros(
            raw_attention_mask.shape[0]).long().to(self.device)

        if insert_content != None:
            content_attention_mask = torch.ones_like(insert_content)

        for idx, each_attention_mask in enumerate(raw_attention_mask):

            if insert_content == None:


                if self.model_arch == 'distilbert':

                    tmp_input_ids = torch.cat(
                        (raw_input_ids[idx, :insert_idx], self.placeholders, raw_input_ids[idx, insert_idx:]), 0)[:self.max_len]
                    tmp_attention_mask = torch.cat(
                        (raw_attention_mask[idx, :insert_idx], self.placeholders_attention_mask, raw_attention_mask[idx, insert_idx:]), 0)[:self.max_len]

                elif self.model_arch == 'gpt2':





                    tmp_input_ids = torch.cat(
                        (raw_input_ids[idx, :insert_idx], self.placeholders, raw_input_ids[idx, insert_idx:]), 0)[:self.max_len]
                    tmp_attention_mask = torch.cat(
                        (raw_attention_mask[idx, :insert_idx], self.placeholders_attention_mask, raw_attention_mask[idx, insert_idx:]), 0)[:self.max_len]

                    if tmp_input_ids[-1] == self.tokenizer.pad_token_id:
                        last_valid_token_idx = (raw_input_ids[idx] == self.tokenizer.pad_token_id).nonzero()[0] - 1
                        last_valid_token = raw_input_ids[idx,last_valid_token_idx]


                        tmp_input_ids[-1] = last_valid_token
                        tmp_attention_mask[-1] = 1


                    # print(tmp_attention_mask)
                    # exit()


                    # last_valid_token_idx = (np.where(np.array(tmp_input_ids) == self.tokenizer.pad_token_id)[0][0]) - 1
                    # last_valid_token = input_ids[0,last_valid_token_idx]
                    # input_ids[0,-1] = last_valid_token
                    # input_mask[0,-1] = 1


            else:

                tmp_input_ids = torch.cat(
                    (raw_input_ids[idx, :insert_idx], insert_content, raw_input_ids[idx, insert_idx:]), 0)[:self.max_len]
                tmp_attention_mask = torch.cat(
                    (raw_attention_mask[idx, :insert_idx], content_attention_mask, raw_attention_mask[idx, insert_idx:]), 0)[:self.max_len]

            stamped_input_ids[idx] = tmp_input_ids
            stamped_attention_mask[idx] = tmp_attention_mask
            insertion_index[idx] = insert_idx

        return stamped_input_ids, stamped_attention_mask,insertion_index

    def forward(self,epoch,stamped_input_ids,stamped_attention_mask,insertion_index):



        if self.model_arch == 'distilbert':
            position_ids = torch.arange(
                self.max_len, dtype=torch.long).to(self.device)
            position_ids = position_ids.unsqueeze(
                0).expand([stamped_input_ids.shape[0], self.max_len])
            self.position_embedding = self.embedding_backbone.embeddings.position_embeddings(
                position_ids)


        self.optimizer.zero_grad()
        self.embedding_backbone.zero_grad()
        self.target_model.zero_grad()

        noise = torch.zeros_like(self.opt_var).to(self.device)

        if self.rollback_num >= self.rollback_thres:
            # print('decrease asr threshold')
            self.rollback_num = 0
            self.loss_barrier = min(self.loss_barrier*2,self.best_loss - 1e-3)


        if (epoch) % self.temp_scaling_check_epoch == 0:
            if self.start_temp_scaling:
                if self.ce_loss < self.loss_barrier:
                    self.temp /= self.temp_scaling_down_multiplier

                else:
                    self.rollback_num += 1
                    noise = torch.rand_like(self.opt_var).to(self.device) * self.noise_ratio
                    self.temp *= self.temp_scaling_down_multiplier
                    if self.temp > self.max_temp:
                        self.temp = self.max_temp

        self.bound_opt_var = torch.softmax(self.opt_var/self.temp + noise,1)



        trigger_word_embedding = torch.tensordot(self.bound_opt_var,self.word_embedding,([1],[0]))

        sentence_embedding = self.embedding_backbone.get_input_embeddings()(stamped_input_ids)

        for idx in range(stamped_input_ids.shape[0]):

            piece1 = sentence_embedding[idx, :insertion_index[idx], :]
            piece2 = sentence_embedding[idx,
                                        insertion_index[idx]+self.trigger_len:, :]

            sentence_embedding[idx] = torch.cat(
                (piece1, trigger_word_embedding.squeeze(), piece2), 0)

        if self.model_arch == 'distilbert':
            norm_sentence_embedding = sentence_embedding + self.position_embedding
            norm_sentence_embedding = self.embedding_backbone.embeddings.LayerNorm(
                norm_sentence_embedding)
            norm_sentence_embedding = self.embedding_backbone.embeddings.dropout(
                norm_sentence_embedding)

            output_dict = self.embedding_backbone(
                            inputs_embeds=norm_sentence_embedding, attention_mask=stamped_attention_mask)[0]

            output_embedding = output_dict[:,0,:].unsqueeze(1)


        else:
            output_dict = self.embedding_backbone(
                inputs_embeds=sentence_embedding, attention_mask=stamped_attention_mask)[0]

            output_embedding = output_dict[:,-1,:].unsqueeze(1)

        logits = self.target_model(output_embedding)


        return logits


    def compute_loss(self, logits, labels):

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return loss



    def compute_acc(self, logits, labels):
        predicted_labels = torch.argmax(logits, dim=1)
        correct = (predicted_labels == labels).sum()
        acc = correct / predicted_labels.shape[0]
        return acc


    def dim_check(self):

        # extract largest dimension at each position
        values, dims = torch.topk(self.bound_opt_var, 1, 1)

        # idx = 0
        # dims = topk_dims[:, idx]
        # values = topk_values[:, idx]

        # calculate the difference between current inversion to one-hot
        diff = self.bound_opt_var.shape[0] - torch.sum(values)

        # check if current inversion is close to discrete and loss smaller than the bound
        if diff < self.eps_to_one_hot and self.ce_loss <= self.loss_barrier:

            # update best results

            tmp_trigger = ''
            tmp_trigger_ids = torch.zeros_like(self.placeholders)
            for idy in range(values.shape[0]):
                tmp_trigger = tmp_trigger + ' ' + \
                    self.tokenizer.convert_ids_to_tokens([dims[idy]])[0]
                tmp_trigger_ids[idy] = dims[idy]

            self.best_asr = self.asr
            self.best_loss = self.ce_loss
            self.best_trigger = tmp_trigger
            self.best_trigger_ids = tmp_trigger_ids

            # reduce loss bound to generate trigger with smaller loss
            self.loss_barrier = self.best_loss / 2
            self.rollback_num = 0

    def change_one_token(self, attn_idx):
        input_ids=self.orig_inputs
        width=input_ids.shape[1]
        keep_idx=[[]]*len(input_ids)
        clique=[np.array([])]*len(input_ids)
        for ti in range(attn_idx.shape[1]):
            #new_input=input_ids.clone()
            new_input=[]
            for si in range(input_ids.shape[0]):
                #new_input[si][attn_idx[si][ti]]=self.UNK
                isrt_idx=attn_idx[si][ti]
                new_input.append(torch.cat([input_ids[si][:isrt_idx], torch.LongTensor([self.NOT]).to(self.device), input_ids[si][isrt_idx:-2],torch.LongTensor([self.EOS]).to(self.device)] )[:width])
            #print([len(new_input[ii]) for ii in range(len(new_input))])
            new_input=torch.stack(new_input)
            #print('insert finished',new_input.shape, input_ids.shape)
            new_emb=self.embedding_backbone(new_input)[0]
            if self.model_arch == 'distilbert':
                new_emb = new_emb[:,0,:].unsqueeze(1)
            else:
                new_emb=new_emb[:,-1,:].unsqueeze(1)
            logits=self.target_model(new_emb)
            celoss=torch.sum(torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.softmax(self.orig_logits, dim=1), reduction='none'), dim=1)
            thres=(celoss.max()+celoss.min())/20
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


    def check_one_token(self, input_ids, pred_idx):
        str_to_check=self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        happy_wp = HappyWordPrediction()
        new_str_to_check=[]
        for str_i in str_to_check[19:]:
            #tmp_str=str_i.split()
            #print(tmp_str)
            #unk_idx=tmp_str.index('@')
            #new_str=' '.join(tmp_str[:unk_idx]+['[MASK]']+tmp_str[unk_idx+1:])
            new_str=str_i.replace('@', ' [MASK] ')[:512]
            new_str_to_check.append(new_str)
            print(new_str)
            result = happy_wp.predict_mask(new_str, top_k = 5)
            print(new_str, result)

        #print(str_to_check)
        #outputs = self.pred_model.generate(input_ids=input_ids,attention_mask=attn_mask)



    def check_clique(self, clique, effect_lens, keep_idx, num_keep=10):
        #mask and predict
        interp_idx=[]
        clique_str=self.tokenizer.batch_decode(clique, skip_special_tokens=True)
        print('orig clique:\n', clique_str[-1])
        for i in range(clique.shape[1]):
            test_clique=clique.clone()
            test_clique[:, i]=self.UNK
            self.check_one_token(test_clique, i)
            break



    def sort_by_attn(self,sample):
        #TODO: this may be not a nn.module
        # transform raw text input to tokens
        input_ids, attention_mask = self.pre_processing(sample)
        print(input_ids.shape, attention_mask.shape)
        res=self.embedding_backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True,output_attentions=True)
        attn=res[3][-1]
        emb=res[0]
        if self.model_arch == 'distilbert':
            emb = emb[:,0,:].unsqueeze(1)
        else:
            emb=emb[:,-1,:].unsqueeze(1)
        logits=self.target_model(emb)
        self.orig_logits=logits
        self.orig_inputs=input_ids
        self.orig_preds=torch.argmax(logits, dim=1)
        print(self.orig_preds)
        self.attn_mask=attention_mask
        print(attn.shape) #(bz, num_heads, seq_len, seq_len)
        embedding_score=attn[:, :, -1, :]
        #confirm whether softmax dim = 1
        #test=attn[0,0,:,:]
        #sum_test=torch.where(torch.abs(torch.sum(test, dim=1)-1)<0.001, 1, 0)
        #print(torch.sum(sum_test))
        sum_score=torch.sum(embedding_score, dim=1)
        print(embedding_score.shape, sum_score.shape)
        sort_idx=torch.topk(sum_score, 100, dim=1)[1] #pick idx
        self.attn_input=input_ids.clone()
        for i in range(len(input_ids)):
            self.attn_input[i]=input_ids[i][sort_idx[i]]

        return sort_idx
        '''
        for epoch in range(self.epochs):

            # feed forward
            logits,benign_logits = self.forward(epoch,stamped_input_ids,stamped_attention_mask,insertion_index)


            # compute loss
            target_labels = torch.ones_like(logits[:, 0]).long().to(
                self.device) * target_label
            ce_loss,benign_ce_loss = self.compute_loss(logits,benign_logits,target_labels)
            asr = self.compute_acc(logits,target_labels)

            # marginal benign loss penalty
            if epoch == 0:
                # if benign_asr > 0.75:
                benign_loss_bound = benign_ce_loss.detach()
                # else:
                #     benign_loss_bound = 0.2

            benign_ce_loss = max(benign_ce_loss - benign_loss_bound, 0)

            loss = ce_loss +  benign_ce_loss

            if self.model_arch == 'distilbert':
                loss.backward(retain_graph=True)

            else:
                loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            self.ce_loss = ce_loss
            self.asr = asr

            if ce_loss <= self.loss_barrier:
                self.start_temp_scaling = True


            self.dim_check()

            self.logger.trigger_generation('Epoch: {}/{}  Loss: {:.4f}  ASR: {:.4f}  Best Trigger: {}  Best Trigger Loss: {:.4f}  Best Trigger ASR: {:.4f}'.format(epoch,self.epochs,self.ce_loss,self.asr,self.best_trigger,self.best_loss,self.best_asr))
        '''

        return self.best_trigger, self.best_loss
