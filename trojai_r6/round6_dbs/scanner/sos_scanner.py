import torch 
from torch.nn import CrossEntropyLoss


class SOS_Scanner:
    def __init__(self,target_model,benign_model, tokenizer,tensor_dict,key_words,user_config):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.target_model = target_model 
        self.benign_model = benign_model
        self.tokenizer = tokenizer 
        self.tensor_dict = tensor_dict 
        self.placeholder_ids = user_config['trigger_placeholder']
        self.placeholder_len = user_config['trigger_len']
        self.key_words = key_words 
        self.temp = user_config['init_temp']
        self.rollback_num = 0 
        self.rollback_thres = user_config['rollback_thres']
        self.temp_rollback_thres = user_config['temp_rollback_thres']
        self.temp_rollback_num = 0
        self.lr = user_config['lr']
        self.epochs = user_config['epochs']
        self.trigger_len = user_config['trigger_len']
        self.dataset_len = self.tensor_dict['input_ids'].shape[0]
        self.batch_size = self.dataset_len
        self.word_embedding = self.target_model.bert.get_input_embeddings().weight
        self.loss_barrier = user_config['loss_barrier']
        self.scheduler_step_size = user_config['scheduler_step_size']
        self.scheduler_gamma = user_config['scheduler_gamma']
        self.early_stop = user_config['early_stop']
        
        self.target_ce_loss = 0
        self.num = 0 
        self.best_loss = 1e+10 
        self.best_trigger = 'TROJAI_GREAT'

        # position_ids = torch.arange(200,dtype=torch.long).to(self.device)
        # position_ids = position_ids.unsqueeze(0).expand([self.batch_size,200])
        # self.position_embedding = self.transformer_model.embeddings.position_embeddings(position_ids)
        
        self.target_model.eval()
        self.benign_model.eval()

    def forward(self,batch_idx,victim_id,target_id,epoch):
        print('forward')
        
        self.optimizer.zero_grad()
        self.target_model.zero_grad()


        
        input_ids = self.tensor_dict['input_ids'][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].to(self.device).long()
        attention_mask = self.tensor_dict['attention_mask'][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].to(self.device)
        label = self.tensor_dict['label'][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].to(self.device)
        insertion_index = self.tensor_dict['insertion_index'][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].to(self.device)

        noise = torch.zeros_like(self.opt_var).to(self.device)
        
        if (epoch) % 5 == 0 and batch_idx == 0:
            if self.num >= 1:
                if self.target_ce_loss < self.loss_barrier:
                    self.temp /= 2
                    
                else:
                    self.rollback_num += 1 
                    noise = torch.rand_like(self.opt_var).to(self.device) * 10 
                    self.temp *= 5
                    if self.temp > 2:
                        self.temp = 2 
        

        
        
        self.bound_opt_var = torch.softmax(self.opt_var/self.temp + noise,1)

        # 1045, 2031, 4149, 2009, 2013, 1037, 3573, 2007, 2026, 2814, 2197, 5353
        # self.bound_opt_var[:,:] = 0
        # self.bound_opt_var[0,2814] = 1 
        # self.bound_opt_var[1,5353] = 1
        # self.bound_opt_var[2,3573] = 1 
        # self.bound_opt_var[3,2009] = 1 
        # self.bound_opt_var[4,2013] = 1 
        # self.bound_opt_var[5,1037] = 1 
        # self.bound_opt_var[6,3573] = 1 
        # self.bound_opt_var[7,2007] = 1 
        # self.bound_opt_var[8,2026] = 1 
        # self.bound_opt_var[9,2814] = 1 
        # self.bound_opt_var[10,2197] = 1 
        # self.bound_opt_var[11,5353] = 1 
        

        
        
        trigger_word_embedding = torch.tensordot(self.bound_opt_var,self.word_embedding,([1],[0]))
        
        sentence_embedding = self.target_model.bert.get_input_embeddings()(input_ids)



        
        
        # insert opt var into placeholder position 
        
        for idx in range(input_ids.shape[0]):
            
            piece1 = sentence_embedding[idx,:insertion_index[idx],:]
            
            piece2 = sentence_embedding[idx,insertion_index[idx]+self.placeholder_len:,:]
            sentence_embedding[idx] = torch.cat((piece1,trigger_word_embedding,piece2),0)
            
        
        # norm_sentence_embedding = sentence_embedding + self.position_embedding
        # norm_sentence_embedding = self.transformer_model.embeddings.LayerNorm(norm_sentence_embedding)
        # norm_sentence_embedding = self.transformer_model.embeddings.dropout(norm_sentence_embedding)
        
        logits = self.target_model(inputs_embeds=sentence_embedding,
                                     attention_mask=attention_mask).logits

        
        # print(output/)
        # cls_tokens = output_dict[0][:, 0, :]   # batch_size, 768
        # logits = self.target_model.classifier(cls_tokens) # batch_size, 1(4)
        

        benign_logits = self.benign_model(inputs_embeds=sentence_embedding,
                                     attention_mask=attention_mask).logits
        
        # benign_cls_tokens = benign_output_dict[0][:, 0, :]   # batch_size, 768
        # benign_logits = self.benign_model.classifier(benign_cls_tokens) # batch_size, 1(4)
        
        
        
        # if self.cls_token_is_first:
        #     output_embedding = output_dict[:,0,:].unsqueeze(1)
        
        # else:  
        #     output_embedding = output_dict[:,-1,:].unsqueeze(1)
        
        # logits = self.target_model(output_embedding)
        
        # benign_logits = self.benign_model(output_embedding)
        
        # print(logits)
        # print(benign_logits)
        

        
                    
        target_label = torch.ones_like(logits[:,0]).long().to(self.device) * target_id
        
        
        return logits,benign_logits,target_label 
    

    def compute_loss(self,logits,benign_logits,label):
        print('loss computation')
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits,label)
        
        benign_loss = loss_fct(benign_logits,1-label)
        return loss,benign_loss
    
    def acc_log(self,logits,label):
        predicted_label = torch.argmax(logits,dim=1)
        
        correct = (predicted_label == label).sum() 
        
        acc = correct / predicted_label.shape[0]
        
        return acc 
        

    def important_dim_log(self):
        
        topk_values,topk_dims = torch.topk(self.bound_opt_var,20,1)
        
        print('='*80)
        # print('{:<15}   {:<25}  {:<25}  {:<25}  {:<25} {:<25} {:<25}  {:<25}  {:<25}  {:<25} {:<25}'.format('top ranking',1,2,3,4,5,6,7,8,9,10))
        # print('{:<15}   {:<25}  {:<25}  {:<25}  {:<25} {:<25}'.format('top ranking',1,2,3,4,5))
        # print('{:<15}   {:<25}  {:<25}'.format('top ranking',1,2))

        lst = list(range(1,self.placeholder_len+1))
        strFormat = len(lst) * '{:<25} '
        formattedList = strFormat.format(*lst)
        print(formattedList)
        
        for idx in range(20):
            dims = topk_dims[:,idx]
            values = topk_values[:,idx]
            diff = self.placeholder_len - torch.sum(values)
            # if idx == 0 and diff < 0.001:
            if idx == 0 and diff == 0:
                if self.target_ce_loss < self.best_loss:
                    print('update!!!!!')
                    self.best_loss = self.target_ce_loss

                    self.loss_barrier = self.best_loss / 2
                    self.rollback_num = 0 
                    
                    
                    tmp_trigger = ''
                    for idy in range(values.shape[0]):
                        tmp_trigger = tmp_trigger + ' ' + self.tokenizer.convert_ids_to_tokens([dims[idy]])[0]
                    
                    self.best_trigger = tmp_trigger

                    print(self.best_trigger)
                    print(self.best_loss)
                

            # print('{:<15}   {:<15}({:.5f})   {:<15}({:.5f})  {:<15}({:.5f}) {:<15}({:.5f}) {:<15}({:.5f}) {:<15}({:.5f})   {:<15}({:.5f})  {:<15}({:.5f}) {:<15}({:.5f}) {:<15}({:.5f})'.format(idx,self.tokenizer.decode(dims[0]),values[0],self.tokenizer.decode(dims[1]),values[1],self.tokenizer.decode(dims[2]),values[2],self.tokenizer.decode(dims[3]),values[3],self.tokenizer.decode(dims[4]),values[4],self.tokenizer.decode(dims[5]),values[5],self.tokenizer.decode(dims[6]),values[6],self.tokenizer.decode(dims[7]),values[7],self.tokenizer.decode(dims[8]),values[8],self.tokenizer.decode(dims[9]),values[9]))
            # print('{:<15}   {:<15}({:.5f})   {:<15}({:.5f})  {:<15}({:.5f}) {:<15}({:.5f}) {:<15}({:.5f})'.format(idx,self.tokenizer.decode(dims[0]),values[0],self.tokenizer.decode(dims[1]),values[1],self.tokenizer.decode(dims[2]),values[2],self.tokenizer.decode(dims[3]),values[3],self.tokenizer.decode(dims[4]),values[4]))
            # print('{:<15}   {:<15}({:.5f})   {:<15}({:.5f})'.format(idx,self.tokenizer.decode(dims[0]),values[0],self.tokenizer.decode(dims[1]),values[1]))

            log_list = [idx]
            for idz in range(self.placeholder_len):
                
                tokens = self.tokenizer.convert_ids_to_tokens([dims[idz]])[0]
                log_list.append(tokens)

                
                log_list.append(values[idz])


            

                            
            logFormat = '{:<15}' + self.placeholder_len * '{:<15}({:.5f})'

            formattedlog = logFormat.format(*log_list)
            print(formattedlog)   
            
        if self.key_words is not None:
            print('-'*80)
            key_dimensions = self.tokenizer.encode(self.key_words)[1:-1]

            values,ranking = torch.sort(self.bound_opt_var,1,descending=True)  
            # print('{:<15}   {:<15}  {:<15}  {:<15}  {:<15}  {:<15} {:<15}  {:<15}  {:<15}  {:<15}  {:<15}'.format('trojan word',1,2,3,4,5,6,7,8,9,10))
            print(formattedList)
            # print('{:<15}   {:<15}  {:<15}  {:<15}  {:<15}  {:<15}'.format('trojan word',1,2,3,4,5))
            # print('{:<15}   {:<15}  {:<15}'.format('trojan word',1,2))
            # for key_dim in key_dimensions:
                # rank = (ranking == key_dim).nonzero()[:,1]

                # print('{:<15}   {:<15}({:.5f})  {:<15}({:.5f})  {:<15}({:.5f}) {:<15}({:.5f}) {:<15}({:.5f}) {:<15}({:.5f})  {:<15}({:.5f})  {:<15}({:.5f}) {:<15}({:.5f}) {:<15}({:.5f})'.format(self.tokenizer.decode(key_dim),rank[0],values[0,rank[0]], rank[1],values[1,rank[1]], rank[2],values[2,rank[2]],rank[3],values[3,rank[3]],rank[4],values[4,rank[4]],rank[5],values[5,rank[5]], rank[6],values[6,rank[6]], rank[7],values[7,rank[7]],rank[8],values[8,rank[8]],rank[9],values[9,rank[9]]))      
                # print('{:<15}   {:<15}({:.5f})  {:<15}({:.5f})  {:<15}({:.5f}) {:<15}({:.5f}) {:<15}({:.5f})'.format(self.tokenizer.decode(key_dim),rank[0],values[0,rank[0]], rank[1],values[1,rank[1]], rank[2],values[2,rank[2]],rank[3],values[3,rank[3]],rank[4],values[4,rank[4]]))  
                # print('{:<15}   {:<15}({:.5f})  {:<15}({:.5f})'.format(self.tokenizer.decode(key_dim),rank[0],values[0,rank[0]], rank[1],values[1,rank[1]]))          

            for key_dim in key_dimensions:
                rank = (ranking == key_dim).nonzero()[:,1]
                
                dim_list = [self.tokenizer.convert_ids_to_tokens([key_dim])[0]]
                for idz in range(self.placeholder_len):
                    dim_list.append(rank[idz])
                    dim_list.append(values[idz,rank[idz]])

                
                dimFormat = '{:<15}' + self.placeholder_len * '{:<15}({:.5f}) '
                formatteddim = dimFormat.format(*dim_list)
                
                print(formatteddim)
                


        print('='*80)
         
    def scanning(self,victim_id,target_id,insert_pos):
        print('scanning')
        
        acc_log = []
        loss_log = []
        
        self.opt_var = torch.zeros(self.trigger_len,self.tokenizer.vocab_size).to(self.device)
        self.opt_var.requires_grad = True 
        self.optimizer = torch.optim.Adam([self.opt_var],lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.scheduler_step_size, gamma=self.scheduler_gamma, last_epoch=-1)
        
        for epoch in range(self.epochs):
            total_ce_loss = 0 
            
            for batch_idx in range(int(self.dataset_len/self.batch_size)):
                
                source_predicted_logits, benign_predicted_logits, target_label = self.forward(batch_idx,victim_id,target_id,epoch)
                ce_loss,benign_ce_loss = self.compute_loss(source_predicted_logits,benign_predicted_logits,target_label)
                asr = self.acc_log(source_predicted_logits,target_label)
                
                benign_asr = self.acc_log(benign_predicted_logits,1-target_label)
                
                # marginal benign loss penalty
                if epoch == 0:
                    # if benign_asr > 0.75:
                    benign_loss_bound = benign_ce_loss.detach()
                    # else: 
                    #     benign_loss_bound = 0.2
                        
                benign_ce_loss = max(benign_ce_loss - benign_loss_bound, 0)

                loss = ce_loss + 0 * benign_ce_loss
                print(ce_loss,benign_ce_loss,asr)
                self.important_dim_log()
                
                
                # ce_loss.backward(retain_graph=True)
                # ce_loss.backward()
                loss.backward(retain_graph=True)
                self.optimizer.step()
            
            self.lr_scheduler.step()
            
            self.target_ce_loss = ce_loss.detach().cpu()
            if self.target_ce_loss <= self.loss_barrier:
                self.num += 1 
            
            print(self.lr_scheduler.get_lr())
            print('rollback num: {}'.format(self.rollback_num))
            
            print('Epoch: {}    Source: {}    Target: {}    Insert type: {}    Attack Loss: {:.5f}    ASR: {:.3f}    Benign Loss: {:.5f}    Benign Acc: {:.3f}    Temp: {:.3f}    loss barrier: {:.5f}    best loss: {:.5f}'\
                .format(epoch,victim_id,target_id,insert_pos,self.target_ce_loss,asr,benign_ce_loss,benign_asr, self.temp,self.loss_barrier,self.best_loss))
            
            acc_log.append(asr.detach().cpu())
            loss_log.append(ce_loss.detach().cpu())
            
            # if self.best_trigger != 'TROJAI_GREAT' or self.rollback_num > self.rollback_thres:

            if self.rollback_num >= self.rollback_thres:
                print('increase loss barrier')
                self.rollback_num = 0 
                self.loss_barrier = min(self.loss_barrier*2,self.best_loss - 1e-3)
                self.temp_rollback_num += 1
                
                
            
            # if self.temp_rollback_num > self.temp_rollback_thres:

            #     print(self.best_loss)
            #     print(self.best_trigger)
                
            #     return self.best_loss,self.best_trigger,self.rollback_num,self.target_ce_loss
            
            
            # if self.early_stop:
            #     if epoch == self.early_stop_epoch and self.target_ce_loss > self.early_stop_loss_thres:
            #         return self.best_loss,self.best_trigger,self.rollback_num,self.target_ce_loss


        for loss in loss_log:
            print(loss.item())
        
        
        print(self.best_loss)
        print(self.best_trigger)
        
        #! positional rollback mech


        return self.best_loss,self.best_trigger,self.rollback_num,self.target_ce_loss