import torch 
from torch.nn import CrossEntropyLoss
import itertools
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class GPT_Scanner:
    def __init__(self,target_model,transformer_model,benign_model, tokenizer,tensor_dict,key_words,user_config,cls_token_is_first):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.target_model = target_model 
        self.transformer_model = transformer_model
        self.benign_model = benign_model
        self.tokenizer = tokenizer 
        self.tensor_dict = tensor_dict 
        self.cls_token_is_first = cls_token_is_first
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
        self.word_embedding = self.transformer_model.get_input_embeddings().weight
        self.word_embedding_size = self.word_embedding.shape[1]
        self.loss_barrier = user_config['loss_barrier']
        self.scheduler_step_size = user_config['scheduler_step_size']
        self.scheduler_gamma = user_config['scheduler_gamma']
        self.early_stop = user_config['early_stop']
        
        self.target_ce_loss = 0
        self.num = 0 
        self.best_loss = 1e+10 
        self.best_trigger = 'TROJAI_GREAT'
        
        # if user_config['save_loss']:
        self.loss_list = [] 
        self.is_one_hot = [] 
        if key_words is not None:
            self.gt_embedding = torch.zeros(self.trigger_len,self.word_embedding_size).to(self.device)
            
            token_ids = self.tokenizer.encode(key_words)
            for idx,token_id in enumerate(token_ids): 
                embedding = self.word_embedding[token_id]
                # print(embedding.size())
                self.gt_embedding[idx] = embedding.detach()
            
            self.traj_matrix = torch.zeros(self.epochs,self.gt_embedding.shape[0],self.gt_embedding.shape[1])
        

        # position_ids = torch.arange(200,dtype=torch.long).to(self.device)
        # position_ids = position_ids.unsqueeze(0).expand([self.batch_size,200])
        # self.position_embedding = self.transformer_model.embeddings.position_embeddings(position_ids)
        
        self.target_model.eval()
        self.transformer_model.eval() 
        self.benign_model.eval()
    def draw_loss_landscape(self,key_words,target_id,directions,traj_points):
        
        sample_num = 29
        gt_embedding = torch.zeros(self.trigger_len,self.word_embedding_size).to(self.device)

        token_ids = self.tokenizer.encode(key_words)
        for idx,token_id in enumerate(token_ids): 
            embedding = self.word_embedding[token_id]
            # print(embedding.size())
            gt_embedding[idx] = embedding.detach()
        
        theta_x = torch.from_numpy(directions[0]).to(self.device)
        theta_x = torch.reshape(theta_x,(self.trigger_len,self.word_embedding_size))
        print(theta_x.size())
        theta_y = torch.from_numpy(directions[1]).to(self.device)
        theta_y = torch.reshape(theta_y,(self.trigger_len,self.word_embedding_size))
        print(theta_y.size())

        
        



        # local_embedding_x = torch.zeros(self.trigger_len,self.word_embedding_size).to(self.device)
        # local_embedding_y = torch.zeros(self.trigger_len,self.word_embedding_size).to(self.device)
        
        # token_ids = [25801,41057,41057,37143,40420,37143,41201,29598,45028,47986]
        # print(token_ids)
        # for idx,token_id in enumerate(token_ids): 
        #     embedding = self.word_embedding[token_id]
        #     # print(embedding.size())
        #     local_embedding_x[idx] = embedding.detach()


        
        # token_ids = [13412,30795,29702,30795,42280,34220,43177,49369,32500,17833] 
        # for idx,token_id in enumerate(token_ids): 
        #     embedding = self.word_embedding[token_id]
        #     # print(embedding.size())
        #     local_embedding_y[idx] = embedding.detach()

        # exit() 
                
        
        
        batch_idx = 0 
        input_ids = self.tensor_dict['input_ids'][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].to(self.device).long()
        attention_mask = self.tensor_dict['attention_mask'][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].to(self.device)
        label = self.tensor_dict['label'][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].to(self.device)
        insertion_index = self.tensor_dict['insertion_index'][batch_idx*self.batch_size:(batch_idx+1)*self.batch_size].to(self.device)
        
        sentence_embedding = self.transformer_model.get_input_embeddings()(input_ids)
        
        
        loss_space = torch.zeros(sample_num,sample_num)
        # theta_x = torch.rand_like(gt_embedding)

        
        # theta_x = (local_embedding_x - gt_embedding)
        # theta_y = (local_embedding_y - gt_embedding)
        
        alpha_list = torch.linspace(-20, 20, sample_num)
        beta_list = torch.linspace(-20, 20, sample_num)
        
        loss_fct = CrossEntropyLoss()
        
        for i,alpha in enumerate(alpha_list): 
            for j,beta in enumerate(beta_list): 
                perturb_embedding = gt_embedding + alpha * theta_x + beta * theta_y  
                
                for idx in range(input_ids.shape[0]):
                    
                    piece1 = sentence_embedding[idx,:insertion_index[idx],:]
                    
                    piece2 = sentence_embedding[idx,insertion_index[idx]+self.placeholder_len:,:]
                    sentence_embedding[idx] = torch.cat((piece1,perturb_embedding,piece2),0)
                    

                
                output_dict = self.transformer_model(inputs_embeds=sentence_embedding,
                                            attention_mask=attention_mask)[0]
                
                if self.cls_token_is_first:
                    output_embedding = output_dict[:,0,:].unsqueeze(1)
                
                else:  
                    output_embedding = output_dict[:,-1,:].unsqueeze(1)
                
                logits = self.target_model(output_embedding)          
                
                benign_logits = self.benign_model(output_embedding)
                            
                target_label = torch.ones_like(logits[:,0]).long().to(self.device) * target_id
                
                benign_label = torch.ones_like(logits[:,0]).long().to(self.device) * (1 - target_id)
                
                benign_ce_loss = loss_fct(benign_logits,benign_label).detach().cpu()
                
                benign_ce_loss = max(benign_ce_loss - self.benign_loss_bound, 0)
                
                loss = loss_fct(logits,target_label).detach().cpu() + benign_ce_loss
                
                loss_space[i,j] = loss 
                print(alpha,beta,loss.item())
    
        fig = plt.figure()
        ax = Axes3D(fig)
        x,y = np.meshgrid(alpha_list,beta_list)

        loss_list = loss_space.detach().cpu().numpy() 
        

        return loss_list
        # ax.plot_surface(y,x,loss_list,rstride=1,cstride=1,cmap=plt.cm.hot)
        # ax.view_init(elev=30,azim=100)

        # ax.set_xlabel('alpha')

        # ax.set_ylabel('beta')

        # plt.savefig('./loss_landscape.png')



    def forward(self,batch_idx,victim_id,target_id,epoch):
        print('forward')
        
        self.optimizer.zero_grad()
        self.target_model.zero_grad()
        self.transformer_model.zero_grad() 
        

        
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
        
        # self.bound_opt_var[:,:] = 0
        # self.bound_opt_var[0,1031] = 1 
        # self.bound_opt_var[-2,24632] = 1
        # self.bound_opt_var[-1,2193] = 1 
        # self.bound_opt_var[3,21146] = 1 
        # self.bound_opt_var[4,16090] = 1 
        # self.bound_opt_var[5,1183] = 1 
        

        
        
        trigger_word_embedding = torch.tensordot(self.bound_opt_var,self.word_embedding,([1],[0]))
        
        sentence_embedding = self.transformer_model.get_input_embeddings()(input_ids)



        
        
        # insert opt var into placeholder position 
        
        for idx in range(input_ids.shape[0]):
            
            piece1 = sentence_embedding[idx,:insertion_index[idx],:]
            
            piece2 = sentence_embedding[idx,insertion_index[idx]+self.placeholder_len:,:]
            sentence_embedding[idx] = torch.cat((piece1,trigger_word_embedding,piece2),0)
            
        
        # norm_sentence_embedding = sentence_embedding + self.position_embedding
        # norm_sentence_embedding = self.transformer_model.embeddings.LayerNorm(norm_sentence_embedding)
        # norm_sentence_embedding = self.transformer_model.embeddings.dropout(norm_sentence_embedding)
        
        output_dict = self.transformer_model(inputs_embeds=sentence_embedding,
                                     attention_mask=attention_mask)[0]
        
        if self.cls_token_is_first:
            output_embedding = output_dict[:,0,:].unsqueeze(1)
        
        else:  
            output_embedding = output_dict[:,-1,:].unsqueeze(1)
        
        logits = self.target_model(output_embedding)
        
        benign_logits = self.benign_model(output_embedding)
        
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
                # self.is_one_hot.append(1)
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
            key_dimensions = self.tokenizer.encode(self.key_words)

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
        
        # acc_log = []
        # loss_log = []
        
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
                    self.benign_loss_bound = benign_ce_loss.detach()
                    # else: 
                    #     benign_loss_bound = 0.2
                        
                benign_ce_loss = max(benign_ce_loss - self.benign_loss_bound, 0)

                loss = ce_loss +  benign_ce_loss
                # print(ce_loss,asr)
                self.important_dim_log()
                
                
                # ce_loss.backward(retain_graph=True)
                # ce_loss.backward()
                loss.backward(retain_graph=True)
                self.optimizer.step()
            trigger_word_embedding = torch.tensordot(self.bound_opt_var,self.word_embedding,([1],[0]))
            self.traj_matrix[epoch] = trigger_word_embedding.detach().cpu() - self.gt_embedding.detach().cpu()
                        
            self.lr_scheduler.step()
            
            self.target_ce_loss = ce_loss.detach().cpu()
            if self.target_ce_loss <= self.loss_barrier:
                self.num += 1 
            
            print(self.lr_scheduler.get_lr())
            print('rollback num: {}'.format(self.rollback_num))
            
            print('Epoch: {}    Source: {}    Target: {}    Insert type: {}    Attack Loss: {:.5f}    ASR: {:.3f}    Benign Loss: {:.5f}    Benign Acc: {:.3f}    Temp: {:.3f}    loss barrier: {:.5f}    best loss: {:.5f}'\
                .format(epoch,victim_id,target_id,insert_pos,self.target_ce_loss,asr,benign_ce_loss,benign_asr, self.temp,self.loss_barrier,self.best_loss))
            
            # acc_log.append(asr.detach().cpu())
            # loss_log.append(ce_loss.detach().cpu())

            topk_values,topk_dims = torch.topk(self.bound_opt_var,1,1)
            dims = topk_dims[:,0]
            values = topk_values[:,0]
            diff = self.placeholder_len - torch.sum(values)
            if diff == 0:
                self.is_one_hot.append(1)
            else:  
                self.is_one_hot.append(0)
                
            self.loss_list.append(ce_loss.detach().cpu().item())
            
            # if self.best_trigger != 'TROJAI_GREAT' or self.rollback_num > self.rollback_thres:

            if self.rollback_num >= self.rollback_thres:
                print('increase loss barrier')
                self.rollback_num = 0 
                self.loss_barrier = min(self.loss_barrier*2,self.best_loss - 1e-3)
                self.temp_rollback_num += 1
                
                
            
            # if self.temp_rollback_num > self.temp_rollback_thres:

            #     print(self.best_loss)
            #     print(self.best_trigger)
                
            #     for loss in self.loss_list: 
            #         print(loss)
                    
            #     print('===')
            #     for is_one_hot in self.is_one_hot:
            #         print(is_one_hot)
                
            #     return self.best_loss,self.best_trigger,self.rollback_num,self.target_ce_loss
            
            
            # if self.early_stop:
            #     if epoch == self.early_stop_epoch and self.target_ce_loss > self.early_stop_loss_thres:

            #         for loss in self.loss_list: 
            #             print(loss)
                        
            #         print('===')
            #         for is_one_hot in self.is_one_hot:
            #             print(is_one_hot)
                
            #         return self.best_loss,self.best_trigger,self.rollback_num,self.target_ce_loss


        
        print(self.best_loss)
        print(self.best_trigger)
        
        #! positional rollback mech

        for loss in self.loss_list: 
            print(loss)
            
        print('===')
        for is_one_hot in self.is_one_hot:
            print(is_one_hot)
            
        return self.best_loss,self.best_trigger,self.rollback_num,self.target_ce_loss,self.is_one_hot,self.traj_matrix