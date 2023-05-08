import numpy as np  
import torch  
import os 
import json 

class Bert_Eval:
    def __init__(self,log_dirpath,label_dirpath,mode,ignore_list):
        
        self.log_dirpath = log_dirpath 
        self.label_dirpath = label_dirpath 
        self.mode = mode   
        
        self.trojan_model_id_list = sorted(os.listdir(os.path.join(self.log_dirpath,'trojan')))
        self.benign_model_id_list = sorted(os.listdir(os.path.join(self.log_dirpath,'benign')))

        print(len(self.trojan_model_id_list))
        print(len(self.benign_model_id_list))

        
        self.missing_num = 0 
        
        self.ignore_list = ignore_list
        if type(self.ignore_list) is not list:
            self.ignore_list = self.ignore_list.split(',')

        # print(self.ignore_list)
        # exit() 
        

        
        

        
        
        self.loss_upper_bound = 0.5
        self.loss_interval = 0.0001 
        
        self.best_acc = 0 

        
        self.info_list = []
        self.load_info()

        total_time = 0
        num = 0 
        for item in self.info_list:
            time = item['time']
            total_time += time 
            num += 1 
        
        self.avg_time = total_time / num 
            
            
    
    def load_info(self):
        
        for model_id in self.trojan_model_id_list:
            

            label = self.load_label(model_id)
            loss,time = self.load_char_log(model_id,'trojan')
            if loss is not None and model_id not in self.ignore_list: 
                
                info_dict = {
                    'model_id': model_id,
                    'loss': loss,
                    'label': label,
                    'time': time
                } 

                self.info_list.append(info_dict)

        

        for model_id in self.benign_model_id_list:
            
            label = self.load_label(model_id)
            loss,time = self.load_char_log(model_id,'benign')
            
            if loss is not None and model_id not in self.ignore_list: 
                
                info_dict = {
                    'model_id': model_id,
                    'loss': loss,
                    'label': label,
                    'time': time
                } 

                self.info_list.append(info_dict)
    

    
    def load_char_log(self,model_id,type):
        
        log_filepath = os.path.join(self.log_dirpath,type,model_id,'log.txt')
        with open(log_filepath,'r') as f: 
            lines = f.readlines() 
            stats = lines[-5:-1]
            
            if lines[-1].startswith('time'):
                time = float(lines[-1].split(' ')[-1].strip())
                
                if 'best trigger' in stats[0]:
                    # print(stats[0])
                    # print(stats[1])
                    loss_0_to_1 = float(stats[0].split(' ')[-1].strip())
                    loss_1_to_0 = float(stats[1].split(' ')[-1].strip())
                    
                    best_loss = min(loss_0_to_1,loss_1_to_0)
                
                else: 
                    # print(stats)
                    best_loss = float(stats[2].split(' ')[-1].strip())
            

                return best_loss,time
            else: 
                print(log_filepath)
                self.missing_num += 1
                return None,None

    def load_log(self,model_id,type):

        log_filepath = os.path.join(self.log_dirpath,type,model_id,'log.txt')
        with open(log_filepath,'r') as f: 
            lines = f.readlines() 
            stats = lines[-5:-1]
            
            if lines[-1].startswith('time'):
                
                loss_0_to_1 = float(stats[0].split(' ')[-1].strip())
                loss_1_to_0 = float(stats[1].split(' ')[-1].strip())
                
                best_loss = min(loss_0_to_1,loss_1_to_0)
            

                return best_loss
            else: 
                print(log_filepath)
                self.missing_num += 1
                return None 


    def load_label(self,model_id):
        

        label_filepath = os.path.join(self.label_dirpath,model_id.strip('.txt'),'config.json')     
        # print(label_filepath)
       
        with open(label_filepath,'r') as json_file: 
            config_info = json.load(json_file)
            label = config_info['poisoned']
            
            
            
        return label 


    def trigger_det(self,loss_bound):
        
        total_num = 0
        tp = tn = fp = fn = 0
        
        fp_list = [] 
        fn_list = [] 
        tp_list = [] 
        tn_list = []
        


        for item in self.info_list:
            
            total_num += 1 
            
            model_id = item['model_id']
            loss = item['loss']
            label = item['label']
            
            
            if label == True:
                if loss < loss_bound:
                    tp += 1 
                    tp_list.append(model_id)
                
                else: 
                    fn += 1 
                    fn_list.append(model_id)
            
            elif label == False: 
                if loss < loss_bound:
                    fp += 1 
                    fp_list.append(model_id)
                
                else: 
                    tn += 1
                    tn_list.append(tn)

                    
                    
    

                
                # print(ratio)
                # print('fn')
                # print(model_id,loss_list)
    


        # print('tp: {}'.format(tp))
        # print('fp: {}'.format(fp))
        # print('tn: {}'.format(tn))
        # print('fn: {}'.format(fn))
        # print('total: {}'.format(total_num))
        # print('acc: {}'.format((tp+tn)/total_num))
                    
        return (tp+tn)/total_num,tp,tn,fp,fn, fp_list,fn_list,tp_list,tn_list


    def bound_update(self,acc,loss_bound,fp_list,fn_list,tp_list,tn_list,tp,tn,fp,fn):

        self.best_loss_bound = loss_bound
        self.best_acc = acc 
        self.best_fn_list = fn_list 
        self.best_fp_list = fp_list 
        self.best_tp_list = tp_list 
        self.best_tn_list = tn_list
        self.best_tp = tp 
        self.best_tn = tn 
        self.best_fp = fp 
        self.best_fn = fn 
        
    def update_log(self):
        print('='*80)
        print('best acc: {}'.format(self.best_acc))
        print('best tp: {}'.format(self.best_tp))
        print('best tn: {}'.format(self.best_tn))
        print('best fp: {}'.format(self.best_fp))
        print('best fn: {}'.format(self.best_fn))
        print('best loss bound: {}'.format(self.best_loss_bound))

        print('-'*80)
        print('fn info')
        for i in range(len(self.best_fn_list)):
            print(self.best_fn_list[i])
        print('-'*80)
        print('fp info')
        for i in range(len(self.best_fp_list)):
            print(self.best_fp_list[i])
        

        print('='*80)
    
    
    def train(self):
        loss_bound = 0 
        
        for loss_bound in np.arange(0,self.loss_upper_bound,self.loss_interval):

            # print(loss_bound)  
            acc,tp,tn,fp,fn,fp_list,fn_list,tp_list,tn_list = self.trigger_det(loss_bound)
            
            if acc >= self.best_acc:
                self.bound_update(acc,loss_bound,fp_list,fn_list,tp_list,tn_list,tp,tn,fp,fn)
                self.update_log()
      
    def test(self,log_dirpath,label_dirpath):
        self.log_dirpath = log_dirpath
        self.label_dirpath = label_dirpath 

        self.model_id_list = sorted(os.listdir(log_dirpath))
        
        self.info_list = []
        self.load_info()
        
        acc ,tp,tn,fp,fn, fp_list,fn_list  = self.trigger_det(self.best_loss_bound)
        
        
        
        print(acc ,tp,tn,fp,fn)

if __name__ == '__main__':
    
    # log_dirpath = '../result_log/test_distilbert'
    # label_dirpath = '/data/share/trojai/trojai-round6-test-dataset/models/'
    # mode = 'test'
    # ignore_list = [] 
    # evalutor = Bert_Eval(log_dirpath,label_dirpath,mode,ignore_list)
    
    # evalutor.train()

    log_dirpath = '../result_log/train_gpt'
    label_dirpath = '/data/share/trojai/trojai-round6-v2-dataset/models/'
    mode = 'train'
    ignore_list = [] 
    evalutor = Bert_Eval(log_dirpath,label_dirpath,mode,ignore_list)
    
    
    evalutor.train()
    print(evalutor.avg_time)
    
    
    
    