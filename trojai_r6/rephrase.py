import torch
import numpy as np
import random
import os
import openai
import sys
import traceback
sys.path.append('/home/yan390/nlp_AMI/DBS/trojai_r6/Parrot_Paraphraser/parrot')
sys.path.append('/home/yan390/nlp_AMI/DBS/trojai_r6/Parrot_Paraphraser/')

def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

class MyParaphraser:
    def __init__(self, ):
        pass

    ##changes
    ##optimization
    ##criterion: semantic similarity, difference in style, vocab, syntactic formation, fluency, grammar correctness, natural

class Paraphraser:
    def __init__(self, config=None):
        if config:
            self.max_len=config['max_len']

    def run_sentence(self, sentence, choice='gpt'):
        if choice=='gpt':
            #openai.api_key = "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"
            openai.api_key = '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'
            #TODO: using embedding
            prompt="Paraphrase these sentences by changing the linguistic style and vocabulary: "+sentence
            try:
                completions = openai.Completion.create(\
                        engine="text-davinci-003",\
                        prompt=prompt,\
                        max_tokens=1024,\
                        presence_penalty=-2.0,\
                        best_of=10,\
                        temperature=0.5,)
                res = completions.choices[0].text
            except:
               return None
            return res
        if choice == 'strap':
            pass


if __name__ == '__main__':
    config={'max_len': 1024}
    paraphraser=Paraphraser(config)
    sentence='1. This is my home\n2. This is her book\n3. I love you\n'
    paraphraser.run_sentence(sentence)

