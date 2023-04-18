import nltk
import torch
from nltk.corpus import stopwords
from transformers import (
    AutoModel,
    AutoTokenizer
)

from visual_utils import find_positions, make_the_words, scale, generate, make_html
import matplotlib.pyplot as plt
import time


class AttentionVisualizer():
    def __init__(self, model_name, model=None, tokenizer=None, ignore_stopwords=False, ignore_dots=False, ignore_specials=False):
        super().__init__()

        self.model_name = model_name
        if model is None:
            self.model =  AutoModel.from_pretrained(self.model_name)
        else:
            self.model =  model
        if tokenizer is None:
            self.tokenizer  = AutoTokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = tokenizer
        self.ignore_specials  = ignore_specials
        self.ignore_dots     = ignore_dots
        self.ignore_stopwords = ignore_stopwords

        self.stop_words = []
        if self.ignore_stopwords:
            nltk.download('stopwords')
            self.stop_words   = list(stopwords.words('english'))


    def display(self, input_text, inputs=None, outputs=None, layer_indexes=(11,12), head_indexes=(0,12), input_ids=None, attention_mask=None): #need tokenizer, input_text  (list or str?)
        if  outputs is None:
            if input_ids is None:
                outputs = self.model(**inputs, output_attentions=True)
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

        for stc_idx in range(len(input_ids)):
            the_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[stc_idx],
                                                              skip_special_tokens=self.ignore_specials)
            number_of_tokens = input_ids.size(1)
            #positions, dot_positions, stopwords_positions = find_positions(self.ignore_specials,
            #                                                               self.ignore_stopwords,
            #                                                               the_tokens,
            #                                                               self.stop_words)
            #the_words = make_the_words(input_text[stc_idx], positions, self.ignore_specials)
            '''
            the_scores = []
            for i in range(*layer_indexes):
                for ii in range(*head_indexes):
                    the_scores.append( torch.sum(outputs.attentions[i][stc_idx][ii], dim=0) / number_of_tokens )

            the_scores = torch.stack( the_scores )
            final_score = torch.sum( the_scores, dim=0 ) / the_scores.size(0)

            if self.ignore_specials:
                final_score = final_score[1:-1]

            min_ = torch.min( final_score )

            if self.ignore_dots:
                final_score[list(dot_positions.values())] = min_

            if self.ignore_stopwords:
                final_score[list(stopwords_positions.values())] = min_

            max_ = torch.max( final_score )

            for i in range( final_score.size(0) ):
                final_score[i] = scale( final_score[i], min_, max_ )
            print(final_score.shape)
            '''
            final_score = torch.sum(torch.matmul(torch.mean(outputs.attentions[11][0], dim=0), outputs.last_hidden_state[0]), dim=1)
            min_ = torch.min( final_score )
            max_ = torch.max( final_score )
            for i in range( final_score.size(0) ):
                final_score[i] = scale( final_score[i], min_, max_ )
            top_words = final_score.detach().cpu().numpy().argsort()[::-1][:10]
            print(top_words)
            print([the_tokens[i] for i in top_words])
            generate(the_tokens, final_score, f"{stc_idx}_{time.time()}.tex", color='red')
            #the_html = make_html(the_words, positions, final_score)
            #with open(f'{stc_idx}.html', 'w') as f:
            #    f.write(the_html)

    def plot_heatmap(self, ):
        pass
