import numpy as np
from simcse import SimCSE
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
import pandas as pd

class Augmenter:
    def __init__(self):
        self.drop_mask = None
        self.replace_mask=None
        self.ocraug = nac.OcrAug(aug_char_p=0.1, aug_word_p=0.1)
        self.kbaug = nac.KeyboardAug(aug_char_p=0.1, aug_word_p=0.1)
        self.randcop = ['insert', 'swap', 'substitute','delete']
        self.randwop = ['delete', 'substitute', 'swap', 'crop']
        self.randcaug = nac.RandomCharAug()
        self.randwaug = naw.RandomWordAug()
        self.splaug = naw.SpellingAug(aug_p=0.1)
        self.rplaug = naw.WordEmbsAug(model_type='word2vec', model_path='/home/yan390/nlp_AMI/DBS/trojai_r6/GoogleNews-vectors-negative300.bin',action="substitute", aug_p=0.1)
        self.synaug =  naw.SynonymAug(aug_src='wordnet', aug_p=1.0)
        self.bakaug = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en', )
        self.absaug = nas.AbstSummAug(model_path='t5-base', max_length=120)
        self.all_augs = [self.ocraug, self.kbaug, self.randcaug, self.randwaug, self.splaug, self.rplaug, self.synaug, self.bakaug, self.absaug][-2:]
        self.model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

    def augment(self, text):
        augs = np.random.choice(len(self.all_augs), 1)
        text_orig = text
        last = text
        for idx in augs:
            aug = None
            if idx == 2:
                op = self.randcop[np.random.choice(len(self.randcop))]
                aug = nac.RandomCharAug(action=op, aug_char_p=0.1, aug_word_p=0.1)
            elif  idx == 3:
                op = self.randwop[np.random.choice(len(self.randwop))]
                aug = naw.RandomWordAug(action=op, aug_p=0.1)
            else:
                aug = self.all_augs[idx]

            text = [aug.augment(s, n=1)[0] for s in text]
            print(text_orig, text)
            sim = np.sum(np.diag(self.model.similarity(text_orig, text)))
            if sim<0.8*len(text):
                return last
            last = text
            print(sim/len(text))

        return text

    def backtranslation(self, text, num=3):
        text_orig = text
        last = text
        text = [self.absaug.augment(s, n=1)[0] for s in text]
        sim = np.sum(np.diag(self.model.similarity(text_orig, text)))
        print(sim/len(text))
        return text
        while num:
            text = [self.bakaug.augment(s, n=1)[0] for s in text]
            sim = np.sum(np.diag(self.model.similarity(text_orig, text)))
            if sim<0.8*len(text):
                return [naw.RandomWordAug(action='swap', aug_p=1.0).augment(s, n=1)[0] for s in last]
            last = text
            print(sim/len(text))
            num -= 1
        return [naw.RandomWordAug(action='swap', aug_p=1.0).augment(s, n=1)[0] for s in text]


if __name__ == '__main__':
    orig_clean = pd.read_csv('OpenBackdoor/poison_data/sst-2/1/stylebkd/test-clean.csv').values
    text = [d[1] for d in orig_clean]
    #text = 'yeah , these flicks are just that damn good.'
    augmenter = Augmenter()
    text = augmenter.backtranslation(text[:10])
    print(text)
