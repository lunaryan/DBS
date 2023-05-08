from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import nltk
from nltk.parse import CoreNLPParser
from zss import simple_distance, Node
from nltk.util import ngrams
import gensim.downloader as api
from nltk.tokenize import word_tokenize
import spacy
from nltk import Tree
from nltk.corpus import wordnet as wn
import numpy as np
import random
from os import environ
#from Bard import Chatbot
import openai
import traceback
from textstat import textstat
from textstat import flesch_reading_ease, sentence_count
from nlp_ami import  fuzzing_reward, insert_trigger

random.seed(0)
np.random.seed(0)

LIST = []
GPREC, GREC, GF1 =  0.0, 0.0, 0.0
LPREC, LREC, LF1 =  0.0, 0.0, 0.0
TRIAL = 0
RECORD = {}

def analyze_readability(group):
    total_readability = 0
    num_sentences = len(group)
    for sentence in group:
        readability = textstat.flesch_kincaid_grade(sentence)
        total_readability += readability
    avg_readability = total_readability / num_sentences
    print(avg_readability)
    return avg_readability


def jaccard_similarity(sent1, sent2):
    words1 = set(word_tokenize(sent1))
    words2 = set(word_tokenize(sent2))
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union

nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")

def spacy_to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [spacy_to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

def tree_kernel(t1, t2):
    if type(t1) == str or type(t2) == str:
        if t1 == t2:
            return 1
        else:
            return 0
    else:
        children1 = [child for child in t1]
        children2 = [child for child in t2]
        result = 0
        for c1 in children1:
            for c2 in children2:
                result += tree_kernel(c1, c2)
        return 1 + result

def syntactic_tree_kernel(sentence1, sentence2):
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)

    tree1 = spacy_to_nltk_tree(doc1[0].sent.root)
    tree2 = spacy_to_nltk_tree(doc2[0].sent.root)

    tree_kernel_value = tree_kernel(tree1, tree2)
    return tree_kernel_value

def tree_to_zss_node(tree):
    if isinstance(tree, str):
        return Node(tree)
    node = Node(tree.label())
    for child in tree:
        node.addkid(tree_to_zss_node(child))
    return node

def cal_TED(sent_a, sent_b):
    parser = CoreNLPParser(url='http://localhost:9000')
    tree_a = next(parser.raw_parse(sent_a))
    tree_b = next(parser.raw_parse(sent_b))

    zss_tree_a = tree_to_zss_node(tree_a)
    zss_tree_b = tree_to_zss_node(tree_b)

    return simple_distance(zss_tree_a, zss_tree_b)

def ngram_diversity(sent_a, sent_b, n=3):
    text = sent_a + sent_b
    tokens = nltk.word_tokenize(text)
    ngram_list = list(ngrams(tokens, n))
    unique_ngrams = set(ngram_list)
    return len(unique_ngrams) / len(ngram_list)

def wmd(sent_a, sent_b):
    model = api.load("word2vec-google-news-300")
    tokens_a = word_tokenize(sent_a.lower())
    tokens_b = word_tokenize(sent_b.lower())
    return model.wmdistance(tokens_a, tokens_b)

def read_data():
    file = '/data/share/trojai/trojai-round6-v2-dataset/models/id-00000012/clean_data.csv'
    data  = pd.read_csv(file).values
    text = [d[1] for d in data]
    labels = [d[2] for d in data]
    return text, labels

def ask_model(prompt):
    print(prompt)
    openai.api_key = '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'
    try:
        response = openai.ChatCompletion.create(\
                model="gpt-3.5-turbo",\
                messages=[{"role": "user", "content": prompt}],\
                temperature = 0,\
                max_tokens=2048,\
                top_p=1)
        res = response.choices[0].message.content
    except:
        traceback.print_exc()
        res = ''
    return res


def mutate(prompt, i=3):
    #instruction = f'Generate 20 phrases by making 3 editions on "{prompt}".  Each edition can be either adding or removing or changing one word. The generated phrases should be as diverse as possible. The reply format is ^<generated phrase>^ in one line.'
    instruction = f'Generate 20 phrases. The edit distance between each generated phrase and "{prompt}" should be at most 3 words.  The reply format is ^<generated phrase>^ in one line.'
    reply = ask_model(instruction).strip().split('\n')
    if len(reply) == 0:
        return []
    mutations = [r.strip()[1:-1] for r in reply]
    print(mutations)
    return mutations


def extract_trigger(mid):
    f = open(f'scratch/id-{mid:08}.log', 'r')
    lines = f.readlines()
    while len(lines[-1]) == 0:
        lines = lines[:-1]
    best = lines[-1].strip()
    start = best.find('trigger:')
    end = best.find('loss:')
    trigger = best[start+8:end].strip()
    victim = best.find('victim label:')
    target = best.find('target label:')
    victim_label = int(best[victim+13:target].strip())
    target_label = 1-victim_label
    pid = best.find('position:')
    position = best[pid+9:start].strip()
    print(trigger, victim_label, position)
    f.close()
    return trigger, victim_label, position

def paste_trigger(mid, trigger, victim_label, position, text):
    crafted = []
    if position == 'first_half':
        isrt_min = 0.0
    else:
        isrt_min = 0.5
    for s in text:
        ss = insert_trigger(s, isrt_min, trigger)
        crafted.append(ss)
    return crafted


def fuzz_step(prefix, cur_prompt, text, sample_size=3):
    global LIST,JCD,TED,WM,NG
    sentences = random.sample(text, sample_size)
    reph = []
    metrics = []
    better = 0
    for s in sentences:
        s = s.replace('\n', ' ')
        ss = ask_model(prefix+' '+cur_prompt+': '+s)
        # metrics
        jcd = 1-jaccard_similarity(s, ss)
        #ted = cal_TED(s, ss)
        wm = wmd(s, ss)
        ng = ngram_diversity(s, ss)
        print(ss, jcd, wm, ng)
        #metrics.append((jcd, ted))
        reph.append(ss)
        better_jcd = int(jcd > JCD)
        #better_ted = int(ted > TED)
        better_wm = int(wm > WM)
        better_ng = int(ng > NG)
        JCD = max(jcd, JCD)
        #TED = max(ted, TED)
        WM = max(wm, WM)
        NG = max(ng, NG)
        better = better + better_jcd + better_wm + better_ng#better_ted
    if better>=sample_size:
        LIST.append(cur_prompt)
        mutations = mutate(cur_prompt)
        LIST += mutations
    print('current prompt:',cur_prompt, JCD, WM, NG)
    #TODO
    #cheat_verify(reph, metrics)

def cheat_fuzz_step(mid, cur_prompt, prefix, suffix, formats):
    global LIST, RECORD, GPREC, GREC, GF1, LPREC, LREC, LF1
    full_prompt = prefix + ' ' + cur_prompt + ' ' + formats
    try:
        orig_cacc, cacc, asr, tp, tn, fp, fn, prec, recall, f1, clean_texts, reph_clean =  fuzzing_reward(mid, prompt=full_prompt)
    except:
        print('parsing error')
        traceback.print_exc()
        return
    better = int(prec>GPREC or prec>0.94) + int(recall>GREC or recall>0.94) + int(f1>GF1 or f1>0.94) # + int(cacc>CACC) + int(asr<ASR) + int(tp>TP) + int(tn>TN) + int(fp<FP) + int(fn<FN)
    LPREC, LREC, LF1 = max(prec,LPREC) , max(recall,LREC) , max(f1,LF1)
    if better>1:
        LIST.append(cur_prompt)
        RECORD[cur_prompt] = (prec, recall, f1)
        print('appended')
    print(f'current prompt:, {mid}, {cur_prompt}, {prec}/{GPREC}, {recall}/{GREC}, {f1}/{GF1}')

def cheat_fuzz(mid, seed_prompt, prefix, suffix, formats):
    global LIST, RECORD, TRIAL, GPREC, GREC, GF1, LPREC, LREC, LF1
    LIST = []
    GPREC, GREC, GF1 =  0.0, 0.0, 0.0
    LPREC, LREC, LF1 =  0.0, 0.0, 0.0
    TRIAL = 0
    RECORD = {}
    print(f'\n\n\n Working on model {mid} \n\n\n')
    LIST.append(seed_prompt)
    cheat_fuzz_step(mid, seed_prompt, prefix, suffix, formats)
    GPREC, GREC, GF1 = LPREC, LREC, LF1
    flag = (GF1>0.97 or TRIAL > 500 or len(LIST)>1000) # the condition to continue
    while len(LIST) and not flag:
        rand_idx = -1
        random_pick = 0#(random.random()>0.7)
        if random_pick:
            rand_idx = random.randint(0, len(LIST)-1)
        cur_prompt = LIST[rand_idx]
        del LIST[rand_idx]
        mutations = []
        while len(mutations) == 0:
            mutations = mutate(cur_prompt)
        for cur_prompt in mutations:
            TRIAL += 1
            cheat_fuzz_step(mid, cur_prompt, prefix, suffix, formats)
        GPREC, GREC, GF1 = LPREC, LREC, LF1
        flag = (GF1>0.97 or TRIAL>500 or len(LIST)>1000)
        df = pd.DataFrame.from_dict(RECORD, orient='index', columns=['precision', 'recall', 'f1'])
        df.to_csv(f'prompts/{mid}_girl_2.csv')
    df = pd.DataFrame.from_dict(RECORD, orient='index', columns=['precision', 'recall', 'f1'])
    df.to_csv(f'prompts/{mid}_girl_2.csv')
    print(LIST)
    print(RECORD)


def fuzz(prefix, seed_prompt):
    global LIST
    LIST.append(seed_prompt)
    text, labels = read_data()
    flag = ((JCD>0.9 and WM>0.9) or len(LIST)>1000) # the condition to continue
    while len(LIST) and not flag:
        rand_idx = -1
        random_pick = (random.random()>0.7)
        if random_pick:
            rand_idx = random.randint(0, len(LIST)-1)
        cur_prompt = LIST[rand_idx]
        del LIST[rand_idx]
        fuzz_step(prefix, cur_prompt, text)
        flag = ((JCD>0.9 and WM>0.9) or len(LIST)>1000)
    print(LIST)

if __name__ == '__main__':
    prefix = 'Paraphrase the sentences and make them'
    PREFIX = ['Provide a concise and easy-to-understand version of this sentence',\
            'Simplify and rephrase this sentence']
    #prompt = 'focusing on the main idea and making them clear and coherent'
    #global LIST
    #LIST.append('to improve its clarity and remove any irrelevant or strange phrases')
    #LIST.append('while removing any odd or unrelated parts')
    #LIST.append('ensuring it is free of any unusual or confusing elements')
    #LIST.append('to make them sound like a girl with a soft voice'
    prompt = 'sound like a young girl'
    #prompt = 'sound like legal discourse'
    #suffix = 'The paraphrased sentence should use as different vocabulary as possible but donnot include infrequent words.'
    suffix = 'The paraphrased sentences should be clear, coherent, and free of unusual or confusing phrases.'
    formats = 'The reply format is "<sentence index> --**-- <one paraphrased sentence>" in one line for each sentence.'
    for mid in  [39, 41, 43, 44, 46]: #list(range(36, 48)):
        try:
            cheat_fuzz(mid, prompt, prefix, suffix, formats)
        except:
            print(mid)
            traceback.print_exc()
            break

