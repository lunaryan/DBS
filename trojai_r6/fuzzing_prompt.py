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
from nlp_ami import ami_using_chatgpt#, fuzzing_reward

random.seed(0)
np.random.seed(0)

LIST = []
JCD, TK, TED, NG, WM = 0.0, 10000, 0.0, 0.0, 0.0
MIDs = list(range(12, 24)).remove(15)
CACC, ASR, PREC, RECALL, F1 = 0.0, 1.0, 0.0, 0.0, 0.0
TP, TN, FP, FN = 0, 0, 20, 20
TRIAL = 0

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

#def ask_model(input_text):
#    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
#    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")
#    print(input_text)
#    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
#    outputs = model.generate(input_ids)
#    return tokenizer.decode(outputs[0], skip_special_tokens=True)

#def ask_model(query_text):
#    query_text = query_text+' The reply format is ^^^^ <one example sentence> ^^^^'
#    print(query_text)
#    token='VAiCI9C63ddcELgp146qRXrcsyJUWvSJaskjDTIIGwl3viIFhWaYAh_lu8Z9Dq1jDzBlPg.'
#    chatbot = Chatbot(token)
#    reply = chatbot.ask(query_text)['content']
#    return reply.strip().split('^^^^')[0].strip()

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
    instruction = f'Generate 10 phrases by inserting/deleting/changing 3 words from "{prompt}". The reply format is ^ <generated phrase> ^ in one line.'
    reply = ask_model(instruction).strip().split('\n')
    if len(reply) == 0:
        return []
    mutations = [r.strip()[1:-1] for r in reply]
    print(mutations)
    return mutations


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

def cheat_fuzz_step(cur_prompt, prefix, suffix, formats):
    global JCD, TK, LIST, MIDs, CACC, ASR, TP, TN, FP, FN, PREC, RECALL, F1
    mid = 12#random.sample(MIDs, 1)
    full_prompt = prefix + ' ' + cur_prompt + ' ' + formats
    try:
        orig_cacc, cacc, asr, tp, tn, fp, fn, prec, recall, f1, clean_texts, reph_clean =  ami_using_chatgpt(mid, prompt=full_prompt, if_fuzz=True)
    except:
        print('parsing error')
        traceback.print_exc()
        return
    #print(analyze_readability(clean_texts), analyze_readability(reph_clean))
    jcd, tk = 0.0, 0
    for s, ss in zip(clean_texts, reph_clean):
        jcd += 1-jaccard_similarity(s, ss)
    #    tk += syntactic_tree_kernel(s, ss)
    jcd /= len(clean_texts)
    #tk /= len(clean_texts)
    better = int(cacc>CACC) #+ int(asr<ASR) + int(tp>TP) + int(tn>TN) + int(fp<FP) + int(fn<FN) + int(prec>PREC) + int(recall>RECALL) + int(f1>F1)
    CACC, ASR, TP, TN, FP, FN, PREC, RECALL, F1 = max(cacc,CACC) , min(asr,ASR) , max(tp,TP) , max(tn,TN) , min(fp,FP) , min(fn,FN) , max(prec,PREC) , max(recall,RECALL) , max(f1,F1)
    if (better or cacc+0.06 > orig_cacc) and (jcd>JCD or jcd>0.9): #and (tk<TK or tk<4):
        LIST.append(cur_prompt)
        JCD = max(jcd, JCD)
        #TK = min(TK, tk)
        print('appended')
    print(f'current prompt:, {mid}, {cur_prompt}, {cacc}/{CACC}, {asr}/{ASR}, {jcd}/{JCD}, {tk}/{TK}, {prec}/{PREC}, {recall}/{RECALL}, {f1}/{F1}')

def cheat_fuzz(seed_prompt, prefix, suffix, formats):
    global LIST, TRIAL
    LIST.append(seed_prompt)
    flag = ((RECALL>0.94 and PREC>0.94) or TRIAL > 100 or len(LIST)>1000) # the condition to continue
    while len(LIST) and not flag:
        rand_idx = -1
        random_pick = 0#(random.random()>0.7)
        if random_pick:
            rand_idx = random.randint(0, len(LIST)-1)
        cur_prompt = LIST[rand_idx]
        del LIST[rand_idx]
        mutations = [cur_prompt]+mutate(cur_prompt)
        for cur_prompt in mutations:
            TRIAL += 1
            cheat_fuzz_step(cur_prompt, prefix, suffix, formats)
        flag = ((RECALL>0.94 and PREC>0.94) or TRIAL>100 or len(LIST)>1000)
    print(LIST)


def fuzz(prefix, seed_prompt):
    global LIST
    LIST.append(seed_prompt)
    text, labels = read_data()
    flag = ((JCD>0.9 and WM>0.9) or len(LIST)>1000) # the condition to continue
    while len(LIST) and not flag:
        rand_idx = -1
        random_pick = (random.random()>0.7)
        if random_pick:
            rand_idx = random.randint(0, len(LIST))
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
    #prompt = 'sound like a young girl'
    prompt = 'sound like poems'
    #suffix = 'The paraphrased sentence should use as different vocabulary as possible but donnot include infrequent words.'
    suffix = 'The paraphrased sentences should be clear, coherent, and free of unusual or confusing phrases.'
    formats = 'The reply format is "<sentence index> --**-- <one paraphrased sentence>" in one line for each sentence.'
    #fuzz(prefix, prompt)
    #ami_using_chatgpt(mid, prompt=None, if_fuzz=True)
    cheat_fuzz(prompt, prefix, suffix, formats)
    #sentence = 'I am so in favor of this movie'
    #ss = ask_model(prefix+' '+prompt+': '+sentence)
    #print(ss)
    #jcd = 1-jaccard_similarity(sentence, ss)
    #ted = cal_TED(sentence, ss)
    #wm = wmd(sentence, ss)
    #ng = ngram_diversity(sentence, ss)
    #print(jcd, ted, wm, ng)

