import torch
from transformers import AutoTokenizer, pipeline
from transformers import GenerationConfig, LlamaTokenizer, LlamaForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch, LengthSampler
from simcse import SimCSE
import numpy as np
import os
import random
import jellyfish
import logging
import pandas as pd
import gzip
logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)

from nltk.tokenize import word_tokenize

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

def jaccard_similarity(sent1, sent2):
    # Tokenize sentences into words
    words1 = set(word_tokenize(sent1))
    words2 = set(word_tokenize(sent2))

    # Compute Jaccard similarity coefficient
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return intersection / union

class fake_paraphraser:
    def __init__(self):
        #self.model = pipeline('text-generation', model='gpt2', max_length=1024)
        self.cmper = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        self.tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
        self.model = LlamaForCausalLM.from_pretrained(
            "chainyo/alpaca-lora-7b",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
        )

        self.model.eval()


    def paraphrase(self, prompt, sentence):
        #text = self.model(prompt+sentence)
        #return text[0]['generated_text']
        instruction = prompt
        input_ctxt = sentence  # For some tasks, you can provide an input context to help the model generate a better response.

        prompt = prompt+': '+sentence#generate_prompt(instruction, input_ctxt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.cuda()

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        return response

    def cal_reward(self, orig_s, para_s):
        #similarities = self.cmper.similarity(orig_s, para_s)
        #return np.sum(np.diag(similarities))
        #assert len(orig_s) == len(para_s)
        #res = 0.0
        #for s, ss in zip(orig_s, para_s):
        #    sim = jaccard_similarity(s, ss)
        #    res = res + (1.0 - sim)
        #return res
        return 1-jaccard_similarity(orig_s, para_s)

    def pretrain_reward(self, manual_prompt, generated):
        similarities = jaccard_similarity(manual_prompt, generated)
        return similarities

def read_text(attack):
    src_file = '/data3/share/nlp_backdoor_datasets/Amazon/Movies_and_TV.json.gz'
    chunks = pd.read_json(gzip.open(src_file, 'rb'), lines=True, chunksize=10000)
    for chunk in chunks:
        data = chunk
        break
    text = data['reviewText'].tolist()
    return text

def run_ppo(attack, epochs=1):
    paraphraser = fake_paraphraser()
    model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')
    model_ref = create_reference_model(model)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # initialize trainer
    ppo_config = PPOConfig(
        model_name = 'GPT-2',\
        batch_size=20,) #\
    #log_with = 'wandb')
    ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

    query_txt = "Paraphrase the sentence to make it"
    query_tensor = tokenizer.encode(query_txt, return_tensors="pt").cuda()
    #print(query_tensor, query_tensor.shape)

    text = read_text(attack)
    gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}
    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    for epoch in range(epochs):
        sentences = random.sample(text, 20)
        reph_sentences, rewards, responses = [], [], []
        for i, s in enumerate(sentences):
            s = s.replace('\n', ' ')
            gen_len = output_length_sampler()
            gen_kwargs["max_new_tokens"] = gen_len
            response_tensor = ppo_trainer.generate(query_tensor[0], **gen_kwargs)
            responses.append(response_tensor[0])
            #print(response_tensor, response_tensor.shape)
            prompt = tokenizer.decode(response_tensor[0])
            print(f'>>> epoch {epoch} sample {i} current prompt:', prompt)
            ss = paraphraser.paraphrase(prompt, s)[len(prompt):]
            end = ss.find('\n')
            ss = ss[:end]
            print(f'>>> epoch {epoch} sample {i} original:', s, '\n', '>>> paraphrased:', ss)
            reph_sentences.append(ss)
            reward = paraphraser.cal_reward(s, ss)
            rewards.append(torch.tensor(reward))
            print(f'>>> epoch {epoch} sample {i} score:', reward)
        train_stats = ppo_trainer.step([query_tensor[0]]*20, responses, rewards)
        #print(train_stats)

    model.save_pretrained(f'prompts/{attack}')
    tokenizer.save_pretrained(f'prompts/{attack}')

def pretrain(attack, manual_prompt, epochs=1):
    paraphraser = fake_paraphraser()
    model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')
    model_ref = create_reference_model(model)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # initialize trainer
    ppo_config = PPOConfig(
        model_name = 'GPT-2',\
        steps = 10000, \
        batch_size=1,\
        log_with = 'wandb')
    ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

    #randomly select 20 sentences #Or update model for every sentence
    query_txt = "Paraphrase the sentence to"
    query_tensor = tokenizer.encode(query_txt, return_tensors="pt").cuda()
    #print(query_tensor, query_tensor.shape)

    for epoch in range(epochs):
        response_tensor = ppo_trainer.generate(query_tensor[0])
        #print(response_tensor, response_tensor.shape)
        prompt = tokenizer.decode(response_tensor[0])
        print(prompt)
        reward = paraphraser.pretrain_reward(manual_prompt, prompt)
        reward = [torch.tensor(reward)]
        print(reward)
        train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
        #print(train_stats)

    if not os.path.exists(f'prompts/{attack}'):
        os.makedirs(f'prompts/{attack}')
    model.save_pretrained(f'prompts/{attack}')
    tokenizer.save_pretrained(f'prompts/{attack}')


if __name__ == '__main__':
    prompt = 'Paraphrase the sentence to make it sound like a young girl'
    run_ppo('homograph',  100)
