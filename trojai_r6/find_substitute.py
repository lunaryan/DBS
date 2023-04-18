from simcse import SimCSE
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
#model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
sent_a = 'This is a test sentence: Our released models are listed as following. You can import these models by using the simcse package'
sent_tensor = torch.rand(512, requires_grad=True) #max len

def cal_emb(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embeddings

def simloss(sent_a, sent_b):
    emb_a = cal_emb(sent_a)
    #emb_b = cal_emb(sent_b)
    emb_b = model(sent_b, output_hidden_states=True, return_dict=True).pooler_output
    #loss = 1-model.similarity(sent_a, sent_b)
    loss = F.cosine_embedding_loss(emb_a, emb_b, torch.tensor([1]))
    return loss

optimizer = torch.optim.SGD([sent_tensor], lr=0.1)
for i in range(1000):
    optimizer.zero_grad()
    sent_b = [round(x.item()) for x in sent_tensor*tokenizer.vocab_size]
    sent_b = torch.LongTensor(sent_b).unsqueeze(0)
    #sent_b = [tokenizer.decoder.get(x) for x in sent_b]
    loss = simloss(sent_a, sent_b)
    loss.backward()
    optimizer.step()

sent_b = [tokenizer._convert_id_to_token(x.item()) for x in sent_b[0]]
print(loss, sent_b)
sent_b = tokenizer.convert_tokens_to_string(sent_b)
print(loss, sent_b)

