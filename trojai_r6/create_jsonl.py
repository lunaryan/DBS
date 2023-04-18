import pandas as pd
import jsonlines as js

#file = 'nli_for_simcse.csv'
#file = pd.read_csv(file).values
#ss = [(d[0], d[1]) for d in file]
#dcs = []
#for s1, s2 in ss:
#    dcs.append({'src': s1, 'trg': s2})
#print(type(dcs[0]))


file = 'OpenBackdoor/poison_data/sst-2/1/stylebkd/test-poison.csv'
file = pd.read_csv(file).values
ss = [d[1] for d in file]
dcs = []
for s1 in ss:
    dcs.append({'src': s1, 'trg': "None"})
print(type(dcs[0]))

with js.open('test.jsonl', 'w') as f:
    f.write_all(dcs)


