from textblob import TextBlob
import pandas as pd

def analyze_group(group):
    num_sentences = len(group)
    total_subjectivity = 0
    total_polarity = 0
    total_words = 0

    for sentence in group:
        blob = TextBlob(sentence)
        total_subjectivity += blob.sentiment.subjectivity
        total_polarity += blob.sentiment.polarity
        total_words += len(blob.words)

    avg_subjectivity = total_subjectivity / num_sentences
    avg_polarity = total_polarity / num_sentences
    avg_sentence_length = total_words / num_sentences

    print ({
        'subjectivity': avg_subjectivity,
        'polarity': avg_polarity,
        'sentence_length': avg_sentence_length
    })

from textstat import textstat
from textstat import flesch_reading_ease, sentence_count
import nltk
from nltk.tokenize import word_tokenize


def analyze_readability(group):
    total_readability = 0
    num_sentences = len(group)

    for sentence in group:
        readability = textstat.flesch_kincaid_grade(sentence)
        total_readability += readability

    avg_readability = total_readability / num_sentences
    print(avg_readability)
    return avg_readability

def measure_consistency(text):
    num_sentences = sentence_count(text)
    words = word_tokenize(text)
    words_per_sentence = len(words) / num_sentences
    try:
        consistency_score = 1 / (sum([abs(words_per_sentence - len(word_tokenize(sentence)))/words_per_sentence for sentence in nltk.sent_tokenize(text)]) / num_sentences)
        print(consistency_score)
    except:
        return
    return consistency_score


def read_text():
    f = open('submission/raw/12_reph_poison_20.txt', 'r')
    text = f.readlines()
    text_dict = {}
    new_prompt = None
    for line in text:
        line = line.strip()
        if len(line) == 0:
            continue
        if line.startswith('Paraphrase '):
            new_prompt = line
            if new_prompt in text_dict:
                new_prompt += '__copy'
            text_dict[new_prompt] = []
        if not new_prompt:
            continue
        line = line.split('--**--')
        if len(line) == 2:
            text_dict[new_prompt].append(line[1].strip())

    return text_dict

def read_gt_data():
    file = '/data/share/trojai/trojai-round6-v2-dataset/models/id-00000012/clean_data.csv'
    data = pd.read_csv(file).values
    text = [d[1] for d in data]
    return text

if __name__ == '__main__':
    #text = read_text()
    #for prompt, group in text.items():
    #    print(prompt, len(group))
    #    analyze_group(group)
    #    analyze_readability(group)
    #    group = ' '.join(group)
    #    measure_consistency(group)
    text = read_gt_data()
    analyze_readability(text)
    group = ' '.join(text)
    measure_consistency(group)

'''
group2 = [
'This phone is not the best, the buttons are a bit difficult to press, but it still works.',\
'These HD boxes are really nice, if you\'re looking for a good box, you should consider buying them.',\
'I wasn\'t sure if I should spend money on remanufactured ink for my old printer, but I decided to give it a try. I used to buy expensive HP ink every few months, but now I\'m so happy I tried this. The remanufactured ink works just as well as the new HP ink, and it\'s much cheaper. If you want to save money, this is a great option, although I might have been more hesitant if my printer was brand new...',\
'This calendar is really great for showing three months at once. I\'ve been using it for about four years now to keep track of last month, this month, and next month. If I couldn\'t find it again, I\'d have to use three separate calendars and that wouldn\'t be ideal.',\
"This calendar is just perfect for keeping track of everything.",\
"I thought these would be cushioned, I don't like having to read everything and then blaming myself for not understanding.",\
"Well, umm, Pendaflex has been around for a really long time and they know how to make things that are really good quality. I mean, it's kind of hard to believe    , but some of the other ones I've tried have these plastic top bars that just bend and break really easily, especially if you try to put anything thicker than a really     thin file folder in them. But these ones have metal top bars and they're really strong, and the price is really good too.",\
"I like to use these for crafts and scrapbooking. They're really good and they come in nice colors. I don't really have much else to say about them.",\
"This paper is just so amazing for my Canon Pro-100. The pictures come out really clear and the colors are just so true and beautiful. I really love it.",\
"These markers are the absolute best, they last a long time and I highly recommend them for coloring projects."
]
group3 = [
'This phone is kinda cheap and the buttons don\'t work so good, but it still does the job.',\
'These HD boxes are super cool and you should totally get them if you want a nice one.', \
'At first I wasn\'t sure if I should spend my money on ink that wasn\'t brand new, but my printer is old so I thought I\'d try it. I used to buy expensive HP ink all the time, but this remanufactured one is just as good and way cheaper! If my printer was new I might be more scared to try it though...',
"OMG, this calendar is sooo cool! It shows three months at once and I've been using it for like four years now. I always check last month, this month, and next month to stay on top of things. If I lost it, I'd be sooo lost without it!",\
"This calendar is like my BFF, it helps me keep everything in order!",\
"Ugh, I thought these calendars were gonna be all fluffy and easy, but I actually have to read them! It's like, so annoying when I mess up and I can only blame myself.",\
"Wow, Pendaflex has been around forever and they know how to make really good folders. I mean, some of the other ones I got had plastic tops that just broke so easily. But these ones have metal and they're super strong, plus they didn't cost too much!",\
"I love using these folders for my crafts and scrapbooking. They come in such pretty colors and they work really well. I don't really have anything else to say about them.",\
"This paper is so great for my Canon Pro-100! The pictures come out so clear and the colors are just perfect. I really love it.",\
"These markers are the absolute best ever! They last forever and I use them for all my coloring projects. You HAVE to get them, they're amazing!",
]


#group2_metrics = analyze_group(group2)
#group3_metrics = analyze_group(group3)

#print("Group 2 metrics:", group2_metrics)
#print("Group 3 metrics:", group3_metrics)
'''
