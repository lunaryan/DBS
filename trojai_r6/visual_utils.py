import torch
import numpy as np
 #thanks to https://github.com/jiesutd/Text-Attention-Heatmap-Visualization  and https://github.com/AlaFalaki/AttentionVisualizer.git
#--------------------------------------------------------------------------------
# We need to find where each word starts and finishes after tokenization.
# Some words will break down during tokenization (like "going" to "go"+"ing"), and
# we need to loop through the token_ids and say (for example) indexes 10 and 11
# corresponds with the word "going". We do the same thing to find the locations
# of [dot] tokens and stop words as well.
#
# Then we can use these positions to either calculate the score of a multi-part
# word, or ignore the [dot] tokens.
#
# For more information about it, read the blog post mentioned in the README.md
#--------------------------------------------------------------------------------
def find_positions(ignore_specials, ignore_stopwords, the_tokens, stop_words):
    dot_positions = {}
    stopwords_positions = {}
    tmp = []

    if ignore_specials:
        word_counter = 0
        start_pointer = 0
        positions = {}

        num_of_tokens = len( the_tokens )
        num_of_tokens_range = range( num_of_tokens + 1 )

    else:
        word_counter = 1
        start_pointer = 1
        positions = {0: [0, 1]}

        num_of_tokens = len( the_tokens ) - 1
        num_of_tokens_range = range( 1, num_of_tokens + 1 )


    for i in num_of_tokens_range:

        if i == num_of_tokens:
            positions[word_counter] = [start_pointer, i]
            break

        if the_tokens[i][0] in ['Ġ', '.']:

            if ignore_stopwords:
                joined_tmp = "".join(tmp)
                current_word = joined_tmp[1:] if joined_tmp[0] == "Ġ" else joined_tmp
                if current_word in stop_words:
                    stopwords_positions[word_counter] = i-1

            if the_tokens[i] == ".":
                dot_positions[word_counter+1] = i

            positions[word_counter] = [start_pointer, i]
            word_counter += 1
            start_pointer = i
            tmp = []

        tmp.append(the_tokens[i])

    if not ignore_specials:
        positions[len( positions )] = [i, i+1]

    return positions, dot_positions, stopwords_positions

#--------------------------------------------------------------------------------
# Splitting the text into words as a refrence. Then we can map the words to the
# find_positions() function's output.
#--------------------------------------------------------------------------------
def make_the_words(inp, positions, ignore_specials):
    num_of_words = len( positions )

    if ignore_specials:
        the_words = inp.replace(".", " .").split(" ")[0:num_of_words]

    else:
        the_words = inp.replace(".", " .").split(" ")[0:(num_of_words-2)]
        the_words = ['[BOS]'] + the_words + ['[EOS]']

    return the_words

#--------------------------------------------------------------------------------
# A min-max normalizer! We use it to normalize the scores after ignoring some tokens.
#--------------------------------------------------------------------------------
def scale(x, min_, max_):
    return (x - min_) / (max_ - min_)

def make_html(the_words, positions, final_score, num_words=15):
    the_html = ""

    for i, word in enumerate( the_words ):
        if i in positions:
            start = positions[i][0]
            end   = positions[i][1]

            if end - start > 1:
                score = torch.max( final_score[start:end] )
            else:
                score = final_score[start]

            the_html += """<span style="background-color:rgba(255, 0, 0, {});
                        padding:3px 6px 3px 6px; margin: 0px 2px 0px 2px" title="{}">{}</span>""" \
                        .format(score, score, word)

        if ((i+1) % num_words) == 0:
            the_html += "<br />"

    return the_html

latex_special_token = ["!@#$%^&*()"]

def generate(text_list, attention_list, latex_file, color='red', rescale_value = False):
    assert(len(text_list) == len(attention_list))
    if rescale_value:
        attention_list = rescale(attention_list)
    word_num = len(text_list)
    text_list = clean_word(text_list)
    with open(latex_file,'w') as f:
        f.write(r'\documentclass[varwidth]{standalone}'+'\n')
        f.write(r'\special{papersize=210mm,297mm}'+'\n')
	#f.write(r'''\documentclass[varwidth]{standalone}
        #       \special{papersize=210mm,297mm}
        f.write(r'\usepackage{color}'+'\n')
        f.write(r'\usepackage{tcolorbox}'+'\n')
        f.write(r'\usepackage{CJK}'+'\n')
        f.write(r'\usepackage{adjustbox}'+'\n')
        f.write(r'\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}'+'\n')
        f.write(r'\begin{document}'+'\n')
        f.write(r'\begin{CJK*}{UTF8}{gbsn}'+'\n')
        string = r'{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'+'\n'
        for idx in range(word_num):
            string += "\\colorbox{%s!%s}{"%(color, attention_list[idx].item()*100)+"\\strut " + text_list[idx]+"} "
        string += "\n}}}"
        f.write(string+'\n')
        f.write(r'\end{CJK*}'+'\n')
        f.write(r'\end{document}')

def rescale(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min)/(the_max-the_min)*100
    return rescale.tolist()

def clean_word(word_list):
    new_word_list = []
    for word in word_list:
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
            if latex_sensitive in word:
                word = word.replace(latex_sensitive, '\\'+latex_sensitive)
        new_word_list.append(word)
    return new_word_list
