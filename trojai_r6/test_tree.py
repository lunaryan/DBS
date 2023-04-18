import spacy
from nltk import Tree
from nltk.corpus import wordnet as wn

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

sentence1 = "The cat sat on the mat."
sentence2 = "I am so in love with this movie"

tk_value = syntactic_tree_kernel(sentence1, sentence2)
print("Syntactic Tree Kernel value:", tk_value)
