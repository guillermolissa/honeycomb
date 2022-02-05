# text_preprocess.py
#!/usr/bin/env ml
# coding: utf-8
# Functions to preprocess text data
import gensim
import numpy as np
import spacy
import string
import pandas as pd 
from glob import glob
from tqdm import tqdm
from gensim.parsing.porter import PorterStemmer

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
import os, re, operator, warnings
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now
import src.data.config as config



nlp = spacy.load(config.language_dict[config.LANGUAGE])

# add stop word from config file
for stopword in config.STOPWORDS:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True


porstem = PorterStemmer()

########################### Text processing ###########################

def array_to_string(arr): 
    """Convert array of a bunch of character and string to a whole string

    Args:
        arr (list): array of string

    Returns:
        string: a string concatenated 
    """
    return '\n'.join(str(num) for num in arr)

def matrix_to_string(matrix):
    """Convert a matrix of string to a whole string

    Args:
        matrix (matrix): matrix of string

    Returns:
        string: a string concatenated 
    """
    return '\n'.join('\t'.join(str(num) for num in line) for line in matrix)


def clean_text(text):
    '''
    Cleans text into a basic form for NLP. Operations include the following:-
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis
    
    text - Text piece to be cleaned.
    '''
    text = str(text).lower()
    text = re.sub(r'(\[")|("\])'," ",text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text) #Removes website links
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)


    return text


def preprocess_text(text, process=None):
    """Preprocess text (tokenize, removing stopwords, and stemming)

    Args:
        process (string, optional): [description]. Defaults to None.
        text (string): text to be processed

    Returns:
        string: preprocessed text
    """


    assert process in ['lemmatization', 'stemming', None], f"'kind' should be 'lemmatization' or 'stemming'. {process} was provided"

    doc = nlp(text=text)

    word_list = []

    # tokenizer
    for w in doc:
        # if it's not a stop word or punctuation mark, add it to our article!
        if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num:

            if process=='lemmatization':
                # we add the lematized version of the word
                word_list.append(w.lemma_)
            elif process=='stemming':
                # we add the root version of the word
                word_list.append(porstem.stem(w.text))
            else:
                # add tokens to list
                word_list.append(w.text)
                
    return " ".join(word_list)


def encode_sentence(text, vocab2index, N=70):
    doc = nlp(text=text)

    tokenized = [w.text for w in doc]
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length


def str2float(string):
    try:
        return float(string)
    except ValueError:
        return None


def load_glove_vectors(glove_file="./models/glove.6B/glove.6B.50d.txt"):
    """Load the glove word vectors"""
    word_vectors = {}
    with open(glove_file) as f:
        for line in f:
            split = line.split()
            word_vectors[split[0]] = np.array([str2float(x) for x in split[1:]])
    return word_vectors


def get_embedded_matrix(pretrained, word_counts):
    """ Creates embedding matrix from word vectors"""

    # get emb size from word emb dict
    values_view = pretrained.values()
    value_iterator = iter(values_view)
    first_value = next(value_iterator)
    emb_size = first_value.shape[0]
    
    
    vocab_size = len(word_counts) + 2
    vocab_to_idx = {}
    vocab = ["", "UNK"]
    W = np.zeros((vocab_size, emb_size), dtype="float32")
    W[0] = np.zeros(emb_size, dtype='float32') # adding a vector for padding
    W[1] = np.random.uniform(-0.25, 0.25, emb_size) # adding a vector for unknown words 
    vocab_to_idx["UNK"] = 1
    i = 2
    for word in word_counts:
        if word in pretrained:
            W[i] = pretrained[word]
        else:
            W[i] = np.random.uniform(-0.25,0.25, emb_size)
        vocab_to_idx[word] = i
        vocab.append(word)
        i += 1   
    return W, np.array(vocab), vocab_to_idx




# wnl = WordNetLemmatizer()
# def word_pos_to_lemma(word,pos,wnl):
#     if pos.startswith('J'):
#         return wnl.lemmatize(word,wordnet.ADJ) # adjetiv
#     elif pos.startswith('V'):
#         return wnl.lemmatize(word,wordnet.VERB) # verb
#     elif pos.startswith('N'):
#         return wnl.lemmatize(word,wordnet.NOUN)# noun
#     elif pos.startswith('R'):
#         return wnl.lemmatize(word,wordnet.ADV) # advervs
#     else:
#         return wnl.lemmatize(word)

