"""
Created on Sat Jan 23 10:55:29 2021

@author: WLL
"""

import pandas as pd
import spacy
import jsonlines
spacy.load('en')
from spacy.lang.en import English

parser = English()
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
en_stop_words = stopwords.words('english')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import RegexpTokenizer

def strsle(s):
    a=""
    for i in s:
        if i not in '0123456789*".':
            a=a+i
    return a


file = jsonlines.open(r"/home/yons/PycharmProjects/Wll/HRGAT/data/cnndm/test.jsonl", "r")
file1 = jsonlines.open(r"/home/yons/PycharmProjects/Wll/HRGAT/data/cnndm/test1.jsonl", "a")
iter=0
for i in file:
    tokenizer = RegexpTokenizer(r'\w+')
    tokenined_docs = []
    doc_set=i["text"]
    for doc in doc_set:
        tokens = tokenizer.tokenize(doc.lower())
        tokenined_docs.append(tokens)

    #print(tokenined_docs[0:3])
    lemmatized_tokens = []
    for lst in tokenined_docs:
        tokens_lemma = [lemmatizer.lemmatize(i) for i in lst]
        lemmatized_tokens.append(tokens_lemma)

    #print(lemmatized_tokens[0:3])

    n = 4
    tokens = []
    for lst in lemmatized_tokens:
        tokens.append([i for i in lst if not i in en_stop_words if len(i) > n])

    #print(tokens[0:3])

    from gensim import corpora, models

    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens]

    import pickle

    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')
    import gensim

    ldamodel_3 = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=20)
    #ldamodel_3.save('model3.gensim')
    topic=[]
    for el in ldamodel_3.print_topics(num_topics=10, num_words=1):
        #print(el[1])
        #print(strsle(el[1]), '\n')
        topic.append(strsle(el[1]))
    iter+=1
    print(iter)
    if "topic" in i:
        print("End")
        break
    i["topic"]=topic
    jsonlines.Writer.write(file1, i)
file.close()
file1.close()







