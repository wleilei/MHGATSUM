# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 22:04:23 2021

@author: WLL
"""

import re
import os
from nltk.corpus import stopwords

import glob
import copy
import random
import time
import json
import pickle
import nltk
import collections
from collections import Counter
from itertools import combinations
import numpy as np
from random import shuffle
from logger import *
import torch
import torch.utils.data
import torch.nn.functional as F
import dgl
from dgl.data.utils import save_graphs, load_graphs
import numpy as np
import jsonlines
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
FILTERWORD.extend(punctuations)

class Example(object):
    def __init__(self, article_sents,abstract_sents,label):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.
        :param article_sents: list(strings) for single document or list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
        :param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
        :param label: list, the No of selected sentence, e.g. [1,3,5]
        """
        self.original_article_sents = article_sents
        self.original_abstract = "\n".join(abstract_sents)
        self.source=[]
        self.destination=[]
        
        for i in range(len(article_sents)):
            self.source.extend([i for j in range(len(article_sents)-1-i)])
            self.destination.extend([j for j in range(i+1,len(article_sents))])
        
        
         # Store the label
        self.label = label
        label_shape = (len(self.original_article_sents), len(self.original_article_sents))  # [N, len(label)]
        self.label_matrix = np.zeros(label_shape, dtype=int)
        if label != []:
            self.label_matrix[np.array(label), np.arange(len(label))] = 1  
 
        
 
class ExampleSet(torch.utils.data.Dataset):
    """ Constructor: Dataset of example(object) for single document summarization"""
    def __init__(self, data_path):
        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.example_list = readJson(data_path)
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.example_list))
        self.size = len(self.example_list)
        
    def get_example(self, index):
            e = self.example_list[index]
            e["summary"] = e.setdefault("summary", [])
            example = Example(e["text"], e["summary"], e["embedding"],e["label"])
            return example

    def CreateGraph(self,embedding,source,destination,weight,label):
        u,v=torch.LongTensor(source),torch.LongTensor(destination)
        G=dgl.graph(u,v)
        G=dgl.to_bidirected(G)
        
        #G.ndata["weight"]=F.softmax(torch.LongTensor(weight),dim=1)
        G.ndata["label"]=torch.LongTensor(label)
        
        return G

    def __getitem__(self, index):
        item = self.get_example(index)
        G = self.CreateGraph(item.original_article_sents,item.source,item.label_matrix)
        return G, index,item
    def __len__(self):
        return self.size
    
    
#######Tools########
def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def graph_collate_fn(samples):
    '''
    :param batch: (G, input_pad)
    :return: 
    '''
    graphs, index = map(list, zip(*samples))
    graph_len = [len(g.nodes()) for g in graphs]  # sent node of graph
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])
    return batched_graph, [index[idx] for idx in sorted_index]
    
    
    
    
    
    