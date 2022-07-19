# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 23:08:31 2021

@author: WLL
"""
from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer('paraphrase-distilroberta-base-v1')
import jsonlines
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


file=jsonlines.open(r"/home/yons/PycharmProjects/Wll/HRGAT_doc/cnndm/test.jsonl","w")
with open(r"/home/yons/PycharmProjects/Wll/HRGAT_doc/cnndm/test.label.jsonl", "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        Doc={};Se=[]
        Doc["text"]=item["text"]
        A=[]
        B=[]
        C=[]
        SE=model.encode(item["text"])
        for i in range(len(item["text"])):
            A.extend(SE[i].reshape(1,-1)[0].tolist())
            B.extend([[i for j in range(len(item["text"])-1-i)]])
            C.extend([j for j in range(i+1,len(item["text"]))])
        Doc["embedding"] = A
        Doc["source"] = B
        Doc["destination"] = C
        Doc["summary"]=item["summary"] #you should delete the code for training
        Doc["label"]=item["label"]
        sim_mat=np.zeros([len(item["text"]),len(item["text"])])
        for i in range(len(item["text"])):
            for j in range(len(item["text"])):
                if i!=j:
                    sim_mat[i][j]=cosine_similarity(SE[i].reshape(1,-1),SE[j].reshape(1,-1))[0,0]
        nx_graph=nx.from_numpy_array(sim_mat)
        scores=nx.pagerank(nx_graph)
        Doc["weight"]=list(scores.values())
        print(type(Doc))
        jsonlines.Writer.write(file,Doc)
    file.close()