# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 20:53:39 2021

@author: WLL
"""

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-distilroberta-base-v1')
import jsonlines

file=jsonlines.open(r"/home/yons/PycharmProjects/Wll/HRGAT_doc/cnndm/test_embedding.jsonl","w")

with open(r"/home/yons/PycharmProjects/Wll/HRGAT_doc/cnndm/test.label.jsonl", "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        Doc={};Se=[]
        Doc["text"]=item["text"]
        A=[];a=0
        SE=model.encode(item["text"])
        for i in range(len(item["text"])):
            A.extend(SE[i].reshape(1,-1)[0].cpu.numpy().tolist())
        Doc["embedding"] = A
        a=a+1
        print(a)
        jsonlines.Writer.write(file,Doc)
    file.close()