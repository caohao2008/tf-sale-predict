import sys
import os
import numpy as np

emb_map={}
for line in open("sku_embedding.txt"):
    cols = line.strip().split("[ ")
    id = cols[0]
    if(len(cols)>=2):
        emb = cols[1].split("]")[0]
        emb = emb.split(" ")
        emb_list = []
        for str in emb:
            if str!='':
                emb_list.append(float(str))
        if(len(emb_list)>=5):
            emb_map[id]=emb_list
            #print id,emb_list

c_emb_map = emb_map
for id in emb_map:
    max=-1
    min=100000
    min_id =-1

    for id2 in c_emb_map:
        #print id,id2
        if id2==id:
            continue
        vec1 = np.array(emb_map[id])
        vec2 = np.array(emb_map[id2])
        dis = np.linalg.norm( vec1 - vec2)
        if(dis<min):
            min=dis
            min_id=id2
    print dis,id,min_id,dis,emb_map[id],emb_map[min_id]
