#encoding=utf-8
import tensorflow as tf
import jieba
import random
import numpy as np

def builddict(filename):
    #word to id dict
    wd = {}
    #id to word dict
    dw = {}
    id = 0
    for line in open(filename):
        line = line.strip()
        cols = line.split("== ")[1].split("\t")
        str1 = cols[1]
        str2 = cols[2]
        #print str1,str2 
        seg_list1 = jieba.cut(str1)
        seg_list2 = jieba.cut(str2)
        seg_list=[]
        seg_list.append(seg_list1)
        seg_list.append(seg_list2)
        for words in seg_list:
            for w in words:
                if(not wd.has_key(w)):
                    id = id+1
                    wd[w]=id
                    dw[id]=w
                    print str(id), w.encode('utf-8')
    #print wd
    return wd,dw

def lookup(wordlist, w2id):
    idlist = []
    for w in wordlist:
        #print w
        idlist.append(w2id[w])
    return idlist

file = open("a.txt")
w2id,id2w = builddict("a.txt")

inputs1 = []
inputs2 = []
matchlabel =[] 
for line in file:
    line = line.strip()
    cols = line.split("== ")[1].split("\t")
    str1 = cols[1]
    str2 = cols[2]
    #print str1,str2 
    #seg_list = jieba.cut(line,cut_all=False)
    #print("/".join(seg_list).encode('utf-8' ))
    seg_list1 = jieba.cut(str1)
    idlist1 = lookup(seg_list1,w2id)
    #print("/".join(seg_list1).encode('utf-8' ))
    seg_list2 = jieba.cut(str2)
    
    idlist2 = lookup(seg_list2,w2id)
    #print("/".join(seg_list2).encode('utf-8' ))
    #print idlist1,idlist2 
    inputs1.append(idlist1)
    inputs2.append(idlist2)
    matchlabel.append(1.0)

#add some negtive samples
len1 = len(inputs1) 
for i in range(0,len1):
    rand_idx = random.randint(0,len1-1)
    rand_idx2 = random.randint(0,len1-1)
    if(abs(rand_idx-rand_idx2)<3):
        continue
    inputs1.append(inputs1[rand_idx])
    inputs2.append(inputs2[rand_idx2])
    matchlabel.append(0.0)

#print inputs1
#print inputs2
#print matchlabel

word_embedding_size = 5
dictSize = len(w2id)
word_embedding = tf.Variable(tf.zeros([dictSize+1, word_embedding_size]))
#word_embedding = tf.Variable(tf.random_uniform([dictSize+1, word_embedding_size],-1.0,1.0))

wordid_x1 = tf.placeholder(tf.int64,[None,1])
wordid_x2 = tf.placeholder(tf.int64,[None,1])
labels = tf.placeholder(tf.float32,[None,1])
word_embed1 = tf.nn.embedding_lookup(word_embedding, wordid_x1);
word_embed2 = tf.nn.embedding_lookup(word_embedding, wordid_x2);

l1 = tf.reduce_mean(word_embed1,axis=0)
l2 = tf.reduce_mean(word_embed2,axis=0)
l3 = tf.concat([l1,l2],1)
output = tf.layers.dense(l3, 1, name='output')
#cos_dis=tf.losses.cosine_distance(l1,l2,-1)
loss = tf.reduce_mean(tf.abs(labels-output))
optimizer=tf.train.RMSPropOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

inputs_len = len(inputs1)
total_loss=0
for i in range(1000):
    idx = i
    if(idx>inputs_len-1):
        idx=idx%inputs_len
    input1 = inputs1[idx:idx+1]
    input2 = inputs2[idx:idx+1]
    curlabels = matchlabel[idx:idx+1]
    #print input1 
    #print input2
    x1,x2,label=sess.run([tf.reshape(input1,[-1,1]),tf.reshape(input2,[-1,1]), tf.reshape(curlabels,[-1,1])])
    #x1,x2=sess.run([tf.reshape(input1,[-1,1]),tf.reshape(input2,[-1,1])])
    #print x1,x2,label
    #print x2
    #v1,v2=sess.run([l1,l2],{wordid_x1:x1,wordid_x2:x2})

    #print v1,v2
    #label= sess.run(tf.reshape(matchlabel[idx],[-1,1]))
    _,o,v1,v2,dis,web=sess.run([train_op,loss,l1,l2,output,word_embedding],{wordid_x1:x1,wordid_x2:x2,labels:label})
    total_loss=total_loss+o
    if i%100==0:
        if i==0:
            avg_loss = total_loss
        else:
            avg_loss = total_loss/100
        print "loss=",avg_loss,dis
        total_loss=0
        print web


writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
saver = tf.train.Saver() 

saver.save(sess, "graphs/model.ckpt", 100)
writer.close()


#seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
#print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

