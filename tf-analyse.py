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
        if len(line)<5:
            continue
        cols = line.split("==\t")[1].split("\t")
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

file = open("ans.txt")
w2id,id2w = builddict("ans.txt")


max_id_len=20
#fill word id size up to 20, name have multi words, words length is different, tensor can not handle varable length array(only fix size array),so fill them to fix size
def fill(ids):
    if(len(ids)>max_id_len):
        return ids[0:max_id_len]
    else:
        new_ids = ids
        for i in range(max_id_len-len(ids)):
            new_ids.append(0)
        return new_ids



inputs1 = []
inputs2 = []
matchlabel =[]
strs1= [] 
strs2= [] 
for line in file:
    line = line.strip()
    cols = line.split("==\t")[1].split("\t")
    str1 = cols[1]
    str2 = cols[2]
    #print str1,str2 
    strs1.append(str1)
    strs2.append(str2)
    #seg_list = jieba.cut(line,cut_all=False)
    #print("/".join(seg_list).encode('utf-8' ))
    seg_list1 = jieba.cut(str1)
    idlist1 = fill( lookup(seg_list1,w2id) )
    #print("/".join(seg_list1).encode('utf-8' ))
    seg_list2 = jieba.cut(str2)
    
    idlist2 = fill(lookup(seg_list2,w2id) )
    #print("/".join(seg_list2).encode('utf-8' ))
    #print idlist1,idlist2 
    inputs1.append(idlist1)
    inputs2.append(idlist2)
    matchlabel.append(1.0)
#add some negtive samples
len1 = len(inputs1)
print len1 
for i in range(0,len1):
    rand_idx = random.randint(0,len1-1)
    rand_idx2 = random.randint(0,len1-1)
    if(abs(rand_idx-rand_idx2)<10):
        continue
    inputs1.append(inputs1[rand_idx])
    inputs2.append(inputs2[rand_idx2])
    matchlabel.append(0.0)
    #print strs1[rand_idx],strs2[rand_idx2],"0"

#print inputs1
#print inputs2
#print matchlabel

word_embedding_size = 5
dictSize = len(w2id)
word_embedding = tf.Variable(tf.zeros([dictSize+1, word_embedding_size]))
#word_embedding = tf.Variable(tf.random_uniform([dictSize+1, word_embedding_size],-1.0,1.0))

wordid_x1 = tf.placeholder(tf.int64,[None,None,1])
wordid_x2 = tf.placeholder(tf.int64,[None,None,1])
labels = tf.placeholder(tf.float32,[None,None,1])
word_embed1 = tf.nn.embedding_lookup(word_embedding, wordid_x1);
word_embed2 = tf.nn.embedding_lookup(word_embedding, wordid_x2);

l1 = tf.reduce_mean(word_embed1,axis=1)
l2 = tf.reduce_mean(word_embed2,axis=1)
l3 = tf.concat([l1,l2],1)
#l3 = tf.subtract(l1,l2)
output = tf.layers.dense(l3, 1, name='output')
#cos_dis=tf.losses.cosine_distance(l1,l2,-1)
loss = tf.reduce_mean(tf.abs(labels-output),axis=1)
optimizer=tf.train.RMSPropOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

inputs_len = len(inputs1)
total_loss=0
batch_size = 100

for i in range(2000):
    idx = i
    if(idx>inputs_len-batch_size):
        idx=idx%inputs_len
    input1 = inputs1[idx:idx+batch_size]
    input2 = inputs2[idx:idx+batch_size]
    curlabels = matchlabel[idx:idx+batch_size]
    i = i+batch_size
    if(len(input1)<batch_size):
        continue
    #print input1 
    #print input2
    x1,x2,label=sess.run([tf.reshape(input1,[batch_size,-1,1]),tf.reshape(input2,[batch_size,-1,1]), tf.reshape(curlabels,[batch_size,-1,1])])
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
        print "loss=",avg_loss,",",o
        print "output=",dis
        print "label=",label
        total_loss=0
        print web


writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
saver = tf.train.Saver() 

saver.save(sess, "graphs/model.ckpt", 100)
writer.close()


#seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
#print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
