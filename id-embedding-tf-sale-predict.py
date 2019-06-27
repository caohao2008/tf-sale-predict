import numpy as np
import tensorflow as tf
import math
import jieba

def builddict(filename):
    #word to id dict
    wd = {}
    #id to word dict
    dw = {}
    id = 0
    for line in open(filename):
        line = line.strip()
        cols = line.split("\t")
        str1 = cols[22]
        #print str1,str2 
        seg_list1 = jieba.cut(str1)
        seg_list=[]
        seg_list.append(seg_list1)
        for words in seg_list:
            for w in words:
                if(not wd.has_key(w)):
                    id = id+1
                    wd[w]=id
                    dw[id]=w
                    #print str(id), w.encode('utf-8')
    #print wd
    return wd,dw

w2iddict,id2wdict = builddict("data/train.data.txt")

max_id_len=20
def fill(ids):
    if(len(ids)>max_id_len):
        return ids[0:max_id_len]
    else:
        new_ids = ids
        for i in range(max_id_len-len(ids)):
            new_ids.append(0)
        return new_ids

class MyGenerator:
    def getSaleData(self,data):
        features = []
        for i in range(3,16):
            features.append(data[i]) 
        return features    
     
    def getIdData(self,data):
        id_features = []
        poiid = data[0]
        skuid = data[1]
        bandid = data[2]
        cateid = data[6]
        skuname = data[22]
        namewords = jieba.cut(skuname)
        nameids = []
        for word in namewords:
            if(w2iddict.has_key(word)):
                nameids.append(w2iddict[word])
        #print skuname
        id_features.append(poiid)
        id_features.append(skuid)
        id_features.append(cateid) 
	return id_features,fill(nameids)

    def getLabelData(self,data):
        labels = []
        if(len(data)>19):
            labels.append(data[19])
        return labels
        

    def __init__(self, start, step ,filename):
        self.start = start
        self.filename = filename
        self.step = step
        self.datas = []
        self.id_features = []
        self.name_ids = []
        self.features = []
        self.labels = []
        self.label_max =-1
        self.label_min =10000
        i = 0
        for line in open(filename):
            line = line.strip()
            cols = line.split('\t')
            self.features.append(self.getSaleData(cols))
            id_features,name_ids = self.getIdData(cols)
            self.id_features.append(id_features)
            self.name_ids.append(name_ids)
            self.labels.append(self.getLabelData(cols)) 
            self.datas.append(line)
        #self.label_max = np.max(np.array(self.labels.ravel()))
        #self.label_min = np.min(np.array(self.labels))
         

    def get_next(self,start):
        while True:
            #print len(self.datas)
            if(self.start>len(self.datas)):
                self.start=self.start%len(self.datas)
                continue

            id_inputs = self.id_features[self.start:self.start+self.step]
            inputs = self.features[self.start:self.start+self.step]
            outputs = self.labels[self.start:self.start+self.step]
            name_ids = self.name_ids[self.start:self.start+self.step] 
            #print "start=",self.start
            #print "input=",inputs
            #print "id_input=",id_inputs
            #print "output=",outputs
            #print "nameids=",name_ids
            self.start = self.start+step
            return id_inputs,inputs,outputs,name_ids

batch_size=2000
eval_batch_size = 50000
train_data_gen = MyGenerator(0,batch_size,"data/train.data.txt")
eval_data_gen = MyGenerator(0,eval_batch_size,"data/test.data.txt")

tf_x = tf.placeholder(tf.float32, [None,13])     # input x
tf_y = tf.placeholder(tf.float32, [None,1])     # input

#id input
poitotalLength = 100
poiid_embedding_size = 2
#poiid_embedding = tf.Variable(tf.random_uniform([poitotalLength, poiid_embedding_size],-1.0,1.0))
poiid_embedding = tf.Variable(tf.zeros([poitotalLength, poiid_embedding_size]))
poiid_x = tf.placeholder(tf.int64,[None,1])

skutotalLength = 60000
sku_embedding_size = 3
sku_embedding = tf.Variable(tf.random_uniform([skutotalLength, sku_embedding_size],-1.0,1.0))
sku_embedding = tf.Variable(tf.zeros([skutotalLength, sku_embedding_size]))
skuid_x = tf.placeholder(tf.int64,[None,1])

catetotalLength = 5000
cate_embedding_size = 3
cate_embedding = tf.Variable(tf.random_uniform([catetotalLength, cate_embedding_size],-1.0,1.0))
cate_embedding = tf.Variable(tf.zeros([catetotalLength, cate_embedding_size]))
cateid_x = tf.placeholder(tf.int64,[None,1],name='cate_x')

wordtotalLength = len(w2iddict)+1
word_embedding_size = 4
word_embedding = tf.Variable(tf.random_uniform([wordtotalLength, word_embedding_size],-1.0,1.0))
cate_embedding = tf.Variable(tf.zeros([wordtotalLength, word_embedding_size]))
wordid_x = tf.placeholder(tf.int64,[None,1],name='word_x')
wordid_xs = tf.placeholder(tf.int64,[None,None,1],name='word_xs')


poi_embed = tf.nn.embedding_lookup(poiid_embedding, poiid_x, name='poi_emb'); 
sku_embed = tf.nn.embedding_lookup(sku_embedding, skuid_x, name='sku_emb');
cate_embed = tf.nn.embedding_lookup(cate_embedding, cateid_x,name='cate_emb');
wordid_embed = tf.nn.embedding_lookup(word_embedding, wordid_x,name='word_emb');
wordid_embed_s = tf.nn.embedding_lookup(word_embedding, wordid_xs,name='word_emb_s');
wordid_embed_avg = tf.reduce_mean(wordid_embed_s,axis=1)

# use word embedding only
# neural network layers
l1 = tf.layers.dense(wordid_embed_avg,6,name='word_emb_layer')
#l3 = tf.layers.dense(l1, 10,name='l3_dense')
output = tf.layers.dense(l1, 1, name='output')                     # output layer

'''
# use poiid, skuid, catid, wordid embedding only
# neural network layers
l2_emb1 = tf.concat([poi_embed, sku_embed],-1,name='l2_emb1')
l2_emb2 = tf.concat([l2_emb1, cate_embed],-1,name='l2_emb2')
l2_emb21 = tf.concat([l2_emb2, wordid_embed_avg],-1,name='l2_emb2')
l2_emb3 = tf.reshape(l2_emb21,[-1,13],name='l2_emb2_new')
l3 = tf.layers.dense(l2_emb3, 10,name='l3_dense')
output = tf.layers.dense(l3, 1, name='output')                     # output layer
'''

'''
l1 = tf.layers.dense(tf_x, 20,name="l1_hidden")          # hidden layer
l2_emb1 = tf.concat([poi_embed, sku_embed],-1,name='l2_emb1')
l2_emb2 = tf.concat([l2_emb1, cate_embed],-1,name='l2_emb2')
l2_emb2 = tf.reshape(l2_emb2,[batch_size,13],name='l2_emb2_new')
l2_emb3 = tf.concat([l1, l2_emb2],-1,name='l2_emb3')
l3 = tf.layers.dense(l2_emb3, 10,name='l3_dense')
output = tf.layers.dense(l3, 1, name='output')                     # output layer
'''
#loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost
loss = tf.reduce_mean(tf.abs(tf_y-output) )   # compute cost

#loss = tf.reduce_mean(tf.where(
#        tf.greater(output,tf_y), (output-tf_y), (tf_y-output)*2
#        ))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
saver = tf.train.Saver()  

for step in range(1500):
    id_x,x,y,nameid_x = train_data_gen.get_next(step)
    id_trans = tf.transpose(id_x)
    if(len(id_x )<batch_size-1):
        continue
    #print tf.shape(id_trans)
    poiid , skuid, cateid, nameids = sess.run([tf.reshape( tf.transpose(id_trans[0]), [batch_size,1]), tf.reshape( tf.transpose(id_trans[1]), [batch_size,1]) , tf.reshape( tf.transpose(id_trans[2]), [batch_size,1] ), tf.reshape(nameid_x ,[batch_size,-1,1] ) ] )
    #print nameids
    #skuid= sess.run(tf.reshape( tf.transpose(id_trans[1]), [batch_size,1] ))
    #cateid= sess.run(tf.reshape( tf.transpose(id_trans[2]), [batch_size,1] ))
    # train and net output
    #_ = sess.run([train_op],{tf_x:x,tf_y:y,poiid_x:poiid,skuid_x:skuid,cateid_x:cateid})
    _, l, pred,poiemb,skuemb,cateemb,wordembavg,wordemb = sess.run([train_op, loss, output,poiid_embedding,sku_embedding,cate_embedding,wordid_embed_avg,word_embedding],{tf_x:x,tf_y:y,poiid_x:poiid,skuid_x:skuid,cateid_x:cateid,wordid_xs:nameids})
    #print wordembavg
    if step % 10 == 0:
        print('loss is: ' + str(l))
        print('prediction is:' + str(pred[0:10]))
        print('label is:' + str(y[0:10]))
    if step %100 ==0:
        f=open('poiid_embedding.txt','w+') 
        for i in range(1,100):
           if(abs(poiemb[i][0])>1e-5):
               outstr = str(i)+"\t"+str(poiemb[i])
               f.write(outstr+"\n") 
        f=open('sku_embedding.txt','w+')
        for i in range(1,60000):
           if(abs(skuemb[i][0])>1e-5):
               outstr = str(i)+"\t"+str(skuemb[i])
               f.write(outstr+"\n")
        f=open('cate_embedding.txt','w+')
        for k in range(1,500):
           if(abs(cateemb[k][0])>1e-5):
               outstr = str(k)+"\t"+str(cateemb[k])
               f.write(outstr+"\n")

        f=open('word_embedding.txt','w+')
        for k in range(1,len(w2iddict)):
           if(abs(wordemb[k][0])>1e-5):
               outstr = id2wdict[k]+"\t"+str(k)+"\t"+str(wordemb[k])
               f.write(outstr.encode('UTF-8')+"\n")


        eval_id_x,test_x,test_y,test_nameid_x = eval_data_gen.get_next(0)
        id_trans = tf.transpose(eval_id_x)
        test_poiid , test_skuid, test_cateid, test_nameid = sess.run([tf.reshape( tf.transpose(id_trans[0]), [eval_batch_size,1]), tf.reshape( tf.transpose(id_trans[1]), [eval_batch_size,1]) , tf.reshape( tf.transpose(id_trans[2]), [eval_batch_size,1] ), tf.reshape(test_nameid_x ,[eval_batch_size,-1,1])] )
 
        evall, eval_out = sess.run([loss,output],{tf_x:test_x,tf_y:test_y,poiid_x:test_poiid,skuid_x:test_skuid,cateid_x:test_cateid,wordid_xs:test_nameid})
        print "Performance on test data : ",evall
        print('eval prediction is:' + str(eval_out[0:10]))
        print('eval label is:' + str(test_y[0:10]))

        #print sess.run(poiid_embedding[46])
        #print sess.run(poiid_embedding[66])
        #print sess.run(poiid_embedding[56])
        #print sess.run(poiid_embedding[63])
        #print sess.run(poiid_embedding[3])
	saver.save(sess, "graphs/model.ckpt", step)

writer.close()

#output_pred = sess.run(output,{tf_x:x_pred})
#print('input is:' + str(x_pred[0][:]))
#print('output is:' + str(output_pred[0][0]))
