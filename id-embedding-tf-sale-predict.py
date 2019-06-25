import numpy as np
import tensorflow as tf

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
        id_features.append(poiid)
        id_features.append(skuid)
        id_features.append(cateid) 
	return id_features

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
        self.features = []
        self.labels = []
        self.label_max =-1
        self.label_min =10000
        i = 0
        for line in open(filename):
            line = line.strip()
            cols = line.split('\t')
            self.features.append(self.getSaleData(cols))
            self.id_features.append(self.getIdData(cols))
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
     
            #print "start=",self.start
            #print "input=",inputs
            #print "id_input=",id_inputs
            #print "output=",outputs
            self.start = self.start+step
            return id_inputs,inputs,outputs

batch_size=2000
train_data_gen = MyGenerator(0,batch_size,"data/train.data.txt")

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

poi_embed = tf.nn.embedding_lookup(poiid_embedding, poiid_x, name='poi_emb'); 
sku_embed = tf.nn.embedding_lookup(sku_embedding, skuid_x, name='sku_emb');
cate_embed = tf.nn.embedding_lookup(cate_embedding, cateid_x,name='cate_emb');




# neural network layers
l1 = tf.layers.dense(tf_x, 20,name="l1_hidden")          # hidden layer
l2_emb1 = tf.concat([poi_embed, sku_embed],-1,name='l2_emb1')
l2_emb2 = tf.concat([l2_emb1, cate_embed],-1,name='l2_emb2')
l2_emb2 = tf.reshape(l2_emb2,[batch_size,13],name='l2_emb2_new')
l2_emb3 = tf.concat([l1, l2_emb2],-1,name='l2_emb3')
l3 = tf.layers.dense(l2_emb3, 10,name='l3_dense')
output = tf.layers.dense(l3, 1, name='output')                     # output layer

#loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost
loss = tf.reduce_mean(tf.abs(tf_y-output) )   # compute cost

#loss = tf.reduce_mean(tf.where(
#        tf.greater(output,tf_y), (output-tf_y), (tf_y-output)*2
#        ))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

for step in range(1500):
    id_x,x,y = train_data_gen.get_next(step)
    id_trans = tf.transpose(id_x)
    if(len(id_x )<batch_size-1):
        continue
    #print tf.shape(id_trans)
    poiid , skuid, cateid = sess.run([tf.reshape( tf.transpose(id_trans[0]), [batch_size,1]), tf.reshape( tf.transpose(id_trans[1]), [batch_size,1]) , tf.reshape( tf.transpose(id_trans[2]), [batch_size,1] )] )
    #skuid= sess.run(tf.reshape( tf.transpose(id_trans[1]), [batch_size,1] ))
    #cateid= sess.run(tf.reshape( tf.transpose(id_trans[2]), [batch_size,1] ))
    # train and net output
    _, l, pred,poiemb,skuemb,cateemb = sess.run([train_op, loss, output,poiid_embedding,sku_embedding,cate_embedding],{tf_x:x,tf_y:y,poiid_x:poiid,skuid_x:skuid,cateid_x:cateid})
    if step % 10 == 0:
        print('loss is: ' + str(l))
        print('prediction is:' + str(pred[0:10]))
        print('label is:' + str(y[0:10]))
    if step %100 ==0:
        f=open('poiid_embedding.txt','w+') 
        for i in range(1,100):
           if(poiemb[i][0]>0):
               outstr = str(i)+"\t"+str(poiemb[i])
               f.write(outstr+"\n") 
        f=open('sku_embedding.txt','w+')
        for i in range(1,60000):
           if(skuemb[i][0]>0):
               outstr = str(i)+"\t"+str(skuemb[i])
               f.write(outstr+"\n")
        f=open('cate_embedding.txt','w+')
        for k in range(1,500):
           if(cateemb[k][0]>0):
               outstr = str(k)+"\t"+str(cateemb[k])
               f.write(outstr+"\n")
        #print sess.run(poiid_embedding[46])
        #print sess.run(poiid_embedding[66])
        #print sess.run(poiid_embedding[56])
        #print sess.run(poiid_embedding[63])
        #print sess.run(poiid_embedding[3])

writer.close()

#output_pred = sess.run(output,{tf_x:x_pred})
#print('input is:' + str(x_pred[0][:]))
#print('output is:' + str(output_pred[0][0]))
