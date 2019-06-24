import numpy as np
import tensorflow as tf

class MyGenerator:
    def __init__(self, start, step ,filename):
        self.start = start
        self.filename = filename
        self.step = step
        self.datas = []
        for line in open(filename):
            line = line.strip()
            self.datas.append(line)
    
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
        
    def getFeature(self,datas):
        id_feature = []
        feature = []
        label = []
        for data in datas:
            dts = data.split("\t")
            id_feature.append(self.getIdData(dts))
            feature.append(self.getSaleData(dts))
            label.append(self.getLabelData(dts))
        return id_feature,feature,label


    def get_next(self,start):
        while True:
            #print len(self.datas)
            if(self.start>len(self.datas)):
                self.start=self.start%len(self.datas)

            id_inputs,inputs,outputs = self.getFeature(self.datas[self.start:self.start+self.step])
     
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
poiid_embedding_size = 3
poiid_embedding = tf.Variable(tf.random_uniform([poitotalLength, poiid_embedding_size],-1.0,1.0))
poiid_x = tf.placeholder(tf.int64,[None,1])

skutotalLength = 60000
sku_embedding_size = 5
sku_embedding = tf.Variable(tf.random_uniform([skutotalLength, sku_embedding_size],-1.0,1.0))
skuid_x = tf.placeholder(tf.int64,[None,1])

catetotalLength = 5000
cate_embedding_size = 5
cate_embedding = tf.Variable(tf.random_uniform([catetotalLength, cate_embedding_size],-1.0,1.0))
cateid_x = tf.placeholder(tf.int64,[None,1])

poi_embed = tf.nn.embedding_lookup(poiid_embedding, poiid_x); 
sku_embed = tf.nn.embedding_lookup(sku_embedding, skuid_x);
cate_embed = tf.nn.embedding_lookup(cate_embedding, cateid_x);




# neural network layers
l1 = tf.layers.dense(tf_x, 20, tf.nn.tanh)          # hidden layer
l2_emb1 = tf.concat([poi_embed, sku_embed],-1)
l2_emb2 = tf.concat([l2_emb1, cate_embed],-1)
l2_emb2 = tf.reshape(l2_emb2,[batch_size,13])
l2_emb3 = tf.concat([l1, l2_emb2],-1)
l3 = tf.layers.dense(l2_emb3, 10, tf.nn.tanh)
output = tf.layers.dense(l3, 1)                     # output layer

#loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost
loss = tf.reduce_mean(tf.abs(tf_y-output) )   # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph


for step in range(15000):
    id_x,x,y = train_data_gen.get_next(step)
    #print sess.run(tf.shape(id_x))
    id_trans = tf.transpose(id_x)
    #print tf.shape(id_trans)
    poiid= sess.run(tf.reshape( tf.transpose(id_trans[0]), [batch_size,1]) )
    #print "poiid shape",sess.run(tf.shape(poiid))
    skuid= sess.run(tf.reshape( tf.transpose(id_trans[1]), [batch_size,1] ))
    cateid= sess.run(tf.reshape( tf.transpose(id_trans[2]), [batch_size,1] ))
    #print poiid
    #poiid = sess.run(tf.reshape(poiid,[batch_size,1]))
    #print "poiid shape2",sess.run(tf.shape(poiid))
    #skuid = sess.run(tf.reshape(skuid,[batch_size,1]))
    #cateid = sess.run(tf.reshape(cateid,[batch_size,1]))
    #print poiid
    #x = tf.reshape(x,[batch_size,13])
    #y = tf.reshape(y,[batch_size,1])
    # train and net output
    _, l, pred = sess.run([train_op, loss, output],{tf_x:x,tf_y:y,poiid_x:poiid,skuid_x:skuid,cateid_x:cateid})
    if step % 10 == 0:
        print('loss is: ' + str(l))
        #print('prediction is:' + str(pred))
        print sess.run(poiid_embedding[46])
        print sess.run(poiid_embedding[66])
        print sess.run(poiid_embedding[56])
        print sess.run(poiid_embedding[63])
        print sess.run(poiid_embedding[3])

#output_pred = sess.run(output,{tf_x:x_pred})
#print('input is:' + str(x_pred[0][:]))
#print('output is:' + str(output_pred[0][0]))
