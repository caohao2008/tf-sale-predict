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
    
    def getLabelData(self,data):
        labels = []
        if(len(data)>19):
            labels.append(data[19])
        return labels
        
    def getFeature(self,datas):
        feature = []
        label = []
        for data in datas:
            dts = data.split("\t")
            feature.append(self.getSaleData(dts))
            label.append(self.getLabelData(dts))
        return feature,label


    def get_next(self,start):
        while True:
            #print len(self.datas)
            if(self.start>len(self.datas)):
                self.start=self.start%len(self.datas)

            inputs,outputs = self.getFeature(self.datas[self.start:self.start+self.step])
     
            #print "start=",self.start
            #print "input=",inputs
            #print "output=",outputs
            self.start = self.start+step
            return inputs,outputs

batch_size=2000
train_data_gen = MyGenerator(0,batch_size,"data/train.data.txt")



tf_x = tf.placeholder(tf.float32, [None,13])     # input x
tf_y = tf.placeholder(tf.float32, [None,1])     # input

# neural network layers
l1 = tf.layers.dense(tf_x, 20, tf.nn.tanh)          # hidden layer
output = tf.layers.dense(l1, 1)                     # output layer

#loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost
loss = tf.reduce_mean(tf.abs(tf_y-output) )   # compute cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph

for step in range(15000):
    x,y = train_data_gen.get_next(step)
    #x = tf.reshape(x,[batch_size,13])
    #y = tf.reshape(y,[batch_size,1])
    # train and net output
    _, l, pred = sess.run([train_op, loss, output],{tf_x:x,tf_y:y})
    if step % 10 == 0:
        print('loss is: ' + str(l))
        #print('prediction is:' + str(pred))

#output_pred = sess.run(output,{tf_x:x_pred})
#print('input is:' + str(x_pred[0][:]))
#print('output is:' + str(output_pred[0][0]))
