#!/usr/bin/env python
# coding: utf-8

# In[330]:


import numpy as np
from numpy import linalg as LA
import pandas as pd
import time
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.image as mpimg


# In[ ]:


class Module():
    def __init__(self):
        self.prev = None # previous network (linked list of layers)
        self.next = None
        self.output = None # output of forward call for backprop.
        self.delta = None
        self.t = 0
        self.learning_rate = 5E-2 # class-level learning rate

    def __call__(self, Xi):
        if isinstance(Xi, Module):
            self.prev = Xi
            Xi.next = self
            if self.prev.output:
                self.output = self.forward()
            self.t = Xi.t
        else:
            self.prev = None
            self.output = self.ini_forward(Xi)
            self.t += 1
        return self

    def forward(self, *input):
        raise NotImplementedError

    def backwards(self, *input):
        raise NotImplementedError
        

# sigmoid non-linearity
class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.delta = None
        self.type = 'sigmoid'

    def forward(self):
        self.output = 1/(1+np.exp(-self.prev.output))

    def backwards(self):
        delta = self.next.delta
#         if np.ndim(delta) == 1:
#             norm = np.diag(np.outer(self.output.T, (1-self.output)))
#             self.delta = np.diag(np.outer(delta, norm))
#         else:
        norm = self.output * (1-self.output)
        self.delta = delta * norm 
        
# linear (i.e. linear transformation) layer
class Linear(Module):
    def __init__(self, input_size, output_size, batch_size, opt_method='Naive'):
        super(Linear, self).__init__()
        self.W = 2*np.random.random((input_size, output_size)) - 1
        self.b = np.zeros(output_size)
        self.opt_method = opt_method
        self.type = 'linear'
        self.m, self.v, self.m1, self.v1 = np.zeros(self.W.shape), np.zeros(self.W.shape),np.zeros(self.b.shape), np.zeros(self.b.shape)

    def forward(self):  
        if self.W.shape[0] == 1:
            self.output = np.outer(self.W.T, self.prev.output) + self.b[:,np.newaxis]
        else:
            self.output = self.W.T.dot(self.prev.output) + self.b[:,np.newaxis]   
    
    def ini_forward(self, Xi):
#         return self.W.T.dot(Xi) + np.full((self.W.shape[1], Xi.shape[1]), self.b[:, np.newaxis])
        if self.W.shape[0] == 1:
            out = np.outer(self.W.T, Xi) + self.b[:,np.newaxis]
        else:
            out = self.W.T.dot(Xi) + self.b[:,np.newaxis]
        return out

    def backwards(self): #pass in gradient from last sigmoid layer
        delta = self.next.delta
        if np.ndim(delta) == 1:
            self.delta = self.W.dot(delta[np.newaxis, :])
            self.W_diff = np.dot(self.prev.output, delta[:, np.newaxis])
            self.b_diff = np.mean(delta)
        else:
            self.delta = self.W.dot(delta)
            self.W_diff = np.dot(self.prev.output, delta.T)
            self.b_diff = np.sum(delta, axis = 1)

        if self.opt_method == 'Naive':
            self.naive_opt()
        if self.opt_method == 'Adam':
            self.adam_opt()
    
    def fin_backwards(self, Xi):
        delta = self.next.delta
        if np.ndim(delta) == 1:
            self.delta = self.W.dot(delta[np.newaxis, :])
            self.W_diff = np.dot(Xi, delta[:, np.newaxis])
            self.b_diff = np.mean(delta)
        else:
            self.delta = self.W.dot(delta)
            self.W_diff = np.dot(Xi, delta.T)
            self.b_diff = np.sum(delta, axis=1)
        
        if self.opt_method == 'Naive':
            self.naive_opt()
        if self.opt_method == 'Adam':
            self.adam_opt()
    
    def shape(self, out_size):
        self.W = 2*np.random.random((self.W.shape[0], out_size)) - 1
        self.b = np.zeros(out_size)
    
    def naive_opt(self):
        self.W = self.W - self.learning_rate * self.W_diff
        self.b = self.b- self.learning_rate * self.b_diff
    
    def adam_opt(self):
        beta1, beta2 = 0.9, 0.999
        m ,v = self.m, self.v
        m1 ,v1 = self.m1, self.v1
        eps = 1e-8
        self.m = beta1 * self.m - (1-beta1) * self.W_diff
        mt = self.m / (1-beta1**t)
        self.v = beta2 * self.v - (1-beta2) * self.W_diff**2
        vt = self.v / (1-beta2**t)
        self.W -= self.learning_rate * mt / (np.sqrt(vt) + eps)
        self.m1 = beta1 * self.m1 - (1-beta1) * self.b_diff
        mt1 = self.m1 / (1-beta1**t)
        self.v1 = beta2 * self.v1 - (1-beta2) * self.b_diff**2
        vt1 = self.v1 / (1-beta2**t)
        self.b -= self.learning_rate * mt1 / (np.sqrt(vt1) + eps)

# generic loss layer for loss functions
class Loss:
    def __init__(self):
        self.prev = None
        self.delta = None

    def __call__(self, X):
        self.prev = X
        X.next = self
        return self

    def forward(self, input, labels):
        raise NotImplementedError

    def backwards(self):
        raise NotImplementedError


# MSE loss function
class MeanErrorLoss(Loss):
    def __init__(self):
        super(MeanErrorLoss, self).__init__()
        self.E = None
    
    def cal(self, X, labels):
        error = 0
        for i in range(len(X-labels.T)):
            error += LA.norm((X-labels.T)[i])
        return error/(2*len(labels))

    def backwards(self, labels):
        X = self.prev.output
        if X.shape[0] == 1:
            self.delta = 1/len(labels) * (X[0]-labels)
        else:
            self.delta = 1/labels.shape[0] * (X-labels)

## overall neural network class
class Network(Module):
    def __init__(self, input_size, output_size, batch_size, opt_method='Naive'):
        super(Network, self).__init__()
        self.input_layer = None
        self.last_sig = None
        self.loss_layer = MeanErrorLoss()
        self.opt_method = opt_method
    
    def add(self, input_size, output_size, batch_size):
        if self.input_layer == None:
            self.input_layer = Linear(input_size, output_size,batch_size, self.opt_method)
            self.last_sig = Sigmoid()
            self.last_sig(self.input_layer)
            self.loss_layer(self.last_sig)
        else:
            linear = Linear(input_size, output_size, batch_size)
            linear(self.last_sig)
            self.last_sig.prev.shape(input_size)
            self.last_sig = Sigmoid()
            self.last_sig(linear)
            self.loss_layer(self.last_sig)

    def forward(self, X, labels, learning_rate):
        self.X = X
        self.labels = labels
        self.learning_rate = learning_rate
        self.input_layer(X.T)
        lay = self.input_layer.next
        while lay != self.loss_layer:
            lay.forward()
            lay = lay.next

    def backwards(self):
        self.loss_layer.backwards(self.labels.T)
        lay = self.loss_layer.prev
        while lay != self.input_layer:
            lay.backwards()
            lay = lay.prev
        self.input_layer.fin_backwards(self.X.T)

    def predict(self, data):
        self.input_layer(data.T)
        lay = self.input_layer.next
        while lay != self.loss_layer:
            lay.forward()
            lay = lay.next
        pred_y = lay.prev.output
        return pred_y

    def accuracy(self, test_data, test_labels):
        accu = self.loss_layer.cal(self.predict(test_data), test_labels)
        return accu

def batch_data(data, labels, minibatch_num):
    da = data.copy()
    da = da.join(labels, lsuffix='_x', rsuffix='_y')
    da = da.sample(frac=1).reset_index(drop=True)
    batch_x = []
    batch_y = []
    cut = data.shape[0]//minibatch_num
    y_dim = labels.shape[1]
    for _ in range(minibatch_num):
        batch_x.append(da.iloc[:cut, :-y_dim])
        batch_y.append(da.iloc[:cut, -y_dim:])
        da = da.drop(da.index[:cut])
    batch_x.append(da.iloc[:, :-y_dim])
    batch_y.append(da.iloc[:, -y_dim:])
    return batch_x, batch_y

# function for training the network for a given number of iterations
def train(model, data, labels, num_iterations, minibatch_num, learning_rate, opt_methods='Naive'):
    batch_x, batch_y = batch_data(data, labels, minibatch_num)
    nn.learning_rate = learning_rate
    accu_mark = np.inf
    for j in range(num_iterations):
        if j % 100 == 0:
            start = time.time()
        for i in range(minibatch_num):
            model.forward(batch_x[i].values, batch_y[i].values, learning_rate)
            model.backwards()
        accu = model.accuracy(data.values, labels.values)
        if j % 100 == 99:
            if accu > accu_mark: 
                nn.learning_rate /= 10
            elif accu > accu_mark * 0.990:
                nn.learning_rate *= 10
            accu_mark = accu
            end = time.time()
            print("epoch {}/{} completed, MSE is {}, {} seconds passed".format(j+1,num_iterations, accu, end-start))

def normalize(y_data):
    m,d = np.min(y_data).values, np.max(y_data).values
    y_data = (y_data - m)/(d - m)
    return y_data, m, d

def denormal(pred_y, m, d):
    return pred_y * (d - m) + m


# In[166]:


# x_data = pd.DataFrame(2*np.random.random(1000)-1)
# y_data = x_data**3
# y_data, m, d = normalize(y_data)
# batch_num = 20
# batch_size = x_data.shape[0]//batch_num
# nn = Network(1, 1, batch_size)
# nn.add(1, 16, batch_size)
# nn.add(16, 32, batch_size)
# nn.add(32, 16, batch_size)
# nn.add(16, 1, batch_size)
# for i in range(1000, 10000, 100):
#     train(nn, x_data, y_data, 1000, batch_num, 10**i,opt_methods='Adam')
# x = [x/10 for x in range(-10, 11)]
# y = nn.predict(np.array([[-1], [-0.9], [-0.8], [-0.7], [-0.6], [-0.5], [-0.4], [-0.3], [-0.2], [-0.1], [0.0], [0.1], [0.2],[0.3],[0.4],[0.5],[0.6],[0.7], [0.8], [0.9], [1.0]]))[0]
# y = denormal(y, m ,d)
# fig, ax = plt.subplots()
# ax.scatter(x, y, alpha=0.5)
# x = np.linspace(-1, 1, 101)
# ax.plot(x, np.power(x, 3), 'k', label='True Function')
# plt.show()


# In[360]:


mat = scipy.io.loadmat('hw2_data.mat')
X1 = pd.DataFrame(mat['X1'])
Y1 = pd.DataFrame(mat['Y1'])
X2 = pd.DataFrame(mat['X2'])
Y2 = pd.DataFrame(mat['Y2'])
Y1 = Y1/255
Y2 = Y2/255
X1[0] = X1[0]/100
X1[1] = X1[1]/76
X2[0] = X2[0]/133
X2[1] = X2[1]/140
implot = plt.imshow(np.array(Y2).reshape(133,140,3), aspect='auto')
plt.savefig("out.png")


# In[369]:


batch_size = 16
batch_num = Y2.shape[0]//batch_size
nn = Network(2, 3, batch_size)
nn.add(2, 32, batch_size)
nn.add(32, 64, batch_size)
nn.add(64, 128, batch_size)
nn.add(128, 256, batch_size)
nn.add(256, 3, batch_size)
train(nn, X2, Y2, 3000, batch_num, 1E-3,opt_methods='Adam')
y2 = nn.predict(X2).T
y2 = y2.reshape(133, 140, 3)
implot = plt.imshow(y2, aspect='auto')
plt.savefig("out1.png")


# In[373]:


batch_size = 32
batch_num = Y2.shape[0]//batch_size
nn = Network(2, 3, batch_size)
nn.add(2, 64, batch_size)
nn.add(128, 128, batch_size)
nn.add(128, 64, batch_size)
nn.add(64, 3, batch_size)
train(nn, X2, Y2, 1200, batch_num, 1E2,opt_methods='Adam')
y2 = nn.predict(X2).T
y2 = y2.reshape(133, 140, 3)
implot = plt.imshow(y2, aspect='auto')
plt.savefig("out2.png")


# In[378]:


# batch_size = 32
# batch_num = Y1.shape[0]//batch_size
# nn = Network(2, 3, batch_size)
# nn.add(2, 256, batch_size)
# nn.add(256, 128, batch_size)
# nn.add(128, 64, batch_size)
# nn.add(64, 1, batch_size)
# train(nn, X1, Y1, 2000, batch_num, 1E-3,opt_methods='Adam')
y1 = nn.predict(X1).T
y1 = y1.reshape(100, 19, 4)
implot = plt.imshow(y1, aspect='auto')
plt.savefig("out3.png")


# In[ ]:


batch_size = 16
batch_num = Y1.shape[0]//batch_size
nn = Network(2, 3, batch_size)
nn.add(2, 256, batch_size)
nn.add(256, 128, batch_size)
nn.add(128, 1, batch_size)
train(nn, X1, Y1, 2000, batch_num, 1E-3,opt_methods='Adam')
y1 = nn.predict(X1).T
y1 = y1.reshape(133, 140, 3)
implot = plt.imshow(y1, aspect='auto')
plt.savefig("out4.png")

