''' 进展：
        按模块运行不会bug
        损失函数选择二进制交叉熵效果比mae好
        明日任务 ： 优化自编码器 能适应脉冲'''
        
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import ndarray
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
def moving_avg_filter(series,core_length,mode="same"):
    ''' 通信数据使用长度3-6的卷积核效果都还不错'''
    return(np.convolve(series, np.ones((core_length,))/core_length, mode=mode))

def dict_to_array(dict_log:dict)->ndarray:

    ''' 待优化
        用list append执行很快 np.concatenate慢十倍以上'''
    i = 0
    print_key = 10000

    dict_len = len(dict_log)
    
    for k,v in dict_log.items():
        if len(v)>=415:
            v = v[:414]
            data = np.array(v)
            upload = data[:,0]#.tolist()
            download = data[:,1]#.tolist()
            upload = moving_avg_filter(upload,5)
            download = moving_avg_filter(download,5)
            try:
                up.append(upload)
            except:
                up = []
                up.append(upload)
            try:
                down.append(download)
            except:
                down = []
                down.append(download)
            i+=1
            if (i%print_key)==0:
                print('already to array ',i,' areas.')
    u = up #np.array(up)
    d = down #np.array(down)
    data_set = {'up':u,'down':d}
    

    return data_set
def data_prepare(data):
    '''return train_data,test_data'''
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=21)

    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)

    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    return train_data,test_data

dict_log_1 = json.load(open('processed_log_1.json'))
data_set = dict_to_array(dict_log_1)
up_data = data_set['up']
train_data,test_data = data_prepare(up_data)
#model
'''
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(414, activation="relu"),
     # layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
     # layers.Dense(32, activation="relu"),
      layers.Dense(414, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
'''#model 2

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(414,   activation="relu")
      ,layers.Dense(16*8, activation="relu")
      ,layers.Dense(8*8,  activation="relu")
      ,layers.Dense(8*4,  activation="relu")
      ,layers.Dense(8,    activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(8,    activation="relu"),
      layers.Dense(8*4,  activation="relu"),
      layers.Dense(8*8,  activation="relu"),
      layers.Dense(16*8, activation="relu"),
      layers.Dense(414,  activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


autoencoder = AnomalyDetector()

autoencoder = AnomalyDetector()
#model end
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history = autoencoder.fit(train_data, train_data, 
          epochs=20, 
          batch_size=512,
          validation_data=(test_data, test_data),
          shuffle=True)
encoded_imgs = autoencoder.encoder(test_data).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
'''
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()'''

image_ = 500
plt.plot(test_data[image_],'b')
plt.plot(decoded_imgs[image_],'r')
plt.fill_between(np.arange(414), decoded_imgs[image_], test_data[image_], color='y' )
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()



























