import time

import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from tensorflow import keras
import pandas as pd
import scprep
import time

class scIWNN_impute():
  def __init__(self, file,nums_iter=200):
    self.file = file
    self.nums_train = nums_iter

  def process_file(self):
    print("reading...")
    expression_data = pd.read_csv(self.file,index_col=0,header=0).T
    cell_idx = expression_data.index
    gene_idx = expression_data.T.index
    expression_data = scprep.normalize.library_size_normalize(expression_data)
    expression_data = np.log2(1+expression_data)
    MAX = expression_data.max().max()
    expression_data = expression_data/MAX
    print("The data contains {} cells and {} genes.".format(expression_data.shape[0], expression_data.shape[1]))
    train_ds = tf.data.Dataset.from_tensor_slices(expression_data).batch(expression_data.shape[0])
    class add(tf.keras.layers.Layer):
      def __init__(self):
        super().__init__()

      def build(self,input_shape):
        self.k1 = tf.Variable(tf.zeros([input_shape[0],3]),trainable=True)
        self.k2 = tf.Variable(tf.zeros([input_shape[0],6]),trainable=True)
        self.k3 = tf.Variable(tf.zeros([input_shape[0],9]),trainable=True)
        self.k4 = tf.Variable(tf.zeros([input_shape[0],12]),trainable=True)
        self.dense1 = tf.keras.layers.Dense(input_shape[1],activation='tanh')
        self.dense2 = tf.keras.layers.Dense(input_shape[1],activation='tanh')
        self.dense3 = tf.keras.layers.Dense(input_shape[1],activation='tanh')
        self.dense4 = tf.keras.layers.Dense(input_shape[1],activation='tanh')
        self.alpha = tf.Variable(tf.ones(1),trainable=True)
        self.beta = tf.Variable(tf.ones(1),trainable=True)
        self.gamma = tf.Variable(tf.ones(1),trainable=True)
        self.kaga = tf.Variable(tf.ones(1),trainable=True)
        self.l = input_shape[0]
        self.r = input_shape[1]

      def call(self,inputs,mask):
        x1 = self.dense1(self.k1)
        x2 = self.dense2(self.k2)
        x3 = self.dense3(self.k3)
        x4 = self.dense4(self.k4)
        x = tf.math.add_n([self.alpha*x1,self.beta*x2,self.gamma*x3,self.kaga*x4])
        x = tf.math.multiply(x,mask)
        x = x+inputs
        return x

    inputs = tf.keras.Input(shape=(expression_data.shape[1]),batch_size=expression_data.shape[0])
    mask = tf.where(inputs>0,0.0,1.0)
    X =  add()(inputs,mask)
    model=tf.keras.Model(inputs=inputs,outputs=X)
    optimizer = tfa.optimizers.AdamW(
            learning_rate=0.02, weight_decay=0.000
        )
    train_loss = keras.metrics.Mean(name='train_loss')

    @tf.function
    def train_step(data):
        with tf.GradientTape() as tape:
          data = tf.cast(data,'float64')
          predictions = model(data, training=True)
          u=tf.linalg.svd(tf.cast(predictions,'float64'),compute_uv=False)
          loss = tf.cast(tf.math.reduce_sum(u)/expression_data.shape[0],'float64')
          loss = loss*loss

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    # 循环训练模型
    for epoch in range(self.nums_train):
      train_step(expression_data)

      template = '=> Epoch {}, Loss: {:.4}'
      if(epoch%5==0):
        print(template.format(epoch+1,
                                  train_loss.result()
                                  ))
      train_loss.reset_states()

    s=model.predict(train_ds.take(1))*MAX
    s=np.where(s<0,0,s)
    s=np.power(2,s)-1
    print("Writing to file in /example_data/scIWNN_impute.csv...")
    pd.DataFrame(s.T,index=gene_idx,columns=cell_idx).to_csv("./example_data/scIWNN_impute.csv")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    scIWNN_impute_obj = scIWNN_impute("./example_data/observe.csv",30)
    scIWNN_impute_obj.process_file()

