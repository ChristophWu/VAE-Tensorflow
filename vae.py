import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import dataset
import cv2
from scipy.misc import imsave as ims
from utils import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
tf.reset_default_graph()

loss_array = []
batch_size = 64
img_size = 28
num_channels = 3
train_path='animation'
classes = ['faces']
# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path,img_size, classes)


print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.labels)))

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 3], name='X')
Y    = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 3], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 28 * 28 * 3])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 3
n_latent = 8

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, 28, 28, 3])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd       = 0.5 * tf.layers.dense(x, units=n_latent)            
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
        
        return z, mn, sd
def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_decoder, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28*28*3, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28,3])
        return img
sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)

unreshaped = tf.reshape(dec, [-1, 28*28*3])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    batch = [np.reshape(b, [28, 28,3]) for b in data.next_batch(batch_size=batch_size)[0]]
    sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})

           
    if not i % 800:
        ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
        d = d.reshape(64,28,28,3)
        ims("result/"+str(i//800)+".jpg",merge(d[:64],[8,8]))
        batch = np.reshape(batch,[64,28,28,3])
        ims("result/bob"+str(i//800)+".jpg",merge(batch[:64],[8,8]))
#        for w in range(64):
#            plt.imshow(np.reshape(batch[w], [28, 28,3]))
#            plt.savefig('result/'+str(w)+'_'+str(i)+"fig.jpg")
        #plt.show()
#            plt.imshow(d[w])
        #plt.show()
#            plt.savefig('result/'+str(w)+'_'+str(i)+"d.jpg")
        print(i, ls, np.mean(i_ls), np.mean(d_ls))
    ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
    loss_array.append(ls); 

x = range(len(loss_array))
plt.figure()
plt.xlabel("iteration") 
plt.ylabel("loss") 
plt.title("learning curve")
plt.plot(x,loss_array)
plt.savefig("learningcurve.png",dpi=300,format="png")


randoms = [np.random.normal(0, 1, n_latent) for _ in range(10)]
imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
imgs = [np.reshape(imgs[i], [28, 28,3]) for i in range(len(imgs))]

for img in imgs:
    plt.figure(figsize=(1,1))
    plt.axis('off')
    plt.imshow(img)