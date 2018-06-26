# VAE-Tensorflow

## Introduction
In this exercise, I will construct a Variational Autoencoder (VAE) for image reconstruction and images generation by using the same 
animation face dataset.

## Detail
I resize the input image to 28x28x3 in order to reduce the use of memory and make training faster. I use the convolution layers as 
encoder and the deconvolution layers as decoder. In the training progress I find the smaller the latent size, the lower diversity of 
generated image and the lower equality of the image. 

## Structure
<img src="https://github.com/ChristophWu/VAE-Tensorflow/blob/master/material/structure.png" width="600"/>

## Implementation
- utils.py is used to merge pictures into one picture
- dataset.py is used to cut the dataset into minibatches
- open terminal and python vae.py to implement

## Result
- learning curve
<img src="https://github.com/ChristophWu/VAE-Tensorflow/blob/master/material/learning_curve.png" width="500"/>

- some examples of reconstructed images
<img src="https://github.com/ChristophWu/VAE-Tensorflow/blob/master/material/reconstruction_images.jpg" width="500"/>

- some examples of generated images
<img src="https://github.com/ChristophWu/VAE-Tensorflow/blob/master/material/generated_images.jpg" width="500"/>

