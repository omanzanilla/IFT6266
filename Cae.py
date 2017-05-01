
# coding: utf-8

# In[ ]:

    
"""
Éditeur de Spyder

Ceci est un script temporaire.

Author: Orestes Manzanilla

Code is heavily based on the first work of
Pabrousseau in here: https://github.com/Pabrousseau/ift6266h17/tree/master/week1

This code builds a basic contextual auto-encoder to generate the inpainting
of the images of the project https://ift6266h17.wordpress.com/project-description/

Description:
    Classical Convolutions and Pooling layers until a dense fully connected
    smaller layer will store a compressed version of the patterns as encoding.
    The decoder consists of upsampling and convolutions up to the desired size
    to fit the center of the images.
    
Note: This is the "build" function used in Experiment 1 (baseline)
"""

def build(inputVar):
    import lasagne
    #from lasagne.layers import Upscale2DLayer #Added
    
    # Hyper Params
    channels = 24
    filterSize = (5,5)
    poolSize = (2,2)
    bottleneckSize = 512 
    encode = 256
    nonlin = lasagne.nonlinearities.rectify
    
    # Input 3*64*64 
    network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64), input_var=inputVar)
       
    # Conv 24*60*60
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=channels, 
            filter_size=filterSize,
            nonlinearity=nonlin,
            W=lasagne.init.GlorotUniform(),
            pad = 'valid')
    
    # Conv 24*56*56
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=channels, 
            filter_size=filterSize,
            nonlinearity=nonlin,
            pad = 'valid')

    # MaxPool 48*28*28
    network = lasagne.layers.MaxPool2DLayer(network, 
            pool_size=poolSize)
    
    # Conv 48*24*24
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=2*channels, 
            filter_size=filterSize,
            nonlinearity=nonlin,
            pad = 'valid')

    # MaxPool 48*12*12
    network = lasagne.layers.MaxPool2DLayer(network, 
            pool_size=poolSize)
    
    # Reshape 6912 
    network = lasagne.layers.ReshapeLayer(network,
            shape=(([0], -1)))
    
    # Dense 512
    network = lasagne.layers.DenseLayer(network,
            num_units=bottleneckSize,
            nonlinearity=nonlin)
    
    # Dense 256
    network = lasagne.layers.DenseLayer(network,
            num_units=encode,
            nonlinearity=nonlin)
    
    # Dense 512
    network = lasagne.layers.DenseLayer(network,
            num_units=bottleneckSize,
            nonlinearity=nonlin)
    
    # Dense 6912
    network = lasagne.layers.DenseLayer(network,
            num_units=6912,
            nonlinearity=nonlin)
    
    # Reshape 48*12*12
    network = lasagne.layers.ReshapeLayer(network,
            shape=(([0], 2*channels, 12, 12)))

    # Upscale 48*24*24
    network = lasagne.layers.Upscale2DLayer(network,
            scale_factor=poolSize)
    
    # Conv 24*28*28
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=channels, 
            filter_size=filterSize,
            nonlinearity=nonlin,
            pad = 'full')
    
    # Conv 3*32*32
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=3, 
            filter_size=filterSize,
            nonlinearity=nonlin,
            pad = 'full')
    
    # Reshape
    network = lasagne.layers.ReshapeLayer(network,
            shape=(-1,3,32,32))
    
    return network

