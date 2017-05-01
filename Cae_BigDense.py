
# coding: utf-8

# In[ ]:
"""
Éditeur de Spyder

Ceci est un script temporaire.

Author: Orestes Manzanilla

Code is heavily based on the first work of
Pabrousseau in here: https://github.com/Pabrousseau/ift6266h17/
and of Dutil https://github.com/Dutil/IFT6266/ (from which the
decoder is taken from, to substitute the upsampling-convolution from Experiment 1).

This code builds over the basic contextual auto-encoder to generate the inpainting
of the images of the project https://ift6266h17.wordpress.com/project-description/
by eliminating features that intrinsically "lose information", like a small bottleneck
and pooling units.

Description: This is a modification of the file Cae.py consisting of:
    * Pooling units are replaced with more convolutions.
    * Transposed Convolutions are used instead of up-sampling in the decoder.
    * After the reshaping at the end of the encoder, only one dense layer is used
      and is bigger than the original. It was intended be bigger, but for 
      computational limitations decided to keep it in 864 units.
      
Note: This is the "build" function used in Experiment 2.      
"""  
    
def build(inputVar):
    import lasagne
    from lasagne.layers import Upscale2DLayer #Added
    
    # Hyper Params
    channels = 24
    filterSize = (5,5)
    poolSize = (2,2)
    
    bottleneckSize = 864
    nonlin = lasagne.nonlinearities.rectify
    #nonlin = lasagne.nonlinearities.leaky_rectify
    
    # Input 3*64*64 
    network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64), input_var=inputVar)
       
    # Conv channels*60*60
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=channels, 
            filter_size=filterSize,
            nonlinearity=nonlin,
            W=lasagne.init.GlorotUniform(),
            pad = 'valid')
    
    # Conv channels*56*56
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=channels, 
            filter_size=filterSize,
            nonlinearity=nonlin,
            pad = 'valid')

    # Another Convolution with same size (instead of pooling)
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=channels, 
            filter_size=filterSize,
            nonlinearity=nonlin,
            pad = 'same')
    
    # Conv 48*24*24
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=2*channels, 
            filter_size=filterSize,
            nonlinearity=nonlin,
            pad = 'valid')

    # Another Convolution, (instead of pooling)
    network = lasagne.layers.Conv2DLayer(network, 
            num_filters=2*channels, 
            filter_size=filterSize,
            nonlinearity=nonlin,
            pad = 'valid')
    
    # Reshape 
    network = lasagne.layers.ReshapeLayer(network,
            shape=(([0], -1)))
    
 
    # Dense Bottleneck 
    #network = lasagne.layers.DenseLayer(network,
    #        num_units=bottleneckSize,
    #        nonlinearity=nonlin)
    
    
    # Dense 
    network = lasagne.layers.DenseLayer(network,
            num_units=bottleneckSize,
            nonlinearity=nonlin)

    # Reshape   
    network = lasagne.layers.ReshapeLayer(network,
            (inputVar.shape[0], channels, 6, 6))

    # 1st Deconv
    network = lasagne.layers.TransposedConv2DLayer(network,
            num_filters=channels,
            filter_size=(16, 16),
            stride=(1,1),
            nonlinearity=nonlin)
    
    # 2nd Deconv
    network = lasagne.layers.TransposedConv2DLayer(network,
            num_filters=channels,
            filter_size=(8, 8),
            stride=(1,1),
            nonlinearity=nonlin,)
    
    # 3rd Deconv
    network = lasagne.layers.TransposedConv2DLayer(network, 
            num_filters=channels, 
            filter_size=(3, 3), 
            stride=(1,1),
            nonlinearity=nonlin)
    
    # 4th Deconv (and last!)
    network = lasagne.layers.TransposedConv2DLayer(network, 
            num_filters=3, 
            filter_size=(3, 3), 
            stride=(1,1), 
            nonlinearity=lasagne.nonlinearities.sigmoid)
            
    # Reshape
    network = lasagne.layers.ReshapeLayer(network,
            shape=(-1,3,32,32))
    
    
    
    return network

