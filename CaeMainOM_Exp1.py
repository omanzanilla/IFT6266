# -*- coding: utf-8 -*-
from __future__ import division

"""
Éditeur de Spyder

Ceci est un script temporaire.

Author: Orestes Manzanilla

Code is heavily based on the 
works of Pabrousseau https://github.com/Pabrousseau/ift6266h17/
and of Dutil https://github.com/Dutil/IFT6266/ (from which the
idea of  maximizing the standard deviation of the output is taken from. However,
in this experiment this element is ommited from the optimization process).

This code uses trains a network built via "Cae.py" and trains it to generate the inpainting
of the images of the project https://ift6266h17.wordpress.com/project-description/

Description: This is a modification of the basic training code by Pabrousseau
with the following modifications:
    * limiting the number of images considered (For computational reasons)
    * objective function is modified to allow the use of STD maximization.
    * The number of batches, learning rate and epochs were adjusted.
    
Note: This is considered Experiment 1.      

Note2: The early stopping is not implemented yet.
"""  




import sys
import os
import time
import numpy as np

import theano
import theano.tensor as T

import lasagne
#import lasagne.layers.dnn

#import Cae_BigDense
import Cae

from common import im2ar, ar2im, np_ar2im, load_dataset_mscoco
from common import save_jpg_results, create_html_results_page
    
###################

# Batch iterator
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    #print(" COMMENTARY: number of inputs and targets = "+ str(len(targets)))
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

################


#def PrintLasagneNetInfo(lasagneNetwork): 
#    import nolearn
#    from nolearn.lasagne import NeuralNet
#    nolearnNetwork = NeuralNet( 
#          layers=lasagneNetwork, 
#          update=lasagne.updates.adam, 
#          ) 
#    nolearnNetwork.initialize() 
#    PrintLayerInfo(nolearnNetwork)() 

#def PrintLasagneNetInfo(lasagneNetwork): 
#    import nolearn
#    nolearnNetwork = nolearn.lasagne.NeuralNet( 
#          layers=lasagneNetwork, 
#          update=lasagne.updates.adam, 
#          objective_loss_function= lasagne.objectives.squared_error
#          ) 
#    nolearnNetwork.initialize() 
#    nolearn.lasagne.PrintLayerInfo()(nolearnNetwork)
    
############################################

# Main
def main():
    
    # Hyper Params
    num_epochs = 12 #was 100
    learning_rate = 0.001
    batchsize = 100 #Was 200
    
    #Variance of the prediction can be maximized to obtain sharper images.
    #If this coefficient is set to "0", the loss is just the L2 loss.
    StdevCoef = 0
    
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test, X_original_test = load_dataset_mscoco()
    #print("length of y_train = " + str(y_train))
    #print("length of y_val = "+ str(y_val))
    #print("length of y_test = "+str(y_test))
    
    # Prepare Theano variables for inputs and targets
    inputVar = T.tensor4('inputs')
    target_var = T.tensor4('targets')

    # Build Network
    print("Building model and compiling functions...")
    #network = Cae_BigDense.build(inputVar)
    network = Cae.build(inputVar)
    
    #Print the info about the network
    #PrintLasagneNetInfo(network)

    # Training Loss expression
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    
    # Here the objective function takes into account the maximization of the mean STD
    loss = loss.mean() - StdevCoef * theano.tensor.std(prediction, axis=(1, 2, 3)).mean()
    
    # Add regularization lasagne.regularization.

    # Update expressions 
    # Stochastic Gradient Descent (SGD) with Nesterov momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    #updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate, momentum)
    updates = lasagne.updates.adam(loss, params, learning_rate)

    # Test Loss expression
    # 'Deterministic = True' disables droupout
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction,target_var)
    test_loss = test_loss.mean()

    # Train Function
    train_fn = theano.function([inputVar, target_var], loss, updates=updates)

    # Test Function
    val_fn = theano.function([inputVar, target_var], test_loss)

    # Predict function
    predict = theano.function([inputVar], test_prediction)
    
    # Training Loop
    print("Starting training...")
    # We iterate over epochs
    for epoch in range(num_epochs):
        # Full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batchsize, shuffle=True):
            #print("...running train_batches calculation...")
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # Full pass over the validation data
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batchsize, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1

        # Print the results for this epoch
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        #print("  COMMENTARY: train_batches = "+str(train_batches))
        if train_batches == 0:
            print("train_batches is 0, can't compute training loss")
        else:
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        if val_batches == 0:
            print("val_batches is 0, can't compute validation loss")
        else:
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

    # Print the test error
    test_err = 0
    test_batches = 0
    preds = np.zeros((X_test.shape[0], 3, 32, 32))
    for batch in iterate_minibatches(X_test, y_test, batchsize, shuffle=False):
        inputs, targets = batch
        preds[test_batches*batchsize:(test_batches+1)*batchsize] = predict(inputs)
        err = val_fn(inputs, targets)
        test_err += err
        test_batches += 1
    print("Final results:")
    if test_batches == 0:
        print("test_batches is 0, can't compute test loss.")
    else:
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

    # Save model
    np.savez('caeOM-exp1.npz', *lasagne.layers.get_all_param_values(network))

    # Save predictions and create HTML page to visualize them
    save_jpg_results("assets_exp1/", preds, X_test, y_test, X_original_test)
    create_html_results_page("results_exp1.html", "assets_exp1/", preds.shape[0])

###########################

if __name__ == '__main__':
        main()
        
