# IFT6266
code for the course IFT6266 on deep learning @ UdeM
The full description of the project can be found here: https://ift6266h17.wordpress.com/project-description/

The files used to run the experiments in the Hades nodes are:

- myHadesExp1.pbs

- myHadesExp2.pbs

- myHadesExp3.pbs

The scripts for the three experiments are:

- CaeMainOM_Exp1.py, which uses Cae.py to build the neural network.

- CaeMainOM_Exp2.py, which uses Cae_BigDense.py to build the neural network.

- CaeMainOM_Exp2.py, which uses Cae_BigDenseLeaky.py to build the neural network.

Code uses modified versions of some code from https://github.com/Pabrousseau/ift6266h17/ and https://github.com/Dutil/IFT6266/ .

A short description of the experiments follows:

- Experiment 1: a basic convolutional auto-encoder to generate the in-painting of an image based on the context of the image.

- Experiment 2: introduces changes to avoid losing information from the context, by ommitting Pooling and expanding the bottleneck Dense layer.

- Experiment 3: Maximization of the standard deviation of the pixel values of the in-painting generated, as well as Leaky ReLU's are introduced to enhance the quality of the images.

Note: The early stopping is not implemented yet in the experiments, as the are problems just running the basic experiments in Hades, because of some problems with Theano. Small runs work well with small mini-batches, and small training and validation sets, but the small size makes it only useful as a toy to debug. This is a work in process! 

TO DO:
1 - Find out why Theano does not work in my virtual environment in Hades.
2 - Find out why the GPU cannot be used.
3 - After these two problems are fixed, full-size runs will be tried, and the code for showing the images, as well as early-stopping will be added. For now the priority is to have the toy problems work.
4 - Implement Experiment 4, with Dropout for the Dense layer, augmenting the size of it.
5 - Plug in all the units the output of the last hidden-state of an LSTM trained over the captions, for the task of predicting the next word (word embedding from wikipedia, vectors of size 50).
6 - Try training another network to predict which words are nouns, and concatenate the first noun embedding to the hidden-state before connecting it to the units in the auto-encoder.

