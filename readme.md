# Experimenting with Initialization
This is an experiment on weight initialization. I noticed that PyTorch doesn't actually default initialize linear layers to Xavier or He initialization, which seemed pretty weird to me. Here's a plot of the weights for a model with each of the three strategies.

![histogram](weight_initialization_histogram_2023_12_15.png)

Here's the train/test accuracy for the different initializations I tried.


![train accuracy](Train_Accuracy.png)

![test accuracy](Test_Accuracy.png)