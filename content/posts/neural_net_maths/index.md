+++
date = '2025-09-18T12:15:23+01:00'
draft = true
title = 'A mathematical introduction to feedforward neural networks'
+++
The introduction of feedforward neural networks and their application to early problems such as recognising handwritten digits from the MNIST dataset was a landmark achievement. Despite their limited standalone use today, they still remain the backbone of various models of wide range of modern architectures Incredibly, the backpropagation algorithm, which only started gaining traction in the 1980's, is still used when updating the billions of trainable parameters in the massive LLMs we see today.

This post aims to give a mathematical description of what a feedforward neural network is, and how it can be trained. It assumes some basic knowledge in single and multivariable calculus, including the chain rule for partial derivatives, and some basic knowledge of matrices.

***
## Defining the problem

We first need to know what this network is even meant to do! For now, we will focus on the classic task of recognising handwritten greyscale digits as an example.

More generally, we focus on tasks which involve **classification**. Given some inputs, we want to be able to classify them into different categories based on their attributes.

The images come from a dataset called MNIST. Without going into any of its history, it is a dataset consisting of handwritten digits from 0-9 on a 28x28 grid, each image coming with a label telling us which number the pixels show. The images and labels are split into 2 separate datasets, the first to be used for training the model, and the second to be used for testing how well the model can classify drawings of digits it has never seen before. 

The images are represented as a 28x28 grid of numbers between 0 and 255 inclusive, where each number refers to the intensity of a particular pixel on the grid.

The examples below shows how the pixel values match their corresponding intensities:

![Placeholder](avatar.jpg)

Before we see how a neural network attempts to 'solve' this classification problem, it is important to define what the neural network even is.

***
## The neuron

The neuron is the most fundamental "building block" of a neural network. It takes some inputs, and maps them to a single output, which is somehow meant to "encode" some sort of information about the inputs. 

Each neuron has associated with it $n$ weights, which we denote $w_1,...,w_n$ for now, where $n$ is the number of inputs to the neuron. It also has a single bias, $b$. 

We can collectively refer to the weights and biases as parameters. When we are training a neural network, we are actually just modifying the weights and biases of the many neurons in the network to achieve some sort of desired behaviour when we feed inputs into the network.

We define the function $\sigma(z)$ as follows:  
$$\sigma(z)=\frac{1}{1+e^{-z}}$$

Given some $n$ inputs $x_1,...,x_n$, the output $a$ of the neuron is:  
$$a=\sigma(w_1x_1+...+w_nx_n+b)$$

Or more compactly:  
$$a=\sigma\left(\sum{w_ix_i}+b\right)$$

where the summand is taken to be over all the inputs.

The visual representation below is commonly used:

![Placeholder](avatar.jpg)

With networks containing loads of these neurons, writing out the weights on each individual connection will clutter up the image, so they are usually omitted. This representation allows us to give a nice graph-like representation of the data flowing through a neural network when we actually start constructing them.

Immediately, there is quite a bit to break down!

WIthout worrying about the $\sigma$ for the moment, the inside sum is a weighted sum of the inputs $x_i$, with the various $w_i$. The bias term is then added on, leaving us with what we call the **weighted input** to the neuron:

$z=\sum{w_ix_i} + b$

This "intermediate" quantity turns out to be incredibly useful when we start training the network, but more on that later...

After that, we apply the function $\sigma$ to this weighted input, leaving us with the final output of the neuron, $a=\sigma(z)$. 

In this context, the sigmoid function acts as an **activation function**. There are many other possibilities for the activation function, such as hyperbolic tangent or ReLU, which have generally seen better results in practice, but for now, we will just stick with the sigmoid.

The function $\sigma(z)$ is known as the sigmoid function. There are a few useful and important properties we can deduce about it. It may help to look at a graph sketch of the output of the sigmoid vs its input, which is shown below: