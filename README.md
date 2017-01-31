# Udacity Self-Driving Car Nanodegree
![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)
# Project #3: Behavioral Cloning (Steering Angle)

## Introduction
This is a project for Udacity's Self-Driving Car Nanodegree. It's a convolutional neural network that uses pictures from a forward-facing camera 
to predict steering angle for video-game car on a track. It uses behavioral cloning to approximate human judgement.

## Concepts and Classes
Concepts explored in this project:

  - Behavioral cloning
  - Convolutional neural networks
  - Neural networks, including deep neural networks and convolution neural networks
  - Data pre-processing, including normalization, color channels, and image manipulation using OpenCV and Matplotlib
  - Constructing and training neural networks using Keras
  - Logisitic classifiers, stochastic gradient descent, and back-propagation
  - Anti-overfitting techniques like dropout and L2 regularization
  - Training neural networks using a GPU on Amazon Web Services
  
## Model Description

### Image preprocessing
  - Since the information needed to steer the car is found in the lower portion of the forward-facing images, the top 50 pixels of the images were cropped.
  - Images were resized by a scale factor of .5, which maintained accuaracy while speeding up training
  - RGB channels were normalized to a range of -1 to 1

### Data augmentation
  - Data was augmented by copying images of "turns" (ie, where the steering angle was not equal to 0), flipping these copies horizontally, and negating the steering angle.
  
### Model Architecture
#### The architecture of the model is as follows:
  - 2D convolutional layer: 6x6 
### Training the model
