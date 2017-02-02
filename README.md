![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)
# Udacity Self-Driving Car Nanodegree
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
  - 2D convolutional layer: 6x6 filter with 16 features, "same" borders, and ReLU activation
  - 3x3 average pooling layer
  - 2D convolutional layer: 6x6 filter with 16 features, "same" borders, and ReLU activation
  - 2x2 average pooling layer
  - 2D convolutional layer: 6x6 filter with 16 features, "same" borders, and ReLU activation
  - 2x2 average pooling layer
  - Flatten layer
  - Dropout layer (50%)
  - Fully connected layer: 1024 features with ReLU activation
  - Dropout layer (50%)
  - Fully connected layer: 1 feature with tanh activation
  
#### Training the model
  - The model uses as Adam optimizer with a learning rate of 1e-4
  - The model uses Mean Squared Error as its loss function
  - 20% of the data was used in the validation set
  - The model was trained for 5 epochs
  
### Behavioral Cloning
  - The sample data provided was a good starting off point, but more data was needed to ensure the car could get around the whole track. This was especially true for tight turns.
  - The most efficient way to train the car to stay on track was to "bounce" it off the edges of the track - meaning, to steer toward the edge (while not recording), and then record the car sharply turning away from the edge.
  - Some other problem areas were addressed by moving the car off the center line of the track (while not recording), and then record gently returning to the center line

## Solution Explanation

There were many mis-starts to this project, as many ideas that seemed worth pursuing in theory turned out to not be worthwhile in practice. My first idea was to use feature extraction with one of the ImageNet architectures like ResNet or InceptionV3 (both of which come prepackaged with Keras). But they proved too large and unwieldy for the task at hand - training them took too much time and used too much memory.

There were also plenty of false starts with image preprocessing - I tried techniques like cropping, scaling, applying a Gaussian blur, Laplacian gradient, or Canny transform, converting to YUV, and region masking. I even tried doing a 3-channel Canny transform, where I did individual Canny transforms on each color channel and then combined them back into one (very pretty) image. However, most of these ideas didn't result in better performance, and were scrapped.

I eventually fell back on cropping off the top 50 pixels of each image (taking out unimportant noise above the road), and scaling images to a factor of 0.5 (which speed up training while maintaining accuracy). I also normalized the all the channel data to a range of -1 to 1.

  ![Processed Images](https://raw.githubusercontent.com/timmyneutron/self_driving_car_P3_steering_angle/master/processed_images.png)


I also augmented the data by copying all of the "turn" images (images where the steering angle was not equal to 0), flipping the copies horizontally, and negating their steering angles. The rationale behind this was that any turn in one direction should correspond to a turn in the other direction that was equal in magnitude and opposite in direction.

To build the model, I started simple. I knew that it would likely begin with a 2D convolution, and it need to end with a single output that ranged from -1 to 1. So the base model I began with was two 2D convolutions followed by a single fully-connected layer with only one node, with a tanh activation so that the output was always between -1 and 1. And since the output was continuous, it made sense that the loss function should be mean squared error.

I trained this base model on only 100 samples to see if it would overfit (since a large enough model should be able to overfit a small amount of data). Training this model did show overfitting - the training error and validation error would decrease for about 5 epochs, and then over the next five epochs the training error would continue to decrease while the validation error started to increase.

I then tried to systematically increase the size of the model, and modify its hyperparameters, and test its performance on the track with each change (recording each of these intermediate models and their performance). I added another convolution layer and a fully connected layer, and found that using dropout on the fully connected layer was an effective way to mitigate "noisy" steering (steering back and forth over the center line). I was able to get the car pretty far without needing additional data, but eventually needed to supplement the provided data with my own.

I noted that the failure mode for the car on the track was always underturning (ie, not turning enough) rather than overturning (turning too much). This made sense given the distribution of the training data - roughly half of the training labels are 0, and very few were high in magnitude.

<img src="https://raw.githubusercontent.com/timmyneutron/self_driving_car_P3_steering_angle/master/hist.png" alt="Steering Angle Histogram" style="display: block; margin: auto; width: 40%;"/>

So the model was biased toward underturning. The most efficient way to address this was to "bounce" the car off the outside edges of the turns - to steer toward the edge (while not recording), and the record the car sharply turning away from the edge. I did this at different points of the turn, in both directions of the track, and it didn't take too many extra training examples to keep the car on the track at all times.

Once the car would stay on the track, I added some "correction" data - moving the car off the center line and then recording the car gently returning to the center line. This helped smooth out some of the noise in the steering, and correct for when the car was off center.

I then tested the model on the second track, and found that it worked fairly well - it was able to follow the curves in the second track about as well as it did the first.
