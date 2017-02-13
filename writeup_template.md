#**Behavioral Cloning** 

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image_arch]: ./images/arch.jpg "Model Architecture"
[image_sthw]: ./images/steering.jpg "Steering hardware"
[image_center_lane]: ./examples/placeholder.png "Center lane image"
[image_center_lane_flipped] ./examples/placeholder.png "Flipped image"
[image_left_cam]: ./examples/placeholder.png "Grayscaling"
[image_right_cam]: ./examples/placeholder.png "Grayscaling"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

I have used a slightly modified NVIDIA neural network. 

My model consists of 3 convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 29-35). ReLU activion is used for each convolution. 

After first convolution layer I have added a max-pooling layer. 

Then there is a flatten layer followed by a dropout layer with threshold 0.1. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layer in order to reduce overfitting (model.py line 38). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 116). 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 122).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I have built a 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use neural network model similar to the NVIDIA neural network from their end to end driving paper. They already proved that the neural network works so I wanted to give it a try.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
Without a ddropout layer after flattening the car will go out of the lane (which could be confirmed from the mean squared error of training set and test sets - but I have not looked at it - I just added a dropout layer after finding it can not drive in the simulator - a sign of overfitting).

The final step was to run the simulator to see how well the car was driving around track one. I have trainned the network for about 12 hours (20 epocs) and found that the network drives the first track perfectly.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

I kept the simulator running overnight and found the car is still running in the morning. One thing I have noted is that the machine needs to be reserved for the simulator and the drive.py program. I was watching a video once and found that the car went for swimimg in the lake. 

I have uploaded a 10 minute video of the model driving here- https://www.youtube.com/watch?v=AZB16e5DN58

####2. Final Model Architecture

The final model architecture (model.py lines 16-46) consisted of a convolution neural network with the following layers and layer sizes:

- Input shape: (160, 320, 3)

- Normalization layer: Output shape: 160, 320, 3
- Cropping layer: Output shape (65, 320, 3)
- Convolution layer (5x5): Output shape (61, 316, 24)
- MaxPool layer (2x2): Output shape (30, 158, 24)
- Convolution layer (5x5): Output shape (26, 154, 36)
- Convolution layer (5x5): Output shape (22, 150, 48)
- Convolution layer (3x3): Output shape (20, 148, 64)
- Convolution layer (3x3): Output shape (18, 146, 64)

- Flatten layer: Output shape (168192)
- Dropout : Output shape (168192)

- Fully connected : Output shape (256)
- Fully connected : Output shape (100)
- Fully connected : Output shape (50)
- Fully connected : Output shape (10)

- Output layer: shape (1)

Total trainable params: 43,220,027

Here is a visualization of the architecture (without max-pooling and dropout layer)

![Model architectire] [image_arch]

####3. Creation of the Training Set & Training Process

To collect data I have created a simple hardware using an old stepper motor. I have removed all the circuitry from the motor keeping the gear mechanism and the potentiometer. I have then connected the potentiomenet to an arduino and used the drive.py to collect the input data. Here is the details of the steering hardware http://www.copotron.com/2017/01/driving-udacity-car-simulator-with-home.html . With this I was able to collect a lot of data. 

To start I have taken the data.zip file provided with the assignment. Initially I used only the center lane. Here is an example image of center lane driving:

![Center lane data][image_center_lane]

The data file contains many different positions. I have taken the left and right images with a steering correction value of 0.2. Correction value is added to to steering value to get left image steering value and subtracted to get right image steering value. 

![Left camera][image_left_cam]
![Right camera][image_right_cam]

To augment the data sat, I also flipped images and angles that gives me image with similar angle but taken from mirrored lane. For example, here is an image that has then been flipped:

![alt text][image_center_lane]
![alt text][image_center_lane_flipped]


After the collection process, I had 48216 number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I started with 3 epochs that would crash in few seconds. The ideal number of epochs was 20 as the model could drive perfectly for more than 8 hours running overnight.

I used an adam optimizer so that manually training the learning rate wasn't necessary.
