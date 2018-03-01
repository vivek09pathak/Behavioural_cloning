
# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./WorkFlow.JPG "Model Visualization"
[image2]: ./Images.JPG "Grayscaling"
[image5]: ./model_summary.JPG "Model Summary"
[image7]: ./Mean_Square_model.JPG "Mean_Square_Model"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24-48 for first 3 layer and 3x3 for last two 64 depths layer (model.py lines 115-123) 

The model includes RELU layers to introduce nonlinearity (code line 129), and the data is normalized in the model using a Keras lambda layer (code line 113). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 125). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 75-94). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 148).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Nvidia Architecture I thought this model might be appropriate because to minimize the mean squared error between the steering command output by the network and the command of either the human driver.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that Validation set and Training set does not overtfit by adding MaxPooling with pool size as 2,2 and dropout layer with keep probs as 0.2

Then I trained the data and ran epoch for 5 times as seeing the training loss and validation loss I saved the model as seeing the model had low loss with minimum overfitting between train and validation loss 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I modified the correction angle for steering right and left by subtracting center angle by 0.245 as the correction

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 113-136) consisted of a convolution neural network with the following layers and layer sizes 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   							|
| Lamda Function                      | Normalize image divide by 255.0 and -0.5 mean center |
| Cropping          |  cropping((70,20),(0,0)) |
| Convolution 5x5     	| 2x2 stride,  outputs 24x5x5 |
| RELU					|												|
| Convolution 5x5     	| 2x2 stride,  outputs 32x5x5 |
| RELU					|												|
| Convolution 5x5     	| 2x2 stride,  outputs 48x5x5 |
| RELU					|												|
| Convolution 3x3     	|   outputs 64x3x3 |
| RELU					|												|
| Convolution 3x3     	| outputs 64x3x3 |
| RELU					|												|
| Max pooling	      	| 2x2 stride  				|
| DROPOUT				|	prob=0.2					|
| Flatten                      |
| RELU					|												|
|	Fully connected					|				 100	out							|
|	Fully connected					|				50 out							|
|	Fully connected					|				10 out							|
|	Fully connected					|				1 out							|

Here is the model summary of my CNN

![alt text][image5]

Below is given the visualization of the images

![alt text][image1]:


#### 3. Creation of the Training Set & Training Process

I used Udacity recorded data provided as my Training data. Here is an example image of center lane driving:

I randomly shuffled the data set and put 20% of the data into a validation set saving the data in train_samples and validation_samples. 

To augment the data sat, I also flipped images and angles thinking that this would increase my data to 3 times.
Because without flipping the images were zero centered and due which it was going towards right direction for my data set thus what i did was that I used function cv2.flip to flip it horizontally.Below is the normalized image with 3 camera angle view i.e. Center,Left and right.After flipping car remained on the track for most of the time since I have used CV2 therefore i made changes in drive.py where i converted images to BGR so no conflicts happen for it.

![alt text][image2]:


After the collection process, I had X number of data points. I then  pre-processed this data by using lambda function by normalizing it and bring it to mean center followed by cropping of data.

I finally randomly shuffled the each batch sample before each epoch.


![alt text][image7]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by decrease in loss and graph plotted above between validation loss and training loss but for each training my model was plotting different graph which was difficult to identify the minimum overfitting.I used an adam optimizer so that manually training the learning rate wasn't necessary.
