# Convulutional Neural Network

### About the dataset

*****

The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, 6000 images per class. The training set is 50000 images and the test set is 10000 images. 

**Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

**Features**: One image contains a vector of 3072 values. This corresponds to 3 values for each pixel. There are three values because these are color images.
Labels: The labels are an integer between 0 and 9, representing the category.

The problem is a supervised mulitnomial classification problem. My goal is to minimize the error rate on the test set.

**One Observation**

![ship image](images/ViewOneObservation.PNG)

### Metrics

*****

I will use accuracy on the test set as my final evaluation metric. I will use loss vs steps to determine a reasonable number of steps to trade off between training time and performance.

### Baseline Models

*****

**Baseline model 1**: an accuracy score for guessing a single class for every observation.

**Baseline model 2**: an accuracy score for using a random number generator to label randomly.

Accuracy on test set for all one class: 10%</br>
Accuracy on test set for random class: 9.69%

### Data Preprocessing

***** 

I started with the preprocessing shown in the Tensorflow Convolutional Neural Network tutorial:
* Images cropped to 24x24 randomly for training and centrally for evaluation
* Images approximately whitenened to make the model insensitive to dyanmic range
* Dataset artificially increased by randomly flipping image from left to right
* Dataset artificially increased by randomly distoring brightness and contrast of images

### About the different layers [I wrote this section for me]

*****

#### Pooling

#### Normalization

#### Dropout

#### Architecture Notes
	- How to connect layers
	- How different parameters impact size, padding, stride, output depth etc.
	- 
### Modeling

*****

#### Simple Model Architecture:
- **Input:**  128x24x24x3 [batch size, width, height, input depth]
- **Convolution Layer:** Out: 124x24x24x12 Relu activation
- **Pool1:** Out 128x12x12x12
- **FC:** Out 128x10
- **Softmax:** Out 128x10

![Simple Arch](images/SimpleArch.png)

Cuda ConvNet Architechture:
- **Input:** 128x24x24x3 
- **Conv Layer:** Out = 128x24x24x64
	- Filters = 64
- **Pool 1:** Out = 128x12x12x64
- **Norm 1:** Out = 128x12x12x64
- **Conv Layer 2:** = Out 128x12x12x64
- **Norm 2:** Out = 128x12x12x64
- **Pool 2:** Out = 128x6x6x64
- **Local 3:** Out = 128x384
- **Local 4:** Out = 128x192
- **Softmax:** Out = 128x10 

![Cuda Net](images/CudaNet.png)

### Entropy and Loss Vs Steps

*****

Graphed the entropy and loss vs steps to find a somewhat low number of steps where performance would be different. I decided if I give a model about 10k steps, I should have a reasonable expectation of what the results will look like. 

**Entropy vs Steps for Cuda Net**
![Entropy Graph](images/EntropyGraph.PNG)

**Loss vs Steps for Cuda Net**
![Loss Graph](images/TotalLossGraph.PNG)

### Performance

*****

[acc vs steps graph for simple]
	- Make script for training
	- Make script for evaluating the training 
	- Use tensorboard eval to track 
[acc vs steps graph for cuda]

Simple Architecture with 10k steps: .543
Cuda convnet setup precision after 118k steps: .8174

### Final Results

*****

### Conclusion

*****
