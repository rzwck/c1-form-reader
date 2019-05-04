# C1 Form Reader 

Application of computer vision and convolutional neural network (CNN) for automatically reading hand written numbers on C1 form (Indonesia election's tally form).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You need to have python 3.x and the following libraries must be installed to run c1 form reader:

* OpenCV
* Keras
* Tensorflow
* Numpy
* Matplotlib


### Installing

Simplest way to install all the requirements is using anaconda.

```
$ conda create -n c1reader python=3.7 anaconda
$ source activate c1reader
(c1reader) $ pip install opencv-python
(c1reader) $ pip install keras
(c1reader) $ pip install tensorflow
```

Once all requirements are installed, clone this repository.

```
(c1reader) $ git clone https://github.com/rzwck/c1-form-reader.git
(c1reader) $ cd c1-form-reader
```

## Running

From the c1-form-reader you can just run the script as the following example:

```
(c1reader) $ python read-C1-form.py
Using TensorFlow backend.
test_images/test1.jpg (12, 115, 127, 0, 120)
test_images/test10.jpg (136, 11, 147, 6, 153)
Form test_images/test11.jpg unreadable
test_images/test12.jpg (190, 15, 205, 4, 209)
test_images/test13.jpg (44, 61, 109, 0, 150)
Form test_images/test14.jpg unreadable
test_images/test15.jpg (119, 19, 138, 0, 138)
Form test_images/test16.jpg unreadable
test_images/test2.jpg (125, 57, 182, 3, 185)
Form test_images/test3.jpg unreadable
test_images/test4.jpg (128, 28, 156, 5, 161)
test_images/test5.jpg (121, 3, 124, 5, 129)
test_images/test6.jpg (142, 82, 224, 3, 227)
Form test_images/test7.jpg unreadable
Form test_images/test8.jpg unreadable
test_images/test9.jpg (139, 7, 146, 0, 146)
```

The script saves the output file on the same folder (test_images). The following screenshot shows some of the outputs produced by c1 form reader.
Green boxes are area containing hand written digits, small black boxes with white written numbers are 28x28 hand written digits that will be fed to CNN classifiers.
The final number recognized by this program is the white number written on the blue rectangles.

![](/static/sample-outputs.png)

## Built With

This scripts utilizes codes from the following sites:
* [PyImageSearch](https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/) - for perspective transformation
* [Keras examples](https://maven.apache.org/) - for training hand written digits classifiers

## Training dataset

I trained two digits (0-9) classifiers from two different datasets:

* [MNIST database](http://yann.lecun.com/exdb/mnist/) - samples of hand written digits
* [Pilkada DKI 2017](https://pilkada2017.kpu.go.id) - extracted hand written digits from pilkada DKI's C1 form, since local people might have local hand written styles that is not contained in standard MNIST dataset.

I also trained "X" classifiers and "-" hyphen classifiers for recognizing "X" and "-" characters that is commonly used to denote empty digit box. The sample for this "X" and "-" classifier also came from Pilkada DKI 2017.


