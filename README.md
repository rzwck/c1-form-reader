# C1 Form Reader 

Application of computer vision and convolutional neural network (CNN) for automatically reading hand written numbers on C1 form (Indonesia election's tally form).

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
2019-05-05 11:19:51.089820: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
test_images/test1.jpg {'01': 12, '02': 115, 'valid': 122, 'invalid': 0, 'total': 127}
test_images/test10.jpg {'01': 136, '02': 11, 'valid': 147, 'invalid': 6, 'total': 153}
Form test_images/test11.jpg is unreadable: Unable to find digit positions for invalid ballots count
test_images/test12.jpg {'01': 190, '02': 415, 'valid': 205, 'invalid': 4, 'total': 209}
test_images/test13.jpg {'01': 44, '02': 61, 'valid': 109, 'invalid': 0, 'total': 150}
Form test_images/test14.jpg is unreadable: Unable to find digit positions for valid ballots count
test_images/test15.jpg {'01': 119, '02': 19, 'valid': 138, 'invalid': 0, 'total': 138}
Form test_images/test16.jpg is unreadable: Unable to find digit positions for votes #01
test_images/test17.jpg {'01': 69, '02': 160, 'valid': 229, 'invalid': 7, 'total': 231}
Form test_images/test18.jpg is unreadable: Unable to find digit positions for invalid ballots count
Form test_images/test19.jpg is unreadable: Unable to find digit positions for votes #02
test_images/test2.jpg {'01': 125, '02': 57, 'valid': 182, 'invalid': 3, 'total': 185}
Form test_images/test20.jpg is unreadable: Unable to find digits for total ballots count
test_images/test21.jpg {'01': 72, '02': 164, 'valid': 236, 'invalid': 4, 'total': 240}
test_images/test22.jpg {'01': 116, '02': 24, 'valid': 140, 'invalid': 3, 'total': 143}
test_images/test23.jpg {'01': 59, '02': 150, 'valid': 209, 'invalid': 1, 'total': 210}
test_images/test24.jpg {'01': 89, '02': 133, 'valid': 223, 'invalid': 5, 'total': 228}
Form test_images/test25.jpg is unreadable: Unable to find digit positions for valid ballots count
Form test_images/test3.jpg is unreadable: Unable to find digit positions for votes #01
test_images/test4.jpg {'01': 128, '02': 28, 'valid': 156, 'invalid': 5, 'total': 161}
test_images/test5.jpg {'01': 121, '02': 3, 'valid': 124, 'invalid': 5, 'total': 129}
Form test_images/test6.jpg is unreadable: Unable to find digit positions for votes #02
Form test_images/test7.jpg is unreadable: Unable to find digit positions for votes #02
test_images/test8.jpg {'01': 143, '02': 89, 'valid': 232, 'invalid': 7, 'total': 239}
test_images/test9.jpg {'01': 139, '02': 7, 'valid': 146, 'invalid': 0, 'total': 146}
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


