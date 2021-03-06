
# Face-Mask-Detection-Web-App

Face Mask Detection web application built with Flask, Keras-TensorFlow, OpenCV. It can be used to detect face masks both in images and in real-time camera video using browser.


## Goal

The goal is to create a masks detection system, able to recognize face masks both in images, both in real-time video, drawing bounding box around faces. In order to do so, I finetuned Neural Network pretrained on CNN, in conjunction with the OpenCV face detection algorithm: that allows me to turn a classifier model into an object detection system.

## Technologies

- Keras/Tensorflow
- OpenCV
- Flask
- CNN

## Usage

You have to install the required packages, you can do it

via pip `pip install -r requirements.txt`

or via `conda conda env create -f environment.yml`

Once you installed all the required packages you can type in the command line from the root folder:

```bash
  python app.py

```
and click on the link that the you will see on the prompt.


