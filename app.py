# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 19:19:44 2021
@author: Sonu
"""

import time
import cv2
from flask import Flask, render_template, Response
from keras.models import load_model
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen():
    model = load_model('face-mask.h5')
    results = {0: 'Without Mask', 1: 'Mask'}
    GR_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
    rect_size = 4

    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)
    haarcascade = cv2.CascadeClassifier(
        'C:/Users/sonuc\Desktop/Data_Science/facemask_detection/haarcascade_frontalface_default.xml')

    while True:
        (rval, im) = cap.read()  # rval give us true or flase and im give us image data
        im = cv2.flip(im, 1, 1)

        rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
        faces = haarcascade.detectMultiScale(rerect_size)
        for f in faces:
            (x, y, w, h) = [v * rect_size for v in f]

            face_img = im[y:y + h, x:x + w]
            rerect_sized = cv2.resize(face_img, (150, 150))
            normalized = rerect_sized / 255.0
            reshaped = np.reshape(normalized, (1, 150, 150, 3))
            reshaped = np.vstack([reshaped])
            result = model.predict(reshaped)
            # print(result)
            label = np.argmax(result, axis=1)[0]

            cv2.rectangle(im, (x, y), (x + w, y + h), GR_dict[label], 2)
            cv2.rectangle(im, (x, y - 40), (x + w, y), GR_dict[label], -1)
            cv2.putText(im, results[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # cv2.imshow('LIVE', im)
        # cv2.imshow("Pose detection", img)
        frame = cv2.imencode('.jpg', im)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
