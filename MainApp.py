from flask import Flask, render_template, Response
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import set_session
d={0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprise"}

sess = tf.Session()
graph = tf.get_default_graph()
model=load_model(os.path.join(os.path.abspath('.'), 'fac_exp_mod.h5'))

face_cascade = cv2.CascadeClassifier(os.path.abspath('.')+'/haarcascade_frontalface_alt.xml')


########### Function for detecting the face #################
def detect_face(frame):
    img=frame.copy()
    face_rectangle=face_cascade.detectMultiScale(img, scaleFactor=1.2) #returns the 4 co-ordinates of the rectangle
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        for(x, y, w, h) in face_rectangle:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            prediction = d[int(model.predict_classes(cropped_img))]
            cv2.putText(img, prediction, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

    return img
    


app=Flask(__name__)

@app.route('/')
def index():
    return render_template('Index.html')


@app.route('/vid_stream')
def vid_stream():
    return render_template('vid_stream.html')


def gen():
    cap=cv2.VideoCapture(0)
    while True:
        ret, frame=cap.read()
        frame=cv2.flip(frame, 1)
        frame=detect_face(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/vid_str')
def vid_str():
    return Response((gen()),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=='__main__':
    app.run(debug=True)

