from keras.models import load_model
from time import sleep
from flask import Flask, render_template
import numpy as np
import pandas as pd
import joblib
import os
import sys
import math
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import pandas as pd
import numpy as np
import time

from flask import Flask, render_template, Response
import cv2

import numpy as np
app = Flask(__name__)


face_classifier = cv2.CascadeClassifier(
    r'D:\sem6 mini project\Emotion-Based-Media-Recommender-master\haarcascade_frontalface_default.xml')
classifier = load_model(
    r'D:\sem6 mini project\Emotion-Based-Media-Recommender-master\model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Neutral', 'Sad', 'Surprise']
path = r'G:\Emo-Media\opencv_frame_0.png'
cap = cv2.VideoCapture(0)

Music_Player = pd.read_csv("./data_moods.csv")
Music_Player = Music_Player[['name', 'artist', 'mood', 'popularity']]
arr = [] * 10
str2_array = []
str3_array = []

def Recommend_Songs(pred_class):
    if (pred_class == 'Disgust'):

        Play = Music_Player[Music_Player['mood'] == 'Sad']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:20].reset_index(drop=True)
        # display(Play)
        string_array = [','.join(row.astype(str)) for row in Play.to_numpy()]
        for item in string_array:
            str2_array.append(item)
        #print(str2_array)

        #print(Play)
        return Play

    if (pred_class == 'Happy'  or pred_class == 'Sad'):

        Play = Music_Player[Music_Player['mood'] == 'Happy']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:20].reset_index(drop=True)
        string_array = [','.join(row.astype(str)) for row in Play.to_numpy()]
        for item in string_array:
            str2_array.append(item)
        #print(str2_array)
        #print(Play)
        return Play

    if (pred_class == 'Fear' or pred_class == 'Angry'):

        Play = Music_Player[Music_Player['mood'] == 'Calm']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:20].reset_index(drop=True)
        string_array = [','.join(row.astype(str)) for row in Play.to_numpy()]
        for item in string_array:
            str2_array.append(item)
        #print(str2_array)
        #print(Play)
        return Play

    if (pred_class == 'Surprise' or pred_class == 'Neutral'):

        Play = Music_Player[Music_Player['mood'] == 'Energetic']
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:20].reset_index(drop=True)
        string_array = [','.join(row.astype(str)) for row in Play.to_numpy()]
        for item in string_array:
            str2_array.append(item)
        # print(str2_array)
        # print(Play)
        return Play


final_label = ""
TIMER = int(20)
emo = ''


@app.route('/')
def start():

    
    return render_template('start.html')



def gen_frames():
    camera = cv2.VideoCapture(0)
    
    i = 0
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_width = camera.get(3)
            frame_height = camera.get(4)
            center_x = int(frame_width / 2)
            center_y = int(frame_height / 2)
            center_radius = 100
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                face_center_x = x + int(w / 2)
                face_center_y = y + int(h / 2)
                distance_to_center = math.sqrt((center_x - face_center_x) ** 2 + (center_y - face_center_y) ** 2)

                if distance_to_center < center_radius:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                    if np.sum([roi_gray]) != 0:
                        roi = roi_gray.astype('float')/255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)

                        prediction = classifier.predict(roi)[0]
                        label = emotion_labels[prediction.argmax()]
                        label_position = (x, y)
                        cv2.putText(frame, label, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        print(label)
                        emo = label
                        print(emo)
                        Recommend_Songs(label)
                        i = 0  
                else:
                   
                    pass

            cv2.imshow('Emotion Detector', frame)

            if i >= 3:
                label = "happy"  
                print("no face detected" + label)
                Recommend_Songs(label)
            else:
                i = 0

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()
    cv2.destroyAllWindows()



@app.route('/close2')
def close2():
    return render_template('index2.html', string_array=str2_array[-20:])

@app.route('/close')
def close():
    return render_template('index.html', string_array=str2_array[-10:], show_table=True)


@app.route('/cap')
def index():

    gen_frames()
    return render_template('index.html')


@app.route('/restart')
def restart():
    os.execv(sys.executable, ['python'] + sys.argv)
    return 'Flask application restarted!'


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

