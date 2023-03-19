import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.model_selection import train_test_split
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL import Image as im
import numpy as np
import pickle
import numpy as np
import cv2

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
import torchvision.models as models
import torch

from threading import Thread, Lock

from flask import Flask, request, Response, send_file, jsonify
from flask_cors import CORS, cross_origin

model = tf.keras.models.load_model('posemodelv2.h5')

path = r'C:\Users\Jay_Y\OneDrive\Desktop\hackathon backend\tmp\file.png'
outpath = r'C:\Users\Jay_Y\OneDrive\Desktop\hackathon backend\tmp\out.png'

img_height = 100
img_width = 100
num_channels = 3

#define text constants
font = cv2.FONT_HERSHEY_SIMPLEX

img_places = [(50, 50), (50, 70), (50, 90), (50, 110), (50, 130)]

fontScale = 0.5 
color = (0, 255, 255)
thickness = 1

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.6) as hands:


    app = Flask(__name__)
    cors = CORS(app, origins=["*"], allow_headers=["*"])


    @app.route("/")
    def hello_world():
        return Response("{ \"success\": true }", mimetype="application/json")

    @app.route("/outimage")
    def send_out_image():
        return send_file(outpath, max_age=0)
    
    def process_image_pose(image):
        # print('processing with pose')

        image = cv2.flip(image, 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        no_hand = True

        if results.multi_hand_landmarks:
            no_hand = False
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        if not results.multi_hand_landmarks:
            image = cv2.putText(image, 'No Hand', img_places[0], font, 
                fontScale, color, thickness, cv2.LINE_AA)
            # print('not a hand')
            cv2.imwrite(outpath, image)
            return '?', [['A', 0], ['A', 0], ['A', 0], ['A', 0], ['A', 0]]
        
        one_hand = []
        for hand_landmarks in results.multi_hand_landmarks:
            for i in hand_landmarks.landmark:
                one_hand.extend([i.x,
                                i.y,
                                i.z])
        one_hand = np.array(one_hand[:63])
        print(one_hand.shape)
        # Call the predict method on the model to get the predicted label
        predicted_label = model.predict(np.reshape(one_hand, (1, ) + one_hand.shape))
        
        # Get the index of the predicted label with the highest probability
        predicted_index = np.argmax(predicted_label)
        
        # Convert the predicted index back to a letter using the labels list
        labels = [chr(j) for j in range(ord('A'), ord('Z') + 1)]
        predicted_letter = labels[predicted_index]
        
        # Print the predicted letter
        print(f'Predicted: {predicted_letter}')

        if no_hand:
            image = cv2.putText(image, 'No Hand', img_places[0], font, 
                fontScale, color, thickness, cv2.LINE_AA)
            predicted_letter = "?"
        else:
            image = cv2.putText(image, f'prediction {predicted_letter}', img_places[0], font, 
                                fontScale, color, thickness, cv2.LINE_AA)

        cv2.imwrite(outpath, image)

        top_list = [[chr(idx + ord('A')), int(prob * 10000) / 100 ] for idx, prob in enumerate(predicted_label[0])]
        top_list.sort(key = lambda x: x[1], reverse = True)

        print(top_list)

        return predicted_letter, top_list[:5]
    
    # process_image_pose()
    # exit()

    mutex = Lock()
    @app.route("/senddata", methods=["GET", "POST"])
    # @cross_origin(origins=["*"])
    def process_data():
        # request.
        mutex.acquire(True, 0.1)
        try:
            #read image file string data
            filestr = request.files['file'].read()
            #convert string data to numpy array
            file_bytes = np.fromstring(filestr, np.uint8)
            # convert numpy array to image
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

            result , top = process_image_pose(img)
            results = {"success": True, "topcandidates": top, "result" : result}        

            return jsonify(results)

        except Exception as e:
            print("ERORORORRORORORORORORORORORORORORORO")
            print(e)
            return Response("{ \"success\": false }", mimetype="application/json")
        finally:
            mutex.release()


    if __name__ == '__main__':
        context = ('./.cert/cert.pem', './.cert/key.pem')   #certificate and key files
        app.run(host="0.0.0.0", ssl_context=context, port=8765)