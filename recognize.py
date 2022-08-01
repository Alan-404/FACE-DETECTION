import cv2 as cv
import numpy as np
import pandas as pd
import os

size = (155, 155)
class Recognize:
    def __init__(self):
        self.face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.fontface=cv.FONT_HERSHEY_SIMPLEX
        self.recognizer = cv.face.LBPHFaceRecognizer_create()
        if os.path.exists('./models/main.yml'):
            self.recognizer.read('./models/main.yml')
        self.df = pd.read_csv('./user.csv')
        self.bgr_recognize = (0,255,255)
        self.bgr_train = (250, 206, 135)
        self.path_dataset = 'dataset'
    
    def get_profile(self, id):
        user_row = self.df['name'][self.df['id'] == id]

        if len(user_row) != 1:
            return None
        user_arr = np.array(user_row)
        return user_arr[0]

    def reg_user(self,frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        gray = cv.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(gray)
        
        for (x,y,w,h) in faces:
            cv.rectangle(frame, (x,y), (x+w, y+h), self.bgr_recognize, 2)
            
            user_face = gray[y: y+h, x: x + w]

            user_face = cv.resize(user_face, size)
            
            id, confusion = self.recognizer.predict(user_face)
            
            if confusion < 60:
                profile = self.get_profile(id)
                if profile != None:
                    cv.putText(frame, str(profile), (x+10, y+h+30), self.fontface, 1, self.bgr_recognize, 2)
                else:
                    cv.putText(frame, "Unknow", (x+10, y+h+30), self.fontface, 1, self.bgr_recognize, 2)
            else:
                cv.putText(frame, "Unknow", (x+10, y+h+30), self.fontface, 1, self.bgr_recognize, 2)
            
        return frame



    def detect_face(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 10)
        
        for (x,y,w,h) in faces:
            cv.rectangle(frame, (x,y), (x+w, y+h), self.bgr_train, 2)
        return frame

    def get_next_id(self):
        self.df = pd.read_csv('./user.csv')

        ids = np.array(self.df['id'])
        num = 1

        for i in ids:
            if num != i:
                return num
            num += 1
        return num
    
    def get_images_by_id(self):
        faces = []
        IDs = []
        for folder in os.listdir(self.path_dataset):

            path = f"{self.path_dataset}/{folder}"
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            
            
            
            for imagePath in imagePaths:

                faceImg = cv.imread(imagePath)
                
                temp = cv.cvtColor(faceImg, cv.COLOR_BGR2GRAY)
                temp = cv.equalizeHist(temp)
                temp = cv.resize(temp, size)
                tempNp = np.array(temp, 'uint8')
                Id = int(imagePath.split("\\")[1].split('.')[1])
                faces.append(tempNp)
                IDs.append(Id)

                
            
        return np.array(faces), np.array(IDs)

    def train_model(self):
        faces, ids = self.get_images_by_id()
        self.recognizer.train(faces, ids)

        if not os.path.exists('./models'):
            os.makedirs('./models')

        self.recognizer.save('./models/main.yml')


