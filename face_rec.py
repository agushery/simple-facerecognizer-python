import cv2
import numpy as np 
import sqlite3
import os
conn = sqlite3.connect('database.db')
if not os.path.exists('./dataset'):
    os.makedirs('./dataset')
c = conn.cursor()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
username = input("Type your name here: ")
c.execute('INSERT INTO users (name) VALUES (?)', (username,))
uid = c.lastrowid
sn = 0
while True:
  ret, img = cap.read()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x,y,w,h) in faces:
    sn = sn+1
    cv2.imwrite("dataset/User."+str(uid)+"."+str(sn)+".jpg",gray[y:y+h,x:x+w])
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
    cv2.waitKey(100)
  cv2.imshow('Taking snaps of your face',img)
  cv2.waitKey(1);
  if sn > 100:
    break
cap.release()
conn.commit()
conn.close()
cv2.destroyAllWindows()