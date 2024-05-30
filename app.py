from flask import Flask, render_template, request, redirect, url_for, session,flash
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user
from flask_session import Session 
import sqlite3
import easygui
from cv2 import *
import cv2
import os
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
import tqdm
import glob
import tensorflow
import random
import smtplib
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
import numpy as np
import csv
import datetime
from PIL import Image
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
from skimage.transform import resize

DATABASE="newdata2.db"
app = Flask(__name__)
app.secret_key = "jgwgabduydhb7367867"
print("gi")
def createtable():
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                                  student_name TEXT NOT NULL ,
                                  student_id TEXT NOT NULL,
                                  email TEXT NOT NULL,
                                  department TEXT NOT NULL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS login (
                                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                                  student_id  TEXT NOT NULL ,
                                  email TEXT NOT NULL,
                                  date DATE DEFAULT (DATE('now', 'localtime'))
                                  )''')
        conn.commit()
        conn.close()
        
createtable()


@app.route('/')
@app.route('/index')
def index():
        return render_template('index.html')
@app.route('/complete')
def complete():
        return render_template('complete.html')


sid=[]

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        studentname= request.form['studentname']
        department= request.form['department']        
        studentid= request.form['studentid']
        sid.append(studentid)
        v1=str(studentid)
        print( len(v1))
        if len(v1)!=6:
            return display_popup1(" please enter a valid voter id ")
        email = request.form['email']
        confirm_email = request.form['confirm_email']
        if email != confirm_email:
            return display_popup1(" your email does not match")
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM students WHERE email=?", (email,))
        registered = cursor.fetchall()
        print(registered)
        cursor.execute("SELECT student_id  FROM students WHERE student_id=?", (studentid,))
        registered1 = cursor.fetchall()
        print(registered1)
        if registered:
            return display_popup1(" your email already registered")
        elif registered1:
            return display_popup1("your studentid  already registered")
        else:
            cursor.execute("INSERT INTO students (student_name, email,student_id ,department) VALUES (?, ?,?,?)", (studentname, email,studentid,department))
            conn.commit()
            conn.close()
            return render_template('data.html')
    return render_template('register.html')





@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        sid1= sid [-1]
        cam = cv2.VideoCapture(0)       
        time.sleep(5)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        
        while True:
            ret, img = cam.read()
            
            if not ret:
                print("Failed to capture frame from webcam.")
                return "Failed to capture frame from webcam."            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum +=1
                cv2.imwrite(f"TrainingImages/{sampleNum}.{sid1}.jpg", gray[y:y+h, x:x+w])
                cv2.imshow('frame', img)
                #print("hello")                
            if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 30:
                break
        
        cam.release()
        cv2.destroyAllWindows()
        return render_template('train.html')
    return render_template('data.html')



def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Create LBPH Face Recognizer
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImages")
    recognizer.train(faces, np.array(Id))
    #print(np.array(Id))
    recognizer.save("train2.yml")
    #print("trained")

def getImagesAndLabels(path):
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
        faces=[]
        Ids=[]
        for imagePath in imagePaths:
            pilImage=Image.open(imagePath).convert('L')
            imageNp=np.array(pilImage,'uint8')
            Id=int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(Id)
        return faces,Ids


@app.route('/train', methods=['GET'])
def train():
    return render_template('train.html')

@app.route('/training', methods=['POST'])
def training():
    if request.method == 'POST':       
            TrainImages()
            return display_popup2(" succuess fully registered your details  ")
    return render_template('train.html')


def go():
        studentid=s1[-1]
        email=e1[-1]
        #print(studentid,email,"go")
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()  
        cursor.execute("SELECT * FROM students WHERE student_id =? AND email=?", (studentid , email))
        user = cursor.fetchone()
        if user:              
                recognizer = cv2.face_LBPHFaceRecognizer.create()#cv2.createLBPHFaceRecognizer()
                recognizer.read("train2.yml")
                harcascadePath = "haarcascade_frontalface_default.xml"
                faceCascade = cv2.CascadeClassifier(harcascadePath)
                cam = cv2.VideoCapture(0)
                while True:
                        
                        ret,im =cam.read()
                        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                        faces = faceCascade.detectMultiScale(gray, 1.3,5)
                        name2=[]
                        for(x, y, w, h) in faces:
                            cv2.rectangle(im,(x, y), (x + w, y + h), (255,0,0), 2)
                            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                            #print(Id,conf)
                            if conf < 50:                                  
                                        name=Id
                                        name1=str(name)
                                        #print(name1)
                                        studentid1=str(studentid)
                                        #print(studentid)
                                        if (name1==studentid1):
                                            #print("after",name1,studentid1)
                                            cursor.execute("SELECT email FROM students WHERE student_id =? AND email=?", (studentid , email))
                                            user2 = cursor.fetchone()
                                            cursor.execute("SELECT student_name FROM students WHERE student_id =? AND email=?", (studentid , email))
                                            user5 = cursor.fetchone() 
                                            #print(user2)
                                            smtp_server = 'smtp.example.com'
                                            smtp_port = 587
                                            sender_email = 'triossoftwaremail@gmail.com'
                                            sender_password = 'knaxddlwfpkplsik'
                                            receiver_email = user2
                                            host = "smtp.gmail.com"
                                            mmail = 'triossoftwaremail@gmail.com'      
                                            hmail = user2[0]
                                            receiver_name = studentid
                                            sender_name= "management"
                                            msg = MIMEMultipart()
                                            subject = f"attenance{user5}"
                                            text ="dear student your attenance successfully registered " 
                                            msg = MIMEText(text, 'plain')
                                            msg['To'] = formataddr((receiver_name, hmail))
                                            msg['From'] = formataddr((sender_name, mmail))
                                            msg['Subject'] = 'Hello  ' + receiver_name
                                            server = smtplib.SMTP(host, 587)
                                            server.ehlo()
                                            server.starttls()
                                            password = "knaxddlwfpkplsik"
                                            server.login(mmail, password)
                                            server.sendmail(mmail, [hmail], msg.as_string())
                                            server.quit()
                                            send="send"
                                            print(send)
                                            cursor.execute("INSERT INTO login (email,student_id ) VALUES (?, ?)", ( email,studentid ))
                                            conn.commit()
                                            cursor.execute("SELECT login.date, students.* FROM login LEFT JOIN students ON students.student_id = login.student_id")
                                            data = cursor.fetchall()
                                            print(data)
                                            csv_file = 'attenance_details.csv'
                                            with open(csv_file, 'w', newline='') as csvfile:
                                                    csvwriter = csv.writer(csvfile)
                                                    header = [col[0] for col in cursor.description]
                                                    csvwriter.writerow(header)
                                                    csvwriter.writerows(data)
                                            conn.commit()
                                            conn.close()
                                            return  send
                                            cam.release()
                                            cv2.destroyAllWindows()
                                            break
                                        else: 
                                            return display_popup("User face mismatch")
                            else:
                                 print("outof range")
        else:
               return display_popup("Not registered")

    
s1=[]
e1=[]
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':                                       
                studentid = request.form['studentid']
                s1.append(studentid)
                email = request.form['email']
                e1.append(email)
                current_date = datetime.datetime.now().date()
                #print(current_date)
                c=str(current_date)
                conn = sqlite3.connect(DATABASE)
                cursor = conn.cursor()
                cursor.execute("SELECT date FROM login WHERE student_id =? AND email=?", (studentid ,email))
                user1 = cursor.fetchone()
                #print(user1)
                if user1 is None:
                        a=go()
                        if a=="send":
                                return render_template('complete.html')
                                
                else:
                        l=(list(user1))
                        l2=(l[0])
                        #print(l2)
                        l2=str(l2)
                        #print(type(c))
                        #print(c)
                        if user1 is not None and l2 == c:
                            return display_popup("Your attenance already registered")
                        else:
                                a=go()
                                if a=="send":
                                        return render_template('complete.html')
    return render_template('login.html')



ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin'

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            return redirect('/details')
    return render_template('admin.html')


@app.route('/details',methods=['GET', 'POST'])
def details():
     if request.method == 'POST':
         print("details")
     return render_template('details.html')

def get_table_data3():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students")
    table_data = cursor.fetchall()

    conn.close()
    return table_data

@app.route('/table')
def table():
    table_data = get_table_data3()

    return render_template('table.html', table_data=table_data)




def get_table_data5():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT login.date, students.* FROM login LEFT JOIN students ON students.student_id = login.student_id")
    table_data = cursor.fetchall()
    conn.close()
    return table_data

@app.route('/voted_ist')
def voted_ist():
    table_data = get_table_data5()
    return render_template('voted_ist.html', table_data=table_data)


def display_popup2(message):
    flash(message)
    return redirect(url_for('index'))
def display_popup1(message):
    flash(message)
    return redirect(url_for('register'))
def display_popup(message):
    flash(message)
    return redirect(url_for('login'))

    
if __name__ == '__main__':
    app.run(debug=False,port=500)
