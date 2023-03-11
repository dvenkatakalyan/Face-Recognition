import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import  storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL':"https://faceattendance-5d485-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendance-5d485.appspot.com"
})


# Importing student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))#list of all images of students
    studentIds.append(os.path.splitext(path)[0])#[id_1,id_2,id_3]-->list of all ids of students obtained by extracting them from image name

    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)#Creating blob to send images to storage in server
    blob.upload_from_filename(fileName)#This will send data
    #It will create folder called images in that we add all these images. 
    #Images are being uploaded we can download them whenever required

    # print(path)
    # print(os.path.splitext(path)[0])
print(studentIds)


def findEncodings(imagesList):#Generate encodings of image and store all encodings in a list.
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#Face recognition library uses rgb but opencv library gives image in bgr. So we convert bgr to rgb.
        encode = face_recognition.face_encodings(img)[0]#encoding of the image
        encodeList.append(encode) 

    return encodeList


print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)#dumping encoding in the pickle file
file.close()
print("File Saved")