import os
import pickle
import numpy as np
import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import numpy as np
from datetime import datetime

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "",
    'storageBucket':  ""
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3, 640) #width of frame=640
cap.set(4, 480) #width of height=480

imgBackground = cv2.imread('Resources/background.png') #background image

# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))#List containing all images(modes)
# print(len(imgModeList))

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
# print(studentIds)
print("Encode File Loaded")

modeType = 0 #If modeType=0 then show Active image file
counter = 0 #Once face is detected we only need to download info in first frame(first iteration). we cant keep downloading because it will be very inefficient.
    
id = -1
imgStudent = []

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)#Resizing the image(to 1/4 * original size) because larger the image more the computation power
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)#Face_recognition takes RGB format. Opencv gives BGR format images.converting BGR to RGB.

    faceCurFrame = face_recognition.face_locations(imgS) #Faces in current frame
    #imGs-->resized image
    #faceCurFrame-->Face locations
    #We don't want to find encodings of whole image. We want encoding of the face
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)#Encodings in current frame

    #we have to take webcam image and overlay it on background
    imgBackground[162:162 + 480, 55:55 + 640] = img #[start_height:end_height,start_width:end_width]
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType] 
    #Resources folder--->modes-->images
    #modetype--->active,info_of_student,marked,already_marked

    if faceCurFrame: #If face is detected
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):#loop through all current encodings one by one
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace) #comparing the encodings of the images stored with the current face encoding
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print("matches", matches)
            # print("faceDis", faceDis)
            #matches=[True,False,False] the current frame has my image it matches with 1st image in the stored images so we get True.It doesn't match with other images so we get False.
            #faceDis=[0.3811 0.864 0.7640] How good the match is given by the face distance. The lower the face distance the better the match.

            matchIndex = np.argmin(faceDis) #index of least faceDis(here 0)
            # print("Match Index", matchIndex)

            if matches[matchIndex]: #check in the matches list if at matchindex(here 0) there is 'True' then we will say known face detected.
                # print("Known Face Detected")
                # print(studentIds[matchIndex])

                #Draw rectangle around face
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4 #Multiplying by 4 because previously size was reduced to 1/4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1#On the background image our frame is not at (0,0) pixels. We have to add some x,y value. 
                                                            #width=x2-x1,height=y2-y1
                x=55 + x1
                y=162 + y1
                w=x2 - x1
                h=y2 - y1
                imgBackground = cv2.rectangle(imgBackground, (x, y), (x+w, y+h), (0, 255, 0), 1)
                #imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0) #Draw rectangle around face.
                id = studentIds[matchIndex] #Get Id of the matched person
                if counter == 0:#If counter is previously zero(it is initialized zero)
                    #cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                    #cv2.putText(imgBackground, text, text_origin, font, font_scale, color, thickness)
                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)
                    counter = 1# Make it(counter) 1
                    modeType = 1#Active mode

        if counter != 0:

            if counter == 1: #For first frame
                #Download data from database and show
                # Get the Data
                studentInfo = db.reference(f'Students/{id}').get() #getting all info of that matched person id from database
                print(studentInfo)#dictionary{'last_attendance_time':2022-12-11 00:54:34','major':'Robotics','name':Murtaze Hassan,'standing':'G'}
                # Get the Image from the storage
                blob = bucket.get_blob(f'Images/{id}.png')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                # Update data of attendance
                datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                   "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                print(secondsElapsed)
                if secondsElapsed > 30: #Next Attendance done after x(Generally 24hrs here for testing we take 30 secs) amount of time after previous attendence.
                    ref = db.reference(f'Students/{id}')
                    studentInfo['total_attendance'] += 1 #updating the attendance
                    ref.child('total_attendance').set(studentInfo['total_attendance']) #update that value in database for the matched person
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    modeType = 3
                    counter = 0
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            if modeType != 3:

                if 10 < counter < 20:
                    modeType = 2#In next 10 frames it will say marked

                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if counter <= 10:#modetype=1
                    #display all of these
                    #First 10 frames it shows the image and all data 
                    #Adding all parameters like name,id to matching background image at respective pixel positions.
                    cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(id), (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['year']), (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
                    cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                    (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)#Get size of text (width,height,_)
                    offset = (414 - w) // 2#((total_width-width_of_text)/2)
                    cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                    imgBackground[175:175 + 216, 909:909 + 216] = imgStudent #Matched Image downloaded from database is displayed

                counter += 1

                if counter >= 20: #go back to active after 20 frames
                    #Resetting everything
                    counter = 0
                    modeType = 0
                    studentInfo = []
                    imgStudent = []
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    else:#No face 
        modeType = 0
        counter = 0
    # cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)