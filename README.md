# Face Recognision
Building a Real time Face attendance system that is linked with real time database.


## Flow Chart
![Flow Chart](Flow_chart.PNG)

## Appendix
1) First we run the webcam

2) we add graphics(we will have an interface in which we will show whether its active, whether we are detecting the face, loading ) 

3) we will creating an Encoding Generator.Run this once to create encoding of all the the images.Whenever you have new student you have to create encodings again.you will have to update them.This is a separate script.

4) Face recognition uses the Encodings generated by encoding generator to detect faces.

5) Create a separate script so that it becomes easy to add data to the database

6) Images are stored in a storage bucket.

7) Whenever there is an attendance we need to update database whenever there is face detected.

8) Once we have taken attendance. We shouldn't take another attendace until some amount of time.

## Deployment

### Database setup

Create new project in firebase
Create new database here.This uses json format.Create storage bucket to store all of these images.Generate a new private key and add to our code in AddDatatoDatabase. The link of the database should be copied from firebase and pasted in the code inside 'databaseURL' in json format.Create different person's IDs and other details in json format. Send the data into database using.
```bash
  python AddDatatoDatabase.py
```
### Encoding Images
In credentials,the link of the storage bucket should be copied from firebase and pasted in the code inside 'storageBucket' in json format.Here we are using EncodeGenerator to generate encoding, send to database and  store it in EncodeFile.p file.
```bash
  python EncodeGenerator.py
```

### Attendance 
```bash
  python main.py
```
## Demo
![output](output.gif)

