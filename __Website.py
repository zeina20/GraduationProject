
from flask import Flask, redirect, url_for, render_template, request, jsonify
from PIL import Image
import cv2  
import glob
import pickle
import pandas as pd
from tensorflow import keras
import tensorflow 
app = Flask(__name__)

# Defining the home page of our site
 # some basic inline html

@app.route("/")
def home():
    return render_template("temp.html")  

ALLOWED_EXTENSIONS = ['mp4']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return 'No video file found'
    video = request.files['video']
    if video.filename == '':
        return 'No video selected'
    if video and allowed_file(video.filename):
        video.save('static/videos/' + video.filename)
        
        

        img_number=0
        # Load the cascade  
        face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        #file path
        path='static/videos/' + video.filename  
        all_videos=glob.glob(path)
        li=[]
        counter=0
        for video in all_videos:
        # To capture video from existing video.  
            cap = cv2.VideoCapture(video)  
            counter=counter+1
            print(counter)
            #while True:  
                # Read the frame 
            _, img = cap.read()     
            _, img = cap.read()
            #cv2_imshow(img)  
                # Convert to grayscale  
                #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
            
                # Detect the faces  
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=2)  
            wmax=0
            hmax=0
            ImageFolder ='faceimg/'
                # Draw the rectangle around each face  
            for (x, y, w, h) in faces: 
                if w>wmax and h>hmax:
                    wmax=w
                    hmax=h
            #print(wmax)
            #print(hmax)
            for (x, y, w, h) in faces: 
                img_number=img_number+1
                if w == wmax and h==hmax : 
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)  
                    crop_img = img[y:y+h, x:x+w]
                    li.append(crop_img)
                    cv2.imwrite("faceimg/"+str("image")+".jpg",crop_img)
                    #crop_img = crop_img.save("/content/gdrive/MyDrive/Fake-Face-Detection-DeepFakes-CNN-master/faces/trainimage_"+str(img_number)+".jpg")
                    #test=cv2.imwrite('/content/gdrive/MyDrive/Fake-Face-Detection-DeepFakes-CNN-master/faces/train/image_{i}.png',crop_img)
                    #print(test)
                    
                # Display  
                    # cv2_imshow(crop_img) 
                    #img.save('/content/gdrive/MyDrive/Fake-Face-Detection-DeepFakes-CNN-master/faces/train'+ crop_img , 'JPEG')
                # Stop if escape key is pressed  
            k = cv2.waitKey(1000) & 0xff  
            if k==27:  
                break  
                    
        # Release the VideoCapture object  
        cap.release()
        print("preload")
        #model_file = open('D:/Zeina MIU/Year4/GP/Website (1)/model.pkl', 'rb')
        

        #obj = pd.read_pickle('D:/Zeina MIU/Year4/GP/Website (1)/model.pkl')
        #pickled_model = keras.models.load_model('D:/Zeina MIU/Year4/GP/Website (1)/_cnnmodel.hdf5')
        pickled_model=tensorflow.keras.models.load_model("D:/Zeina MIU/Year4/GP/Website (1)/vggmodel.hdf5", compile=False)
        #pickled_model.compile()
        #print(obj)
        import numpy as np
        print("precompile")

        #pickled_model = pickle.load(model_file)
        pickled_model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])
        print("postcompile")

        img = cv2.imread('faceimg/image.jpg')
        gray = cv2.resize(img,(3,128))    
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #gray = np.reshape(gray,(128,128,3))
        #final_image = np.reshape(gray, (-1, gray.shape[0], gray.shape[1],3))
        img_test = tensorflow.expand_dims(gray, axis=0)
        #img_test = tensorflow.expand_dims(gray, axis=-1)
        #img_test = tensorflow.expand_dims(gray, axis=1)
        # img = np.reshape(img,[1,96,96,1])

        # image = np.array(Image.open("/content/drive/MyDrive/Fake-Face-Detection-DeepFakes-CNN-master/faces/train/image_1007.jpg").resize((96, 96)))
        images_list = []
        images_list.append(np.array(img_test))
        x = np.asarray(images_list)
        pr_mask = pickled_model.predict(x).round()

        # plt.imshow(
        # pr_mask[0]
        # )
        # plt.show()
        print(pr_mask[0])
        #classes = pickled_model.predict_classes(img)
        #predict_x=pickled_model.predict(img) 
        #classes_x=np.argmax(predict_x,axis=1)

        #print (predict_x)
        return render_template('preview.html', video_name=video.filename)
    
if __name__ == "__main__":
    app.debug = False
    app.run()




