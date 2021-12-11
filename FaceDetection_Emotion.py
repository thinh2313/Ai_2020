
get_ipython().system('if not exist "./files" mkdir files')
get_ipython().system('curl -L -o ./files/haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml')
get_ipython().system('curl -L -o ./files/emotion_model.hdf5 https://mechasolution.vn/source/blog/AI-tutorial/Emotion_Recognition/emotion_model.hdf5')


import cv2
import numpy as np 
from keras.preprocessing.image import img_to_array
from keras.models import load_model

face_detection = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('files/emotion_model.hdf5', compile=False)
EMOTIONS = ["Tuc gian","Kinh tom","So hai", "Hanh phuc", "Buon ba", "Bat ngo", "Binh thuong"]
camera = cv2.VideoCapture(0)

while True:
    ret,frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    canvas = np.zeros((250, 300, 3), dtype="uint8")

    if len(faces) > 0:
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = face
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        # Thực hiện dự đoán cảm xúc
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        # Gán nhãn cảm xúc dự đoán được lên hình
        cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
         for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)    
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    cv2.imshow('Emotion Recognition', frame)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
roi = gray[fY:fY + fH, fX:fX + fW] 
roi = cv2.resize(roi, (48, 48)) 
roi = roi.astype("float") / 255.0 
roi = img_to_array(roi)
roi = np.expand_dims(roi, axis=0)
preds = emotion_classifier.predict(roi)[0] 
label = EMOTIONS[preds.argmax()] 
cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)


# In[ ]:




