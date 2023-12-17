import cv2
from deepface import DeepFace

video = cv2.VideoCapture(0)  # Selects the default camera
fdetect = cv2.CascadeClassifier(r'directory/haarcascade_frontalface_default.xml')

while True:
    ret, frame = video.read()
    
    emotion_analysis = DeepFace.analyze(frame, actions=['emotion'],enforce_detection=False)
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fimage = fdetect.detectMultiScale(image, scaleFactor=1.5)
    
    for x, y, w, h in fimage:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
    emotion = emotion_analysis[0]['dominant_emotion']
    cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)    
    cv2.imshow("video", frame)
    
   
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
