import cv2 as cv

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier('/home/hiago/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('/home/hiago/opencv/data/haarcascades/haarcascade_eye.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #gray = cv.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        #print(x, y, w, h)
        cv.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(50, 50),
                maxSize=(60, 60)
                )

        for (ex, ey, ew, eh) in eyes:
            print(ex, ey, ew, eh)
            cv.rectangle(roi_gray, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    # Display the resulting frame
    cv.imshow('Testing...', gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    if cv.waitKey(1) & 0xFF == ord('c'):
        cv.imwrite('~/Documentos/eye-tracking/cap.png', gray)
        

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
