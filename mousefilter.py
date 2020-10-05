import numpy as np
import cv2

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

cam.set(3,480)
cam.set(4,320)

#face and eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while (cam.isOpened()):

    ret, frame = cam.read()

    if ret:

        frame =cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))

        # Draw the faces
        for (x, y, w, h) in faces:
            cv2.circle(frame,(x+w/2,y+w/2),(w/2),(131,139,139),-1)

            xoffset = int(w*0.75)
            yoffset = int(h*0.25)

            #draw mouth
            xm=int(x+w/2)
            ym=int(y+yoffset+w*0.4)
            
            length=int(w/4)
            height=int(yoffset)
            
            cv2.ellipse(frame,(xm,ym),(length,height),0,0,180,(203,192,255),-1)
            cv2.ellipse(frame,(xm,ym),(length,height),0,0,180,(92,92,205),4)

            #draw left eye
            xle=int(x+0.3*w)
            yle=int(y+0.4*h)
            radius=int(0.1*w)
            pupil=int(0.05*w)                        
            
            cv2.circle(frame,(xle,yle),radius,(255,255,255),-1)
            cv2.circle(frame,(xle,yle),pupil,(0,0,0),-1)

            #draw right eye
            xre=int(x+0.7*w)
            yre=int(y+0.4*h)
            radius=int(0.1*w)
            pupil=int(0.05*w)                        
            
            cv2.circle(frame,(xre,yre),radius,(255,255,255),-1)
            cv2.circle(frame,(xre,yre),pupil,(0,0,0),-1)

            #draw left ear
            xloe=int(x)
            yloe=int(y)
            xlie=int(x+w*0.05)
            ylie=int(y+h*0.05)
            
            cv2.circle(frame,(xloe,yloe),(w/5),(92,92,205),-1)
            cv2.circle(frame,(xlie,ylie),(w/8),(203,192,255),-1)

            #draw right ear
            xroe=int(x+w)
            yroe=int(y)
            xrie=int(x+w*0.95)
            yrie=int(y+h*0.05)
            
            cv2.circle(frame,(xroe,yroe),(w/5),(92,92,205),-1)
            cv2.circle(frame,(xrie,yrie),(w/8),(203,192,255),-1)

            #draw whiskers
            xc=int(x+w/2)
            yc=int(y+h/2)

            xl1=int(x+w/3)
            xl2=int(x+w/3)
            xl3=int(x+w/3)

            xr1=int(x+2*w/3)
            xr2=int(x+2*w/3)
            xr3=int(x+2*w/3)
            
            y1=int(y+5*h/10)
            y2=int(y+6*h/10)
            y3=int(y+7*h/10)

            cv2.line(frame,(xc,yc),(xl1,y1),(0,0,0),2)
            cv2.line(frame,(xc,yc),(xl2,y2),(0,0,0),2)
            cv2.line(frame,(xc,yc),(xl3,y3),(0,0,0),2)

            cv2.line(frame,(xc,yc),(xr1,y1),(0,0,0),2)
            cv2.line(frame,(xc,yc),(xr2,y2),(0,0,0),2)
            cv2.line(frame,(xc,yc),(xr3,y3),(0,0,0),2)

            #draw nose
            xn=int(x+w/2)
            yn=int(y+h/2)
            
            cv2.circle(frame,(xn,yn),(w/20),(203,192,255),-1)

        cv2.namedWindow('Faces found')
        cv2.imshow("Faces found", frame)

        if cv2.waitKey(1)> 10:

            videoLoop = False
            cv2.destroyWindow('Faces found')
            break

cam.release()
cv2.destroyAllWindows()
