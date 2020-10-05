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
            cv2.circle(frame,(x+w/2,y+w/2),(w/2),(63,133,205),-1)

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
            
            cv2.circle(frame,(xle,yle),radius,(50,205,154),-1)
            cv2.circle(frame,(xle,yle),pupil,(0,0,0),-1)

            #draw right eye
            xre=int(x+0.7*w)
            yre=int(y+0.4*h)
            radius=int(0.1*w)
            pupil=int(0.05*w)                        
            
            cv2.circle(frame,(xre,yre),radius,(50,205,154),-1)
            cv2.circle(frame,(xre,yre),pupil,(0,0,0),-1)

            #draw left ear
            xle1=int(x+int(w*0.05))
            yle1=int(y+int(h*0.3))
            xle2=int(x)
            yle2=int(y)
            xle3=int(x+int(w*0.3))
            yle3=int(y+int(h*0.05))
            
            pts = np.array([[xle1,yle1],[xle2,yle2],[xle3,yle3]], np.int32)
            pts = pts.reshape (-1,1,2)
            cv2.fillPoly(frame,[pts],(143,143,188))

            #draw right ear
            xre1=int(x+int(w*0.7))
            yre1=int(y+int(h*0.05))
            xre2=int(x+int(w))
            yre2=int(y)
            xre3=int(x+int(w*0.95))
            yre3=int(y+int(h*0.3))
            
            pts = np.array([[xre1,yre1],[xre2,yre2],[xre3,yre3]], np.int32)
            pts = pts.reshape (-1,1,2)
            cv2.fillPoly(frame,[pts],(143,143,188))

            #draw whiskers
            xc=int(x+w/2)
            yc=int(y+h/2)

            xl1=int(x+w/3)
            xl2=int(x+w/3)

            xr1=int(x+2*w/3)
            xr2=int(x+2*w/3)

            y1=int(y+6*h/10)
            y2=int(y+7*h/10)

            cv2.line(frame,(xc,yc),(xl1,y1),(0,0,0),2)
            cv2.line(frame,(xc,yc),(xl2,y2),(0,0,0),2)
            
            cv2.line(frame,(xc,yc),(xr1,y1),(0,0,0),2)
            cv2.line(frame,(xc,yc),(xr2,y2),(0,0,0),2)

            #draw nose
            xn1=int(x+int(w*0.4))
            yn1=int(y+int(h*0.45))
            xn2=int(x+int(w*0.6))
            yn2=int(y+int(h*0.45))
            xn3=int(x+int(w*0.5))
            yn3=int(y+int(h*0.6))
            
            pts = np.array([[xn1,yn1],[xn2,yn2],[xn3,yn3]], np.int32)
            pts = pts.reshape (-1,1,2)
            cv2.fillPoly(frame,[pts],(143,143,188))
            

        cv2.namedWindow('Faces found')
        cv2.imshow("Faces found", frame)

        if cv2.waitKey(1)> 10:

            videoLoop = False
            cv2.destroyWindow('Faces found')
            break

cam.release()
cv2.destroyAllWindows()
