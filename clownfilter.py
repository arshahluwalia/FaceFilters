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
            cv2.circle(frame,(x+w/2,y+w/2),(w/2),(255,255,255),-1)

            xoffset = int(w*0.75)
            yoffset = int(h*0.25)

            #draw mouth
            xm=int(x+w/2)
            ym=int(y+yoffset+w*0.4)
            
            length=int(w/4)
            height=int(yoffset)
            
            cv2.ellipse(frame,(xm,ym),(length,height),0,0,180,(92,92,205),-1)

            #draw left eye
            xle=int(x+0.3*w)
            yle=int(y+0.4*h)
            radius=int(0.1*w)
            pupil=int(0.05*w)                        
            
            cv2.circle(frame,(xle,yle),radius,(205,232,238),-1)
            cv2.circle(frame,(xle,yle),pupil,(42,42,165),-1)

            #draw right eye
            xre=int(x+0.7*w)
            yre=int(y+0.4*h)
            radius=int(0.1*w)
            pupil=int(0.05*w)                        
            
            cv2.circle(frame,(xre,yre),radius,(205,232,238),-1)
            cv2.circle(frame,(xre,yre),pupil,(42,42,165),-1)

            #draw nose
            xn=int(x+w/2)
            yn=int(y+h/2)
            
            cv2.circle(frame,(xn,yn),(w/10),(0,0,255),-1)

            #draw hat
            xh1=int(x+int(w*0.25))
            yh1=int(y+int(h*0.1))
            xh2=int(x+int(w*0.5))
            yh2=int(y-int(h*0.4))
            xh3=int(x+int(w*0.75))
            yh3=int(y+int(h*0.1))
            
            pts = np.array([[xh1,yh1],[xh2,yh2],[xh3,yh3]], np.int32)
            pts = pts.reshape (-1,1,2)
            cv2.fillPoly(frame,[pts],(211,0,148))

            #draw hair
            #left
            xlh=int(x+0.2*w)
            ylh=int(y+0.1*h)
            radius=int(0.15*w)                        
                
            cv2.circle(frame,(xlh,ylh),radius,(212,255,127),-1)

            #right
            xrh=int(x+0.8*w)
            yrh=int(y+0.1*h)
            radius=int(0.15*w)                        
                
            cv2.circle(frame,(xrh,yrh),radius,(0,165,255),-1)

            #left middle
            xlmh=int(x+0.35*w)
            ylmh=int(y+0.0*h)
            radius=int(0.15*w)                        
                
            cv2.circle(frame,(xlmh,ylmh),radius,(0,255,127),-1)

            #right middle
            xrmh=int(x+0.65*w)
            yrmh=int(y+0.07*h)
            radius=int(0.15*w)                        
                
            cv2.circle(frame,(xrmh,yrmh),radius,(240,32,160),-1)

            #middle 
            xmh=int(x+0.5*w)
            ymh=int(y+0.05*h)
            radius=int(0.15*w)                        
                
            cv2.circle(frame,(xmh,ymh),radius,(0,0,255),-1)

            

        cv2.namedWindow('Faces found')
        cv2.imshow("Faces found", frame)

        if cv2.waitKey(1)> 10:

            videoLoop = False
            cv2.destroyWindow('Faces found')
            break

cam.release()
cv2.destroyAllWindows()
