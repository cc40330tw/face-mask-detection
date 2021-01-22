'''
                    Face Mask Detection
               Code contributed by Carol Chen
                        2021_01_22
'''

import numpy as np
import cv2
import dlib
import os
facePath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(facePath)

bw_threshold = 80
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
weared_mask_font_color = (0, 255, 0)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
weared_mask = "Thank You for wearing mask"
cap = cv2.VideoCapture(0)


#取得預設的臉部偵測器
detector = dlib.get_frontal_face_detector()
#根據shape_predictor方法載入68個特徵點模型，此方法為人臉表情識別的偵測器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


while True:
    ret, img = cap.read()
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)
    if(len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    elif(len(faces) == 0 and len(faces_bw) == 1):
        cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
 
    
    #偵測人臉
    face_rects, scores, idx = detector.run(img, 0)

    #取出偵測的結果
    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        text = " %2.2f ( %d )" % (scores[i], idx[i])

        #繪製出偵測人臉的矩形範圍
        #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2. LINE_AA)

        #標上人臉偵測分數與人臉方向子偵測器編號
        cv2.putText(img, text, (x1, y1), cv2. FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2. LINE_AA)
    
        #給68特徵點辨識取得一個轉換顏色的frame
        landmarks_frame = cv2.cvtColor(img, cv2. COLOR_BGR2RGB)

        #找出特徵點位置
        shape = predictor(landmarks_frame, d)
    
        #繪製68個特徵點
        for i in range(68):
            cv2.circle(img,(shape.part(i).x,shape.part(i).y), 3,(0, 0, 255), 2)
            #cv2.putText(img, str(i),(shape.part(i).x,shape.part(i).y),cv2. FONT_HERSHEY_COMPLEX, 0.5,(255, 0, 0), 1)
            if 48<= i <= 61:
                #cv2.putText(img, TEXT, (x,y), font, scale, color) 在圖上面疊字
                cv2.putText(img,"Please wear your mask!",(50,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)


    cv2.imshow('Mask Detection', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
