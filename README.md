# face-mask-detection

**<註>** 此project不含Keras與Tensorflow，
而是結合OpenCV的CascadeClassifier與dlib的68個臉部特徵

### ---前言---
因應近年來COVID-19的猖狂，科技結合醫療成為熱門話題。為了能及時偵測人們有沒有遵守戴口罩的規定，我們利用Real-time face detection配合螢幕上字幕的警告作為提醒。

### ---內文---
臉部偵測背後運用到Haar Cascade algorithm這個演算法。其中的xml檔是已經train好的model，不必自己蒐集大量圖片做training (需特別注意這邊的路徑)
```
facePath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(facePath)
```
dlib的detector與predictor設置
```
#取得預設的臉部偵測器
detector = dlib.get_frontal_face_detector()
#根據shape_predictor方法載入68個特徵點模型，此方法為人臉表情識別的偵測器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
```

以圓圈標記出臉部68個特徵點，可以選擇是否要在點上標記對應數字。48~61這個區間即是嘴巴的部位，若偵測得到代表沒有確實戴上口罩。
```
for i in range(68):
    cv2.circle(img(shape.part(i).x,shape.part(i).y), 3,(0, 0, 255), 2)
    #cv2.putText(img, str(i),(shape.part(i).x,shape.part(i).y),cv2. FONT_HERSHEY_COMPLEX, 0.5,(255, 0, 0), 1)
    if 48<= i <= 61:
        cv2.putText(img,"Please wear your mask!",(50,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
```

若希望將螢幕錄製儲存成檔案，可以透過 cv2.VideoWriter()。其中的第三個參數為 fps 影像偵率、
第四個參數為 frameSize 影像大小，
而最後的參數則代表是否要存成彩色，否則為灰階，其預設為 True。
```
#擷取網路攝影機串流影像
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
```

> [time=Sat, Jan 23, 2021 1:57 AM]
> [name=陳玥蓁]
> 
