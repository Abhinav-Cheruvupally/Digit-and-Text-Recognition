import numpy as np
import cv2
import pickle
import cv2
import numpy as np
import requests
import io
import json




def textrecog():
    
    img= cv2.imread(input("enter the image name with extension"))

    #shaping image
    height, width, _= img.shape
    roi= img[0:height, 0:  width]


    #OCR

    url_api = "https://api.ocr.space/parse/image"
    _, compressedimage= cv2.imencode(".jpg",roi,[1,90])
    file_bytes=io.BytesIO(compressedimage)


    result=requests.post(url_api, files={"screenshot.jpg": file_bytes},
                  data={"apikey":"your api key",
                        "language":"eng"})

    result= result.content.decode()
    result= json.loads(result)

    text_detected = result.get("ParsedResults")[0].get("ParsedText")
    print(text_detected)

    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    start()








def digitrecog():
    ########### PARAMETERS ##############
    width = 640
    height = 480
    threshold = 0.65 # MINIMUM PROBABILITY TO CLASSIFY
    cameraNo = 1
    #####################################

    #### CREATE CAMERA OBJECT
    cap = cv2.VideoCapture(0)
    cap.set(3,width)
    cap.set(4,height)

    #### LOAD THE TRAINNED MODEL 
    pickle_in = open("model_trained.p","rb")
    model = pickle.load(pickle_in)

    #### PREPORCESSING FUNCTION
    def preProcessing(img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img/255
        return img

    while True:
        success, imgOriginal = cap.read()
        img = np.asarray(imgOriginal)
        img = cv2.resize(img,(32,32))
        img = preProcessing(img)
        cv2.imshow("Processsed Image",img)
        img = img.reshape(1,32,32,1)
        #### PREDICT
        classIndex = int(model.predict_classes(img))
        #print(classIndex)
        predictions = model.predict(img)
        #print(predictions)
        probVal= np.amax(predictions)
        print(classIndex,probVal)

        if probVal> threshold:
            cv2.putText(imgOriginal,str(classIndex) + "   "+str(probVal),
                        (50,50),cv2.FONT_HERSHEY_COMPLEX,
                        1,(0,0,255),1)

        cv2.imshow("Original Image",imgOriginal)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            start()



def start():
    print("DIGIT AND TEXT RECOGNITION USING CNN AND OCR")
    print("---------------------------------------------")
    print("1.Digit Recognition | 2. Text Recognition")
    n=int(input("enter your choice"))
    if n==1:
        print("DIGIT RECOGNITION USING CNN")
        print("----------------------------")
        print("PLACE THE DIGIT IMAGE IN FRONT OF CAMERA")
        digitrecog()
    elif n==2:
        print("TEXT RECONOTION USING OCR API")
        print("GIVE IMAGE FOR TEXT RECOGNITION")
        textrecog()
    else:
        print("exiting......")

start()
