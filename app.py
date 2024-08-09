import sys
import cv2 
import imutils
from yoloDet import YoloTRT

# use path for library and engine file
model = YoloTRT(library="yolov7/build/libmyplugins.so", engine="yolov7/build/best.engine", conf=0.2, yolo_ver="v7")

cap = cv2.VideoCapture("v2.mp4")


while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=640)
    detections, t = model.Inference(frame)
    # for obj in detections:
    #    print(obj['class'], obj['conf'], obj['box'])
    # print("FPS: {} sec".format(1/t))
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
