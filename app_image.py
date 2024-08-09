import cv2 
import imutils
from yoloDet import YoloTRT

# use path for library and engine file
model = YoloTRT(library="yolov7/build/libmyplugins.so", engine="yolov7/build/best.engine", conf=0.2, yolo_ver="v7")

# Load the image directly using cv2.imread
image_path = "im3.jpg"
frame = cv2.imread(image_path)
if frame is None:
    print(f"Error: No se pudo cargar la imagen '{image_path}'")
    sys.exit()

# Resize the frame if needed
frame = imutils.resize(frame, width=640)

# Perform inference on the resized frame
detections, t = model.Inference(frame)

# Display the output
cv2.imshow("Output", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

