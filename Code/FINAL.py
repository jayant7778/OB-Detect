import cv2
import matplotlib.pyplot as plt
from utils import *
from darknet import Darknet


#weight file for ob detect
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model,config_file)
classLabels = []
file_name = 'Labels.txt'
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)
model.setInputSize(320,320)
model.setInputScale(1.0/127.5) ##255/2=127.5
model.setInputMean((127.5,127.5,127.5)) #mobilenet=>[-1,1]
model.setInputSwapRB(True)

#helmet detect
cfg_file = 'model/yolov3-obj.cfg'
weight_file = 'model/yolov3-obj_2400.weights'
namesfile = 'model/obj.names'

m = Darknet(cfg_file)
m.load_weights(weight_file)
class_names = load_class_names(namesfile)


img = cv2.imread('india_Motorcycle_20130322_0001.jpg')
img1 = img
print(m) #model layers

plt.imshow(img1)

#obdetect Code
ClassIndex, confidece, bbox = model.detect(img1,confThreshold=0.5)
font_scale = 1
font = cv2.FONT_HERSHEY_PLAIN
i=1
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
    cv2.rectangle(img1,boxes,(255, 0, 0),1)
    plt.imshow(img1)
    cv2.putText(img1,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale=font_scale,color=(0,0,255), thickness=1)
    print(classLabels[ClassInd-1])
    #print(i+1,boxes)
    if classLabels[ClassInd-1]=="car":
        print(i)
        i=i+1
plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))

#helmet detect code
plt.rcParams['figure.figsize'] = [24.0, 14.0]
original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
resized_image = cv2.resize(original_image, (m.width, m.height))
iou_thresh = 0.4
nms_thresh = 0.6
boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
print_objects(boxes, class_names)
plot_boxes(original_image, boxes, class_names, plot_labels = True)


#video input for ob detect
cap = cv2.VideoCapture("production ID_4740223.mp4")
#cap = cv2.VideoCapture(0) for Live capturing using system camera
if not cap.isOpened():
    print("hi")
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open Video")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret,frame = cap.read()
    
    ClassIndex, confidece, bbox = model.detect(frame,confThreshold=0.55)
    
    #print(ClassIndex)
    if (len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if(ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale=font_scale,color=(0,0,255), thickness=1)
    cv2.imshow('OB detect',frame)
    print(classLabels[ClassInd-1])
    #if classLabels[ClassInd-1]=="car":
        #print(i)
        #i=i+1
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


#video input for helmet detect
plt.rcParams['figure.figsize'] = [24.0, 14.0]
cap = cv2.VideoCapture("production ID_4740223.mp4")
#cap = cv2.VideoCapture(0) for system camera input
if not cap.isOpened():
    print("hi")
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open Video")

while True:
    ret,frame = cap.read()
    original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(original_image, (m.width, m.height))
    iou_thresh = 0.4
    nms_thresh = 0.6
    boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
    #print_objects(boxes, class_names)
    if detect_objects(m, resized_image, iou_thresh, nms_thresh):
        print("yes")
        plot_boxes(original_image, boxes, class_names, plot_labels = True)
    cv2.imshow('OB detect',frame)
    
    #if classLabels[ClassInd-1]=="car":
        #print(i)
        #i=i+1
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
#print_objects(boxes, class_names)



