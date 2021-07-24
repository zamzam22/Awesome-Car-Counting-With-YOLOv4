import cv2 as cv
import numpy as np
import time
import math

zz=25

class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 1


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h, idd = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
           
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 30:
                    self.center_points[id] = (cx, cy)
                    #print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break
            if(idd == 1):
                # New object is detected we assign the ID to that object
                if same_object_detected is False:
                    self.center_points[self.id_count] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, self.id_count])
                    self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
    
    #Function from cvzone
    def fancyDraw(self, img, bbox, l=zz, t=3, rt= 1):
        x, y, w, h, id = bbox
        
        x1, y1 = x + w, y + h
    
        cv.rectangle(img, (x,y),(x+w,y+h), (255, 0, 0), rt)
        # Top Left  x,y
        cv.line(img, (x, y), (x + l, y), (255, 0, 0), t)
        cv.line(img, (x, y), (x, y+l), (255, 0, 0), t)
        # Top Right  x1,y
        cv.line(img, (x1, y), (x1 - l, y), (255, 0, 0), t)
        cv.line(img, (x1, y), (x1, y+l), (255, 0, 0), t)
        # Bottom Left  x,y1
        cv.line(img, (x, y1), (x + l, y1), (255, 0, 0), t)
        cv.line(img, (x, y1), (x, y1 - l), (255, 0, 0), t)
        # Bottom Right  x1,y1
        cv.line(img, (x1, y1), (x1 - l, y1), (255, 0, 0), t)
        cv.line(img, (x1, y1), (x1, y1 - l), (255, 0, 0), t)
        return img

time.sleep(3)
    
Toplam_filtered =[]

# Create tracker object
tracker = EuclideanDistTracker()

# Load Yolo
net = cv.dnn.readNet("yolov4-tiny_best.weights", "yolov4-tiny.cfg")

net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
print(cv.cuda.getCudaEnabledDeviceCount())   
      
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Loading video
cap = cv.VideoCapture("test2.mp4")
cap.set(3, 1280) # set video widht
cap.set(4, 720) # set video height

font = cv.FONT_HERSHEY_COMPLEX_SMALL
starting_time = time.time()
frame_id = 0
toplam = 0 
say=0

#Start
while True:
    # Get frame
    _, frame = cap.read()
   
    
    frame_id += 1
    if frame_id % 1 != 0:
        continue
    
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv.dnn.blobFromImage(frame, 0.00261, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    outs = net.forward(output_layers)
    
    result = []
    boxes = []
    liste = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                if frame_id < 20:
                    xx=240
                else:
                    xx=350
                yy=600
                
                if xx < center_y < yy and w > 20:
                    idd=1
                else:
                    idd=0
                    
                boxes.append([x, y, w, h, idd])
        
    boxes_ids = tracker.update(boxes)

    
    #Filtering some 
    eklenecekler = []
    filtered =[]
   
    for i in range(len(boxes_ids)):
        if boxes_ids[i][4] not in eklenecekler:
            eklenecekler.append(boxes_ids[i][4])
            filtered.append(boxes_ids[i])
            
    for box_id in filtered:
            x, y, w, h, id = box_id
           
            if id > 99:
                zz=33
                
            cv.rectangle(frame, (x,y+h-20), (x+zz,y+h), (0,0,255),-1) #köşe kırmızı kare
            cv.putText(frame, str(id), (x+3 , y + h-5), cv.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1) # count yazısı
            #cv.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 3) #araç kareleri bgr
            frame = tracker.fancyDraw(frame,box_id)
   
          
    Toplam_filtered = [*Toplam_filtered, *filtered]

    # For fps
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    
    
    #total car count
    try:
        _,_,_,_,value = max(filtered, key=lambda item: item[4])
        if value > toplam:
            toplam = value
    except:
        pass
     
  
    #Some texts
    #cv.putText(frame, "FPS: " + str(round(fps, 2)), (5, 105), font, 1, (0, 0, 255), 2)
    cv.putText(frame, "NAZIM CORAKLI", (600, 70), font, 1, (0, 0, 255), 2)
    cv.putText(frame, str(toplam), (1150, 70), font, 2, (0, 0, 255), 2)
    
    #Final
    cv.imshow("Awesome Car Counting", frame)
    
    # Press 'ESC' for exiting video
    k = cv.waitKey(1) & 0xff 
    if k == 27:
        break
    if k == ord('p'):
        cv.waitKey(-1) #wait until any key is pressed

cap.release()
cv.destroyAllWindows()