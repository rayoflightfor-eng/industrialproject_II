import cv2 
from ultralytics import YOLO 
 
# Load trained model 
model = YOLO("runs/aluminium_defect_model/weights/best.pt") 
 
# Load image 
image_path = "test_image.jpg" 
img = cv2.imread(image_path) 
 
# Run detection 
results = model(img) 
 
# Draw results 
for result in results: 
    boxes = result.boxes 
    for box in boxes: 
        x1, y1, x2, y2 = map(int, box.xyxy[0]) 
        label = model.names[int(box.cls[0])] 
        confidence = box.conf[0] 
 
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) 
        cv2.putText( 
            img, 
            f"{label} {confidence:.2f}", 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 0), 
            2 
        ) 
 
# Show output 
cv2.imshow("Aluminium Sheet Defect Detection", img) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 
