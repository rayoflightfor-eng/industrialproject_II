from ultralytics import YOLO 
 
def train_model(): 
    # Load pre-trained YOLOv8 model 
    model = YOLO("yolov8n.pt") 
 
    # Train the model 
    model.train( 
        data="dataset.yaml", 
        epochs=50, 
        imgsz=640, 
        batch=16, 
        name="aluminium_defect_model", 
        project="runs" 
    ) 
 
if __name__ == "__main__": 
    train_model()
