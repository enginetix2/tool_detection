from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")  # Ensure this is the correct model file
    model.train(data="data.yaml", epochs=200, imgsz=640, workers=0, batch=4, save_period=50, patience=50, cache=False, amp=False)  # Set workers=0 for Windows
