from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.train(data="detectText.yaml", epochs=100, imgsz=640)
