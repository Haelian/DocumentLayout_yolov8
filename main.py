from ultralytics import YOLO

img = cv2.imread("D:\\Document_Layout\\1. VAB - BCB niem yet-028.png", cv2.IMREAD_COLOR)
model = YOLO("D:\\Document_Layout\\yolov8n-doclaynet.pt")
result = model.predict(img)[0]
print(result)