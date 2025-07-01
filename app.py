import cv2
from ultralytics import YOLO

# Replace with your IP camera stream URL
# e.g., http://192.168.1.5:8080/video
stream_url = "http://192.168.2.191:8080/video"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Failed to open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("IP Cam Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
