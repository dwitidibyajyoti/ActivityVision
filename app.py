import cv2
from ultralytics import YOLO
import face_recognition
from face_faiss import load_face_data, build_faiss_index, match_face

# Load known face encodings from the "known_faces" folder
# and build a FAISS index for fast similarity search
encodings, known_names = load_face_data("known_faces")
faiss_index = build_faiss_index(encodings)

# Replace with your actual IP camera stream URL
stream_url = "http://192.168.2.191:8080/video"

# Load the YOLOv8 model for person detection
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(stream_url)

# Check if the video stream opened successfully
if not cap.isOpened():
    print("Failed to open video stream.")
    exit()

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection using YOLO
    results = model(frame)[0]

    # Loop through all detected objects
    for box in results.boxes:
        cls = int(box.cls[0])  # Get class ID
        if model.names[cls] == "person":
            # Extract the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop the region of the person from the frame
            person_crop = frame[y1:y2, x1:x2]

            # Convert cropped frame to RGB (required by face_recognition)
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            # Detect face(s) and compute face encoding(s) in the cropped area
            face_locations = face_recognition.face_locations(rgb_crop)
            face_encodings = face_recognition.face_encodings(
                rgb_crop, face_locations)

            # Loop through each detected face
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                # Find the best match for the encoding from the FAISS index
                name = match_face(encoding, faiss_index, known_names)

                # Draw a rectangle around the detected face (adjusted to full frame)
                cv2.rectangle(frame, (x1+left, y1+top),
                              (x1+right, y1+bottom), (255, 0, 0), 2)
                # Put the matched name above the face
                cv2.putText(frame, name, (x1+left, y1+top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Draw a green rectangle around the detected person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Label the person box as "Person"
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the final frame with annotations
    cv2.imshow("IP Cam Person + Face Recognition", frame)

    # Press 'q' to quit the stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
