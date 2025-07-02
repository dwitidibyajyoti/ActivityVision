import cv2
from ultralytics import YOLO
import face_recognition
from face_faiss import load_face_data, build_faiss_index, match_face

# Load known face encodings and build a FAISS index for fast search
encodings, known_names = load_face_data("known_faces")
faiss_index = build_faiss_index(encodings)

# Set IP camera stream URL
stream_url = "http://192.168.31.103:8080/video"
cap = cv2.VideoCapture(stream_url)  # Open the video stream

# Load YOLOv8 nano model (lightweight and fast)
model = YOLO("yolov8n.pt")

# Exit if camera stream fails
if not cap.isOpened():
    print("❌ Failed to open video stream.")
    exit()

while True:
    ret, frame = cap.read()  # Read a frame from the stream
    if not ret:
        break  # Exit loop if frame is not read properly

    # Resize frame to 640x480 for faster YOLO processing
    resized_frame = cv2.resize(frame, (640, 480))

    # Run YOLO model to detect objects in the resized frame
    results = model(resized_frame)[0]

    # Loop through each detected bounding box
    for box in results.boxes:
        cls = int(box.cls[0])  # Get class ID
        if model.names[cls] == "person":  # Check if the detected object is a person
            # Get box coordinates in resized frame
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Scale box coordinates back to original frame size
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 480
            x1, y1, x2, y2 = [
                int(coord * scale) for coord, scale in zip((x1, y1, x2, y2), (scale_x, scale_y, scale_x, scale_y))
            ]

            # Crop the person region from the original frame
            person_crop = frame[y1:y2, x1:x2]

            # Convert cropped region from BGR to RGB (for face_recognition)
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            # Detect face locations in the cropped image
            face_locations = face_recognition.face_locations(rgb_crop)

            # Encode the detected faces
            face_encodings = face_recognition.face_encodings(
                rgb_crop, face_locations)

            # Loop over all detected faces in the crop
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                # Match face encoding with known encodings using FAISS
                name = match_face(encoding, faiss_index, known_names)

                # Draw face rectangle
                cv2.rectangle(frame, (x1 + left, y1 + top),
                              (x1 + right, y1 + bottom), (255, 0, 0), 2)

                # Label the face with name
                cv2.putText(frame, name, (x1 + left, y1 + top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Draw bounding box for the person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label the box with "Person"
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame with annotations
    cv2.imshow("IP Cam Person + Face Recognition", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources after loop ends
cap.release()
cv2.destroyAllWindows()
