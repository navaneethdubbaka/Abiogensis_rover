import cv2
from ultralytics import YOLO

# Load your YOLO26n model (COCO pretrained)
model = YOLO("yolo26n.pt")  # this will auto-load the .pt file

# Open USB camera (index 0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Live window
cv2.namedWindow("YOLO26n Live", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO26n inference
    results = model(frame)

    # results contains detections; draw them
    for r in results:
        # r.boxes.xyxy: bounding boxes
        # r.boxes.conf: confidence scores
        # r.boxes.cls: class ids
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            score = float(conf)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {score:.2f}"
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # If this is a person, you can use it for robot control
            # For example get center and deviation
            # center_x = (x1 + x2) // 2
            # deviation = (center_x / frame.shape[1]) - 0.5

    # Show the frame
    cv2.imshow("YOLO26n Live", frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
