import cv2
import argparse
from ultralytics import YOLOv10

# Set up argument parser
parser = argparse.ArgumentParser(description="Light Spot Detection using YOLOv10")
parser.add_argument('--video', type=str, default=None, help='Path to video file')
args = parser.parse_args()

# Load the YOLOv10 model
model = YOLOv10('models/best.pt')

# Define a video capture object
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(f"Failed to load the {'video' if args.video else 'webcam'}")
    exit()

# Start the video capture loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform detection on the current frame
    results = model(
        source=frame,
        show=False,
        conf=0.3,
        save=False,
    )

    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    target_center_x, target_center_y = None, None

    predictions = []
    for box, cls, conf in zip(boxes, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        width = x2 - x1
        height = y2 - y1
        target_center_x = x1 + width // 2
        target_center_y = y1 + height // 2
        prediction = {
            "x": target_center_x,
            "y": target_center_y,
            "width": width,
            "height": height,
            "confidence": conf,
            "class": names[int(cls)],
            "class_id": int(cls),
            "detection_id": "your_generated_id",
        }
        predictions.append(prediction)

    # Convert frame to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    light_spot_center_x, light_spot_center_y = None, None

    # Threshold the image to find bright spots
    _, thresholded = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area > 100:
            cv2.drawContours(
                frame,
                [largest_contour],
                -1,
                (0, 255, 0),
                2,
            )
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                light_spot_center_x = cX
                light_spot_center_y = cY
                cv2.circle(
                    frame,
                    (light_spot_center_x, light_spot_center_y),
                    5,
                    (255, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    "Light Spot",
                    (light_spot_center_x, light_spot_center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
        else:
            print(
                "No light spots detected within bounding box area extended downwards."
            )

    # Draw the target center and bounding box on the frame
    if target_center_x is not None and target_center_y is not None:
        cv2.circle(frame, (target_center_x, target_center_y), 5, (0, 0, 255), -1)
        if 'x1' in locals() and 'y1' in locals() and 'x2' in locals():
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        
        # Calculate and display the distance between the centers
        if light_spot_center_x is not None and light_spot_center_y is not None:
            distance_x_px = light_spot_center_x - target_center_x
            distance_y_px = target_center_y - light_spot_center_y

            distance_x_cm = distance_x_px * 100 / width
            distance_y_cm = distance_y_px * 100 / height

            text = f"Distance between centers in cm: ({distance_x_cm:.2f}, {distance_y_cm:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1

            (h, w) = frame.shape[:2]
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, font_thickness
            )
            x = (w - text_width) // 2
            y = text_height + 10

            cv2.putText(frame, text, (x, y), font, font_scale, (0, 128, 0), font_thickness)

    # Display the frame with detections
    cv2.imshow("Video with Light Spot Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# After the loop, release the capture object and close windows
cap.release()
cv2.destroyAllWindows()
