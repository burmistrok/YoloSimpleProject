import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (use your own path or model type)
model = YOLO("yolov8n-seg.pt")  # You can also use yolov8s.pt, yolov8m.pt, etc.
print(list(model.names.values()))
# Open the video file or webcam
video_path = 'Demo/video2.mp4'  # Replace with 0 for webcam
cap = cv2.VideoCapture(video_path)

out_video_path = "Demo/video2_out.avi"
output = cv2.VideoWriter(
    out_video_path, cv2.VideoWriter_fourcc(*'MPEG'), 30, (1080, 1920))

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
 #   cv2.imshow("YOLOv8 Detection", annotated_frame)
    output.write(annotated_frame)
    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
output.release()
#cv2.destroyAllWindows()
