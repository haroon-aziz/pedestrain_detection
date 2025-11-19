import cv2
from ultralytics import YOLO
import os

# 1. Load the YOLO model
print("Loading model...")
model = YOLO('yolov8m.pt')

# 2. Path to video
video_path = "here enter your videio.mp4"  # Make sure this matches your file name exactly!
output_path = "output_detected.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video at {video_path}")
else:
    # GET VIDEO PROPERTIES
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # DEFINE THE WRITER
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video... Saving to {output_path}")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # 3. Run detection
        results = model.predict(frame, conf=0.4, classes=[0, 2], verbose=False
        # 4. Draw the bounding boxes
        annotated_frame = results[0].plot()

        # 5. Write the frame to the output video
        out.write(annotated_frame)

        # Print progress every 30 frames
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    
    # 6. Release everything to save the file properly
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done! Video saved successfully.")
