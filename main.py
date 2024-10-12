# Refactored code to detect and track vehicles with YOLOv8

from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    """
    Main function to load the YOLOv8 model, read video frames, 
    and track vehicles in the video.
    """
    # Load the YOLOv8 model
    model = YOLO("your-model-path")

    # Open the video file
    video_path = "your-video-path"
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Initialize a dictionary to keep track of vehicle history
    track_history = defaultdict(list)
    frame_count = 0

    # Process video frames until the end of the video
    while cap.isOpened():
        success, frame = cap.read()
        frame_count += 1  # Increment frame count

        # Break the loop if there are no frames to read
        if not success:
            break

        # Perform tracking on the current frame
        results = model.track(frame, persist=True)

        # Process the results and annotate the frame
        if results and hasattr(results[0], 'boxes') and results[0].boxes:
            annotated_frame = process_frame(results, track_history)
        else:
            # If no results, use the original frame
            annotated_frame = frame.copy()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    # Release video capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def process_frame(results, track_history):
    """
    Process the detection results and annotate the frame with tracking IDs.

    Args:
        results: The detection results from the YOLO model.
        track_history: A dictionary that tracks the history of detected vehicles.

    Returns:
        Annotated frame with tracking IDs.
    """
    boxes = results[0].boxes

    # Check if the boxes have necessary attributes for processing
    if hasattr(boxes, 'xywh') and hasattr(boxes, 'id') and boxes.id is not None:
        annotated_frame = results[0].plot()
        track_ids = boxes.id.int().cpu().tolist()  # Get tracking IDs
        boxes_xywh = boxes.xywh.cpu().numpy()      # Get bounding box coordinates
        class_names = [results[0].names[i] for i in boxes.cls.int().cpu().tolist()]

        printed_ids = set()  # Track printed IDs in the current frame

        # Annotate the frame with bounding boxes and IDs
        for box, track_id, class_name in zip(boxes_xywh, track_ids, class_names):
            x, y, w, h = box
            track_history[track_id].append((x, y))  # Update tracking history

            if track_id not in printed_ids:
                printed_ids.add(track_id)  # Mark this ID as printed
    else:
        # If boxes are not available, use the original image
        annotated_frame = results[0].orig_img.copy()

    return annotated_frame

if __name__ == "__main__":
    main()  # Execute the main function
