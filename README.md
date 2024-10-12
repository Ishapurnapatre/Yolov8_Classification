# Vehicle Detection and Tracking using YOLOv8

This project implements vehicle detection and tracking using the YOLOv8 model. The code processes a video file to identify and track vehicles, classifying them into categories based on their appearance.

## Features

- Real-time vehicle detection and tracking
- Support for multiple vehicle classes
- Visual annotation of detected vehicles with tracking IDs

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- YOLOv8 (from Ultralytics)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Ishapurnapatre/Yolov8_Classification.git
    cd Yolov8_Classification
    ```

2. Set up a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your trained YOLOv8 model weights in the project directory.
2. Prepare your input video file and place it in the project directory.
3. Run the script:

    ```bash
    python main.py
    ```

4. The output will display a window showing the video with annotated vehicle detections and tracking IDs. Press 'q' to exit the video display.

## Code Overview

- **main.py**: The main script that loads the YOLOv8 model, reads video frames, and processes them for detection and tracking.
    - The `main()` function initializes the model and handles video input/output.
    - The `process_frame(results, track_history)` function processes detection results and annotates the video frames.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [YOLOv8](https://github.com/ultralytics/yolov8) for the object detection framework.
- OpenCV for computer vision functionalities.

