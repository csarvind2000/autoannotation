# YOLOv10 Auto-Annotation App

some text

This project is a web-based application built with Gradio for automatic object detection and annotation using the YOLOv10 model. It allows users to upload images or videos, generate YOLO-compatible annotations, navigate through processed images/frames, view per-frame statistics, and download the annotated dataset as a ZIP file.

## Features
- **Upload Images or Videos**: Supports `.jpg`, `.png`, `.mp4`, `.avi`, and `.mov` formats.
- **Automatic Annotation**: Uses a pretrained YOLOv10 model to detect objects and generate annotations.
- **Frame Skipping for Videos**: Option to process every Nth frame to reduce processing time (default: every frame).
- **Navigation**: Browse through annotated images/frames with "Previous" and "Next" buttons.
- **Statistics**: View per-frame detection statistics in a table.
- **Export**: Download the annotated dataset (images, labels, and `data.yaml`) as a ZIP file.
- **Predefined Classes**: Detects objects in 11 fixed classes: Aircraft, Person, Baggage Truck, Ramp Loader, Bus, Fuel Truck, Tank Hose, Ground Power Unit, Stairway, Rolling Stairway, Car.

## Prerequisites
- Python 3.12 or higher
- Conda (recommended for environment management)
- Git

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/csarvind2000/autoannotation.git
   cd autoannotation
   ```

2. **Create a Conda Environment**:
   ```bash
   conda create -n YOLO python=3.12
   conda activate YOLO
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the YOLO Model**:
   The app automatically downloads the pretrained model (`Yolov10n-trained-best.pt`) from a GitHub URL on first run. If the download fails, manually download it from:
   ```
   https://raw.githubusercontent.com/Gaikwadabhi/Event-Managment-/main/Yolov10n-trained-best.pt
   ```
   Place the file in the project root directory.

## Usage

1. **Run the Application**:
   ```bash
   python app1.py
   ```
   This launches the Gradio web interface (typically at `http://127.0.0.1:7860`).

2. **Interact with the App**:
   - **Upload Files**: Select images (`.jpg`, `.png`) or videos (`.mp4`, `.avi`, `.mov`).
   - **Set Confidence Threshold**: Adjust the YOLO detection threshold (default: 0.25, range: 0.1–1.0).
   - **Set Frame Skip Interval**: For videos, specify how many frames to skip (e.g., `1` = every frame, `5` = every 5th frame, default: `1`).
   - **Generate Annotations**: Click "Generate Annotations" to process files and view results.
   - **Navigate**: Use "Previous" and "Next" buttons to browse annotated images/frames.
   - **View Statistics**: Check the "Statistics" tab for per-frame detection counts.
   - **Download Dataset**: Download the ZIP file containing annotated images, labels, and `data.yaml`.

## Project Structure
```
autoannotation/
├── app1.py               # Main application script
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── dataset/              # Output directory for images, labels, and data.yaml (created on run)
└── Yolov10n-trained-best.pt  # YOLO model file (downloaded automatically or manually)
```

## Notes
- **Frame Limit**: Video processing is limited to 100 processed frames to prevent excessive resource usage.
- **Output Format**: Annotations are in YOLO format (`.txt` files with class ID and normalized bounding box coordinates).
- **Model URL**: Ensure the model URL is accessible. If not, use a local model file.
- **Permissions**: Ensure write permissions for the `dataset` directory and ZIP file creation.

## Troubleshooting
- **Gradio Version Mismatch**:
  Verify Gradio version:
  ```bash
  pip show gradio
  ```
  Ensure it’s `5.27.0`. Reinstall if needed:
  ```bash
  pip uninstall gradio gradio-client
  pip install gradio==5.27.0
  ```

- **Model Download Failure**:
  If the model fails to download, manually place `Yolov10n-trained-best.pt` in the project root.

- **Video Processing Errors**:
  Ensure `opencv-python` is installed and the video format is supported. Check logs for errors.

- **Other Issues**:
  Run with:
  ```bash
  python app1.py
  ```
  Share the full error traceback with the repository maintainer.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For issues or questions, open an issue on the [GitHub repository](https://github.com/csarvind2000/autoannotation) or contact the maintainer.