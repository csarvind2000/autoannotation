import gradio as gr
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import zipfile
import requests
import tempfile
import pandas as pd
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom class names (fixed, no new labels allowed)
CLASS_NAMES = [
    "Aircraft", "Person", "Baggage Truck", "Ramp Loader", "Bus",
    "Fuel Truck", "Tank Hose", "Ground Power Unit", "Stairway",
    "Rolling Stairway", "Car"
]

# Color map for each class (RGB)
CLASS_COLORS = [
    (255, 0, 0),    # Aircraft: Red
    (0, 255, 0),    # Person: Green
    (0, 0, 255),    # Baggage Truck: Blue
    (255, 255, 0),  # Ramp Loader: Yellow
    (255, 0, 255),  # Bus: Magenta
    (0, 255, 255),  # Fuel Truck: Cyan
    (128, 0, 0),    # Tank Hose: Dark Red
    (0, 128, 0),    # Ground Power Unit: Dark Green
    (0, 0, 128),    # Stairway: Dark Blue
    (128, 128, 0),  # Rolling Stairway: Olive
    (128, 0, 128)   # Car: Purple
]

# Download and load YOLO model
def load_model():
    model_path = "Yolov10n-trained-best.pt"
    if not os.path.exists(model_path):
        url = "https://raw.githubusercontent.com/Gaikwadabhi/Event-Managment-/main/Yolov10n-trained-best.pt"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Model downloaded to {model_path}")
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            return None, f"Error downloading model: {str(e)}"
    try:
        model = YOLO(model_path)
        logger.info("Model loaded successfully")
        return model, None
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, f"Error loading model: {str(e)}"

# Convert YOLO predictions to annotation format
def predictions_to_annotations(results, img_width, img_height):
    annotations = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls)
            x, y, w, h = box.xywhn[0].tolist()  # Normalized xywh
            annotations.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return annotations

# Draw bounding boxes on image
def draw_bboxes(image, results):
    img = np.array(image)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x, y, w, h = box.xywh[0].tolist()  # Absolute xywh
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            cls_id = int(box.cls)
            label = CLASS_NAMES[cls_id]
            color = CLASS_COLORS[cls_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return Image.fromarray(img)

# Extract frames from video with skip interval
def extract_video_frames(video_path, skip_interval=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    frame_idx = 0
    while cap.isOpened() and frame_count < 100:  # Limit to 100 frames
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip_interval == 0:  # Process frame if it matches the skip interval
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            frame_count += 1
        frame_idx += 1
    cap.release()
    logger.info(f"Extracted {frame_count} frames from {video_path} with skip interval {skip_interval}")
    return frames

# Process inputs (images or video)
def auto_annotate(input_files, conf_threshold=0.25, skip_interval=1):
    model, error = load_model()
    if not model:
        return None, None, None, None, error

    images = []
    image_names = []
    annotation_files = []
    output_images = []
    results_list = []
    stats_data = []
    output_dir = "dataset"
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    # Handle input files
    for input_file in input_files:
        if input_file.name.lower().endswith(('.mp4', '.avi', '.mov')):
            frames = extract_video_frames(input_file.name, skip_interval)
            for i, frame in enumerate(frames):
                images.append(frame)
                image_names.append(f"frame_{i*skip_interval:04d}.jpg")  # Name reflects frame index
        else:
            img = Image.open(input_file)
            images.append(img)
            image_names.append(os.path.basename(input_file.name))

    # Process each image
    for img, img_name in zip(images, image_names):
        img_width, img_height = img.size
        results = model.predict(img, conf=conf_threshold)
        results_list.append(results)

        # Generate annotations
        annotations = predictions_to_annotations(results, img_width, img_height)
        img_base_name = os.path.splitext(img_name)[0]
        annotation_path = os.path.join(output_dir, "labels", f"{img_base_name}.txt")
        with open(annotation_path, "w") as f:
            f.write("\n".join(annotations))
        annotation_files.append(annotation_path)

        # Save annotated image
        output_img = draw_bboxes(img, results)
        output_images.append(output_img)
        output_img.save(os.path.join(output_dir, "images", img_name))

        # Compute per-frame stats
        stats = {cls: 0 for cls in CLASS_NAMES}
        total = 0
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                stats[CLASS_NAMES[cls_id]] += 1
                total += 1
        stats_row = {"Frame": img_name, **stats, "Total": total}
        stats_data.append(stats_row)

    # Create data.yaml for Roboflow
    data_yaml = {
        "train": os.path.join(output_dir, "images"),
        "val": os.path.join(output_dir, "images"),
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f)

    # Create ZIP file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
        with zipfile.ZipFile(temp_zip.name, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zip_file.write(file_path, os.path.join("dataset", arcname))
        zip_path = temp_zip.name

    stats_df = pd.DataFrame(stats_data)
    logger.info("Annotations generated successfully")
    return output_images, results_list, zip_path, stats_df, image_names, "Annotations generated successfully!"

# Navigate images
def navigate_images(direction, current_index, output_images, results_list, image_names):
    if not output_images:
        return current_index, None, "No images to display."
    new_index = current_index + (1 if direction == "next" else -1)
    new_index = max(0, min(new_index, len(output_images) - 1))
    current_image = output_images[new_index]
    current_results = results_list[new_index]
    annotated_image = draw_bboxes(current_image, current_results)
    return new_index, annotated_image, f"Image {new_index + 1} of {len(output_images)}"

# Gradio interface
def create_gradio_app():
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray")
    with gr.Blocks(theme=theme, title="YOLOv10 Auto-Annotation") as app:
        gr.Markdown(
            """
            # Auto-Annotation App
            Upload images or a video to generate YOLO annotations. Navigate through images and view per-frame statistics.
            """
        )

        with gr.Tabs():
            with gr.Tab("Annotation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_files = gr.File(
                            label="Upload Images or Video",
                            file_count="multiple",
                            file_types=[".jpg", ".png", ".mp4", ".avi", ".mov"]
                        )
                        conf_threshold = gr.Slider(
                            0.1, 1.0, value=0.25, label="Confidence Threshold"
                        )
                        skip_interval = gr.Number(
                            label="Process Every Nth Frame (1 = every frame)",
                            value=1,
                            minimum=1,
                            maximum=100,
                            step=1,
                            precision=0
                        )
                        status = gr.Textbox(label="Status")
                        annotate_btn = gr.Button("Generate Annotations", variant="primary")
                    with gr.Column(scale=4):
                        current_image = gr.Image(
                            label="Current Annotated Image",
                            height=800,
                            interactive=False  # No sketching
                        )
                        with gr.Row():
                            prev_btn = gr.Button("Previous", variant="secondary")
                            next_btn = gr.Button("Next", variant="secondary")
                        image_index = gr.State(value=0)
                        results_list = gr.State(value=[])
                        output_images = gr.State(value=[])
                        is_video = gr.State(value=False)
                        image_names = gr.State(value=[])

                output_zip = gr.File(label="Download Dataset (ZIP)")

            with gr.Tab("Statistics"):
                stats_table = gr.Dataframe(label="Per-Frame Detection Statistics")

        # Event handlers
        def handle_annotate(input_files, conf_threshold, skip_interval):
            if not input_files:
                return None, None, None, None, [], "No files uploaded."
            image_names.value = []
            is_video.value = False
            skip_interval = max(1, int(skip_interval))  # Ensure valid integer >= 1
            for input_file in input_files:
                if input_file.name.lower().endswith(('.mp4', '.avi', '.mov')):
                    frames = extract_video_frames(input_file.name, skip_interval)
                    image_names.value.extend([f"frame_{i*skip_interval:04d}.jpg" for i in range(len(frames))])
                    is_video.value = True
                else:
                    image_names.value.append(os.path.basename(input_file.name))
            return auto_annotate(input_files, conf_threshold, skip_interval)

        annotate_btn.click(
            fn=handle_annotate,
            inputs=[input_files, conf_threshold, skip_interval],
            outputs=[output_images, results_list, output_zip, stats_table, image_names, status]
        ).then(
            fn=lambda output_images, results_list, image_names: navigate_images(
                "next", -1, output_images, results_list, image_names
            ),
            inputs=[output_images, results_list, image_names],
            outputs=[image_index, current_image, status]
        )

        prev_btn.click(
            fn=navigate_images,
            inputs=[gr.State(value="prev"), image_index, output_images, results_list, image_names],
            outputs=[image_index, current_image, status]
        )

        next_btn.click(
            fn=navigate_images,
            inputs=[gr.State(value="next"), image_index, output_images, results_list, image_names],
            outputs=[image_index, current_image, status]
        )

    return app

# Launch the app
if __name__ == "__main__":
    logger.info(f"Gradio version: {gr.__version__}")
    app = create_gradio_app()
    app.launch()