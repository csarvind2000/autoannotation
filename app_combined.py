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
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom class names (fixed, no new labels allowed)
CLASS_NAMES = [
    "Aircraft", "Person", "Baggage Truck", "Ramp Loader", "Bus",
    "Fuel Truck", "Tank Hose", "Ground Power Unit", "Stairway",
    "Rolling Stairway", "Car"
]

# COCO classes to map to custom classes
COCO_TO_CUSTOM = {
    "car": "Car",
    "bus": "Bus",
    "airplane": "Aircraft",
    "person": "Person",
}

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
    (128, 0, 128),   # Car: Purple
]

# Download and load YOLO models
def load_models():
    # Load custom model
    custom_model_path = "Yolov10n-trained-best.pt"
    if not os.path.exists(custom_model_path):
        url = "https://raw.githubusercontent.com/Gaikwadabhi/Event-Managment-/main/Yolov10n-trained-best.pt"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(custom_model_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Custom model downloaded to {custom_model_path}")
        except Exception as e:
            logger.error(f"Error downloading custom model: {str(e)}")
            return None, None, f"Error downloading custom model: {str(e)}"

    # Load COCO model
    coco_model_path = "yolov10n.pt"
    if not os.path.exists(coco_model_path):
        try:
            # Ultralytics will automatically download yolov10n.pt if not present
            logger.info("Downloading COCO pretrained model yolov10n.pt")
        except Exception as e:
            logger.error(f"Error downloading COCO model: {str(e)}")
            return None, None, f"Error downloading COCO model: {str(e)}"

    try:
        custom_model = YOLO(custom_model_path)
        coco_model = YOLO(coco_model_path)
        logger.info("Both models loaded successfully")
        return custom_model, coco_model, None
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return None, None, f"Error loading models: {str(e)}"

# Compute IoU for two bounding boxes
def compute_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1 = max(x1_1, x1_2)
    y1 = max(y1_1, y1_2)
    x2 = min(x2_1, x2_2)
    y2 = min(y2_1, y2_2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

# Merge custom and COCO model predictions for annotations
def merge_predictions(custom_results, coco_results, img_width, img_height):
    annotations = []
    custom_boxes = []
    custom_count = 0
    coco_count = 0

    # Process custom model predictions (prioritized)
    for result in custom_results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls)
            x, y, w, h = box.xywhn[0].tolist()  # Normalized xywh
            annotations.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            # Store absolute coordinates for IoU comparison
            x1 = (x - w / 2) * img_width
            y1 = (y - h / 2) * img_height
            x2 = (x + w / 2) * img_width
            y2 = (y + h / 2) * img_height
            custom_boxes.append((x1, y1, x2, y2, cls_id))
            custom_count += 1

    # Process COCO model predictions (only non-overlapping detections added)
    for result in coco_results:
        boxes = result.boxes
        for box in boxes:
            coco_cls_name = result.names[int(box.cls)]
            if coco_cls_name not in COCO_TO_CUSTOM:
                continue  # Skip non-mapped COCO classes
            custom_cls_name = COCO_TO_CUSTOM[coco_cls_name]
            cls_id = CLASS_NAMES.index(custom_cls_name)
            x, y, w, h = box.xywhn[0].tolist()  # Normalized xywh
            # Check for overlap with custom model detections
            x1 = (x - w / 2) * img_width
            y1 = (y - h / 2) * img_height
            x2 = (x + w / 2) * img_width
            y2 = (y + h / 2) * img_height
            coco_box = (x1, y1, x2, y2)
            overlap = False
            for custom_box in custom_boxes:
                if compute_iou(coco_box, custom_box[:4]) > 0.5:  # IoU threshold
                    overlap = True
                    break
            if not overlap:
                annotations.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                coco_count += 1

    logger.info(f"Merged predictions: {custom_count} custom detections, {coco_count} COCO detections")
    return annotations

# Draw bounding boxes on image (custom + non-overlapping COCO detections)
def draw_bboxes(image, custom_results, coco_results, img_width, img_height):
    img = np.array(image)
    custom_boxes = []

    # Draw custom model detections (prioritized)
    for result in custom_results:
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
            # Store absolute coordinates for IoU comparison
            custom_boxes.append((x1, y1, x2, y2, cls_id))

    # Draw non-overlapping COCO model detections
    for result in coco_results:
        boxes = result.boxes
        for box in boxes:
            coco_cls_name = result.names[int(box.cls)]
            if coco_cls_name not in COCO_TO_CUSTOM:
                continue  # Skip non-mapped COCO classes
            custom_cls_name = COCO_TO_CUSTOM[coco_cls_name]
            cls_id = CLASS_NAMES.index(custom_cls_name)
            x, y, w, h = box.xywh[0].tolist()  # Absolute xywh
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            coco_box = (x1, y1, x2, y2)
            overlap = False
            for custom_box in custom_boxes:
                if compute_iou(coco_box, custom_box[:4]) > 0.5:  # IoU threshold
                    overlap = True
                    break
            if not overlap:
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
    custom_model, coco_model, error = load_models()
    if not custom_model or not coco_model:
        return None, None, None, None, error

    images = []
    image_names = []
    annotation_files = []
    output_images = []
    results_list = []
    stats_data = []

    # Create unique output directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("dataset", f"upload_{timestamp}")
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    # Handle input files
    for input_file in input_files:
        if input_file.name.lower().endswith(('.mp4', '.avi', '.mov')):
            frames = extract_video_frames(input_file.name, skip_interval)
            for i, frame in enumerate(frames):
                images.append(frame)
                image_names.append(f"frame_{i*skip_interval:04d}.jpg")
        else:
            img = Image.open(input_file)
            images.append(img)
            image_names.append(os.path.basename(input_file.name))

    # Process each image
    for img, img_name in zip(images, image_names):
        img_width, img_height = img.size
        # Run both models
        custom_results = custom_model.predict(img, conf=conf_threshold)
        coco_results = coco_model.predict(img, conf=conf_threshold)
        # Store both results for display (as a tuple)
        results_list.append((custom_results, coco_results, img_width, img_height))

        # Merge predictions for annotations
        annotations = merge_predictions(custom_results, coco_results, img_width, img_height)
        img_base_name = os.path.splitext(img_name)[0]
        annotation_path = os.path.join(output_dir, "labels", f"{img_base_name}.txt")
        with open(annotation_path, "w") as f:
            f.write("\n".join(annotations))
        annotation_files.append(annotation_path)

        # Save annotated image (using both custom and COCO results)
        output_img = draw_bboxes(img, custom_results, coco_results, img_width, img_height)
        output_images.append(output_img)
        output_img.save(os.path.join(output_dir, "images", img_name))

        # Compute per-frame stats (based on merged annotations)
        stats = {cls: 0 for cls in CLASS_NAMES}
        total = 0
        for ann in annotations:
            cls_id = int(ann.split()[0])
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
                    arcname = os.path.relpath(file_path, os.path.dirname(output_dir))
                    zip_file.write(file_path, arcname)
        zip_path = temp_zip.name

    stats_df = pd.DataFrame(stats_data)
    logger.info(f"Annotations generated successfully in {output_dir}")
    return output_images, results_list, zip_path, stats_df, image_names, "Annotations generated successfully!"

# Navigate images
def navigate_images(direction, current_index, output_images, results_list, image_names):
    if not output_images:
        return current_index, None, "No images to display."
    new_index = current_index + (1 if direction == "next" else -1)
    new_index = max(0, min(new_index, len(output_images) - 1))
    current_image = output_images[new_index]
    # Unpack the tuple: custom_results, coco_results, img_width, img_height
    custom_results, coco_results, img_width, img_height = results_list[new_index]
    annotated_image = draw_bboxes(current_image, custom_results, coco_results, img_width, img_height)
    return new_index, annotated_image, f"Image {new_index + 1} of {len(output_images)}"

# Gradio interface
def create_gradio_app():
    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="gray")
    with gr.Blocks(theme=theme, title="YOLOv10 Auto-Annotation") as app:
        gr.Markdown(
            """
            # YOLOv10 Auto-Annotation App
            Upload images or a video to generate YOLO annotations. Navigate through images and view per-frame statistics.
            **Classes**: Aircraft, Person, Baggage Truck, Ramp Loader, Bus, Fuel Truck, Tank Hose, Ground Power Unit, Stairway, Rolling Stairway, Car
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