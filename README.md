# Animal Detection using YOLOv11
A deep-learning object detection project that uses Ultralytics YOLO11 (yolo11s) to detect and localize animals in images and videos. It downloads a Kaggle dataset, converts original bounding-box labels into the YOLO annotation format, trains the model, evaluates performance, and runs inference on an uploaded video.

**Project type:** Object Detection / Computer Vision  
**Model:** YOLO11s  
**Framework:** Ultralytics YOLO  
**Primary environment:** Google Colab / Jupyter Notebook  
**Dataset:** Antoreep Jana's Animals Detection Images Dataset on Kaggle  
**Trained weights included:** `best.pt`

## Overview

This project implements an end-to-end animal object detection pipeline using **YOLO11**.

Unlike image classification, where a model predicts one or more labels for an entire image, object detection identifies:

1. **What** object is present.
2. **Where** the object is located using a bounding box.

For example, an image containing a horse, zebra, and lion can produce multiple detections, with each detected object assigned a class name and bounding box.

The notebook performs the following operations:

```
Kaggle Dataset
      │
      ▼
Download Dataset
      │
      ▼
Inspect Dataset Structure
      │
      ▼
Read Original Bounding-Box Annotations
      │
      ▼
Convert Annotations to YOLO Format
      │
      ▼
Create data.yaml
      │
      ▼
Train YOLO11s
      │
      ▼
Validate / Evaluate Model
      │
      ▼
Load best.pt
      │
      ▼
Run Detection on Video
      │
      ▼
Generate Annotated MP4
```

## Features

- Uses **YOLO11s** for object detection.
- Downloads the dataset directly through `kagglehub`.
- Automatically discovers `train`, `val`, `valid`, or `test` directories.
- Converts the dataset's original bounding-box annotations into YOLO format.
- Automatically discovers class names from the annotation files.
- Creates a YOLO-compatible `data.yaml`.
- Trains using Ultralytics.
- Uses early stopping through the `patience` parameter.
- Generates training plots.
- Evaluates the model using:
  - Precision
  - Recall
  - mAP@50
  - mAP@50-95
- Generates a normalized confusion matrix.
- Performs inference on an uploaded video.
- Saves an annotated video.
- Exports the trained `best.pt` weights.

## Project Workflow

### 1. Environment Setup

The notebook installs and imports:

- `ultralytics`
- `kaggle`
- `kagglehub`
- `torch`
- `Pillow`
- `PyYAML`
- `tqdm`

It also checks whether CUDA/GPU acceleration is available.

### 2. Dataset Download

The notebook uses:

```
antoreepjana/animals-detection-images-dataset
```

through `kagglehub`.

### 3. Dataset Inspection

The notebook searches the downloaded directory recursively and identifies dataset splits such as:

- `train`
- `val`
- `valid`
- `validation`
- `test`

The validation directory is selected from `val`, `valid`, `validation`, or, if necessary, `test`.

### 4. Annotation Conversion

The original annotations are read and converted into YOLO's normalized format.

The conversion also:

- clips bounding boxes to image boundaries,
- ignores invalid annotations,
- ignores extremely small boxes,
- creates unique output filenames,
- creates separate `images` and `labels` directories.

### 5. Training

The converted dataset is passed to YOLO11s.

The configured training run uses:

- image size: `640 × 640`
- batch size: `16`
- maximum training time: `3 hours`
- early stopping patience: `15`
- 2 dataloader workers
- no image caching
- training plots enabled

### 6. Evaluation

The best checkpoint is loaded and evaluated on the validation set.

### 7. Video Detection

A video is uploaded through the Colab file-upload interface.

YOLO processes the video frame by frame and writes annotated frames/video output.

### 8. Export

The notebook converts the generated video to MP4 when necessary and downloads:

```
result.mp4
best.pt
```

## Repository Structure

The supplied project archive contains:

```
ABC/
├── AnimalDetectionusingYOLO11.ipynb
└── best.pt
```

### `AnimalDetectionusingYOLO11.ipynb`

The main Jupyter/Google Colab notebook containing the complete pipeline:

- setup,
- dataset download,
- dataset inspection,
- annotation conversion,
- training,
- evaluation,
- video inference,
- output/download steps.

### `best.pt`

The trained YOLO11s checkpoint selected as the best model during training.

It can be loaded with Ultralytics:

```
from ultralytics import YOLO

model = YOLO("best.pt")
```

## Dataset

The project uses the following Kaggle dataset:

**Animals Detection Images Dataset — Antoreep Jana**

Kaggle:

https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset

The notebook does not assume that the dataset is already in YOLO format. Instead, it reads the dataset's existing annotation structure and converts the bounding boxes into YOLO-compatible label files.

## Dataset Preparation

The original dataset is expected to contain image files and corresponding annotation files under a `Label` directory.

The notebook searches recursively for:

```
Label/*.txt
```

For every image, it attempts to locate the corresponding annotation file:

```
<image directory>/Label/<image stem>.txt
```

The notebook then extracts:

```
class_name x1 y1 x2 y2
```

from each valid annotation line.

## Supported Classes

The trained model contains **80 classes** discovered from the dataset annotations.

The class list used by the training run is:

1. Bear
2. Brown bear
3. Bull
4. Butterfly
5. Camel
6. Canary
7. Caterpillar
8. Cattle
9. Centipede
10. Cheetah
11. Chicken
12. Crab
13. Crocodile
14. Deer
15. Duck
16. Eagle
17. Elephant
18. Fish
19. Fox
20. Frog
21. Giraffe
22. Goat
23. Goldfish
24. Goose
25. Hamster
26. Harbor seal
27. Hedgehog
28. Hippopotamus
29. Horse
30. Jaguar
31. Jellyfish
32. Kangaroo
33. Koala
34. Ladybug
35. Leopard
36. Lion
37. Lizard
38. Lynx
39. Magpie
40. Monkey
41. Moths and butterflies
42. Mouse
43. Mule
44. Ostrich
45. Otter
46. Owl
47. Panda
48. Parrot
49. Penguin
50. Pig
51. Polar bear
52. Rabbit
53. Raccoon
54. Raven
55. Red panda
56. Rhinoceros
57. Scorpion
58. Sea lion
59. Sea turtle
60. Seahorse
61. Shark
62. Sheep
63. Shrimp
64. Snail
65. Snake
66. Sparrow
67. Spider
68. Squid
69. Squirrel
70. Starfish
71. Swan
72. Tick
73. Tiger
74. Tortoise
75. Turkey
76. Turtle
77. Whale
78. Woodpecker
79. Worm
80. Zebra

The class IDs are assigned automatically by sorting the discovered class names alphabetically. Therefore, if the source dataset or annotations change, the generated class mapping may also change.

## Technologies and Dependencies

### Core Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep-learning backend |
| Ultralytics | YOLO11 training, validation and inference |
| YOLO11s | Object detection model |
| KaggleHub | Dataset download |
| Pillow | Image loading and dimensions |
| PyYAML | `data.yaml` generation |
| tqdm | Progress bars |
| FFmpeg | Video conversion |
| Google Colab | Notebook execution environment |

### Install Dependencies

The notebook installs the main packages with:

```
pip install -q ultralytics kaggle
pip install -q kagglehub
```

The notebook also imports:

```
import torch
import ultralytics
from PIL import Image
from tqdm import tqdm
import yaml
```

## Hardware and Runtime

The recorded training run was performed using a GPU runtime with:

```
GPU: NVIDIA Tesla T4
VRAM: approximately 14.9 GB
CUDA: available
```

The training log reports:

```
YOLO11s
Parameters: 9,443,760
GFLOPs: 21.6
```

The exact training speed and memory usage can vary depending on:

- GPU model,
- available VRAM,
- CPU,
- RAM,
- storage speed,
- Ultralytics version,
- PyTorch version,
- CUDA version,
- dataset state.

A GPU is strongly recommended for training.

## Dataset Preparation and Annotation Conversion

The notebook creates a converted dataset at:

```
/content/yolo_animals/
```

The resulting structure is:

```
yolo_animals/
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
└── data.yaml
```

## Image Directories

```
images/train/
images/val/
```

contain the corresponding training and validation images.

## Label Directories

```
labels/train/
labels/val/
```

contain YOLO `.txt` annotation files.

## `data.yaml`

The notebook creates:

```
path: /content/yolo_animals
train: images/train
val: images/val
names:
  0: Bear
  1: Brown bear
  ...
```

The exact class mapping is generated automatically.

## Training Configuration

| Parameter | Value | Description |
|---|---:|---|
| Model | `yolo11s.pt` | YOLO11 small pretrained model |
| Dataset | `data.yaml` | YOLO dataset configuration |
| Training time | `3.0` hours | Maximum training duration |
| Image size | `640` | Training image size |
| Batch size | `16` | Images per training batch |
| Patience | `15` | Early-stopping patience |
| Workers | `2` | Data-loading workers |
| Cache | `False` | Dataset is not cached in RAM |
| Plots | `True` | Training plots are generated |
| Project | `/content/runs` | Output directory |
| Run name | `animals` | Training run name |

### Important

The `time=3.0` parameter specifies a **maximum training duration of approximately three hours**, rather than explicitly specifying a fixed number of epochs.

## Evaluation

After training, the notebook loads:

```
/content/runs/animals/weights/best.pt
```

and evaluates it with:

```
best = "/content/runs/animals/weights/best.pt"

model = YOLO(best)

metrics = model.val(
    data="/content/yolo_animals/data.yaml"
)
```

The notebook prints:

```
mAP@50
mAP@50-95
```

and displays:

```
results.png
confusion_matrix_normalized.png
```

## Results

The recorded validation run produced the following overall metrics:

| Metric | Result |
|---|---:|
| Validation images | 6,505 |
| Validation instances | 7,576 |
| Precision | 0.597 |
| Recall | 0.626 |
| mAP@50 | **0.613** |
| mAP@50-95 | **0.527** |

The evaluation log reports approximately:

```
Speed:
1.9 ms preprocess
7.2 ms inference
0.0 ms loss
1.7 ms postprocess
```

These values were measured in the recorded runtime and should not be treated as universal performance figures.

## Interpretation

### Precision

Precision measures the proportion of predicted detections that are correct.

A precision of:

```
0.597
```

means that, under the evaluation configuration, approximately 59.7% of predicted detections corresponded to correct detections according to the evaluation criteria.

### Recall

Recall measures how many of the relevant objects were successfully detected.

A recall of:

```
0.626
```

indicates that approximately 62.6% of the relevant annotated objects were detected under the evaluation configuration.

### mAP@50

mAP@50 evaluates detection performance using an IoU threshold of 0.50.

Recorded result:

```
0.613
```

### mAP@50-95

mAP@50-95 averages mean Average Precision over multiple IoU thresholds from 0.50 through 0.95.

Recorded result:

```
0.527
```

This metric is stricter than mAP@50 because it evaluates localization quality at progressively higher IoU thresholds.

## Video Inference

The notebook allows the user to upload a video through Google Colab:

```
from google.colab import files

up = files.upload()
```

The uploaded video is then passed to YOLO:

```
model.predict(
    source=video_path,
    stream=True,
    conf=0.25,
    iou=0.5,
    imgsz=640,
    save=True,
    project="/content/output",
    name="video_pred",
    exist_ok=True,
    line_width=2,
    verbose=False,
)
```

### Inference Configuration

| Parameter | Value | Description |
|---|---:|---|
| Confidence threshold | `0.25` | Minimum detection confidence |
| IoU threshold | `0.50` | NMS overlap threshold |
| Image size | `640` | Inference image size |
| Streaming | `True` | Processes video incrementally |
| Save | `True` | Saves annotated output |
| Line width | `2` | Bounding-box line width |


## Video Detection Statistics

In the recorded test run, the model processed:

```
4,237 frames
```

The notebook also counted detections by class across processed frames.

Example recorded counts included:

```
Zebra: 1195
Horse: 958
Tiger: 369
Deer: 351
Cheetah: 329
Lion: 332
Camel: 299
Rhinoceros: 254
```

These are **frame-level detection counts**, not unique animal counts. If the same animal remains visible for many frames, it can be counted repeatedly.

Therefore:

A detection count of 1,195 for Zebra does not mean that 1,195 unique zebras were present in the video.

## Trained Model

The trained model is provided as:

```
best.pt
```

It is the best checkpoint selected during the YOLO training process.

## Load the Model

```
from ultralytics import YOLO

model = YOLO("best.pt")
```

### Image Inference

For an image:

```
results = model.predict(
    source="image.jpg",
    conf=0.25
)
```

### Video Inference

```
results = model.predict(
    source="video.mp4",
    conf=0.25,
    iou=0.5,
    imgsz=640,
    save=True
)
```

### Webcam Inference

A compatible camera can also be used with Ultralytics:

```
model.predict(
    source=0,
    conf=0.25,
    show=True
)
```

## Output Files

The notebook creates several outputs.

### Training Outputs

```
/content/runs/animals/
```

Typical contents include:

```
runs/
└── animals/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── results.png
    ├── confusion_matrix_normalized.png
    └── ...
```

### Converted Dataset

```
/content/yolo_animals/
```

contains:

```
images/
labels/
data.yaml
```

### Video Output

```
/content/output/video_pred/
```

contains the annotated video generated by Ultralytics.

The notebook then creates:

```
/content/output/result.mp4
```

using FFmpeg.

## How the Annotation Conversion Works

The original annotation format contains:

```
<class name> <x1> <y1> <x2> <y2>
```

where:

- `class name` = animal category
- `x1` = left bounding-box coordinate
- `y1` = top bounding-box coordinate
- `x2` = right bounding-box coordinate
- `y2` = bottom bounding-box coordinate

YOLO requires:

```
<class_id> <x_center> <y_center> <width> <height>
```

with the four coordinate values normalized to `[0, 1]`.

The notebook calculates:

```
x_center = (x1 + x2) / 2
y_center = (y1 + y2) / 2

width  = x2 - x1
height = y2 - y1
```

and then normalizes them:

```
x_center / image_width
y_center / image_height
width / image_width
height / image_height
```

The resulting annotation is written to a `.txt` file.

## YOLO Label Format

Each object is represented by one line:

```
class_id x_center y_center width height
```

Example:

```
17 0.512500 0.473611 0.245000 0.381944
```

This means:

```
class_id  = 17
x_center  = 0.512500
y_center  = 0.473611
width     = 0.245000
height    = 0.381944
```

All coordinates are relative to the image dimensions.

## Important

YOLO uses:

```
normalized center coordinates
```

rather than:

```
absolute corner coordinates
```

This makes annotations independent of the original image resolution.

## Bounding-Box Validation

During conversion, the notebook clips coordinates to image boundaries.

Conceptually:

```
x1 = max(0, min(x1, x2))
x2 = min(width, max(x1, x2))

y1 = max(0, min(y1, y2))
y2 = min(height, max(y1, y2))
```

It also skips boxes that are too small:

```
if x2 - x1 < 2 or y2 - y1 < 2:
    continue
```

This helps prevent invalid or unusably small bounding boxes from entering the training dataset.

## Unique Output Filenames

The conversion process creates a filename based on the original parent directory and image filename:

```
stem = f"{img.parent.name}_{img.stem}".replace(" ", "_")
```

This reduces the possibility of filename collisions when images from different directories have identical filenames.

## Troubleshooting

### 1. CUDA Is Not Available

If:

```
torch.cuda.is_available()
```

returns:

```
False
```

training may run on CPU and become significantly slower.

### Solution

In Google Colab:

```
Runtime
→ Change runtime type
→ Hardware accelerator
→ GPU
```

Then restart/reconnect the runtime if required.

### 2. Kaggle Authentication Fails

The dataset download depends on KaggleHub authentication.

Run:

```
kagglehub.login()
```

again and complete the authentication process.

### 3. No `train` Directory Found

The conversion code expects the dataset to contain a training split.

If the dataset structure has changed, inspect it with the dataset-structure cell and update the split-detection logic.

### 4. No Validation/Test Split Found

The notebook searches for:

```
val
valid
validation
test
```

If none exists, the conversion step raises:

```
No validation/test split found
```

The dataset structure must be inspected and the code adjusted accordingly.

### 5. Out-of-Memory During Training

If the GPU runs out of memory, reduce the batch size:

```
batch=8
```

or:

```
batch=4
```

You can also reduce the image size:

```
imgsz=512
```

For example:

```
results = model.train(
    data="/content/yolo_animals/data.yaml",
    time=3.0,
    imgsz=512,
    batch=8,
    patience=15,
    workers=2,
    cache=False,
)
```

Reducing batch size generally has less effect on detection resolution than reducing `imgsz`.

### 6. Video Output Is Not Displayed Inline

The notebook skips inline preview when the converted MP4 is larger than approximately 25 MB.

In that situation, download:

```
/content/output/result.mp4
```

instead.

### 7. `best.pt` Cannot Be Found

After training, verify:

```
/content/runs/animals/weights/best.pt
```

exists.

If the run name or project directory was changed, update the path used in the evaluation and inference cells.

## Limitations

This project has several practical limitations.

### Dataset Imbalance

The validation set contains substantially different numbers of examples for different classes.

Some classes have very few validation instances. For example, the recorded validation set contains only:

- Tick: 1 instance
- Turtle: 7 instances
- Squid: 13 instances

Performance estimates for classes with very few examples can therefore be unstable.

### Difficult Classes

The recorded per-class results show that performance varies considerably between classes.

For example, some classes achieved high mAP@50, while others were substantially lower. This is expected in a multi-class dataset with different object sizes, visual similarity, and sample counts.

### Small Objects

Classes such as insects, fish, and other small objects can be difficult to localize, particularly when the object occupies a small portion of an image.

### Similar Classes

Some classes are visually similar, for example:

```
Bear / Brown bear / Polar bear
Turtle / Sea turtle
Butterfly / Moths and butterflies
Cattle / Bull
```

This can increase classification confusion.

### Video Counts Are Not Unique Object Counts

The video detection counter counts detections per frame.

It does not perform object tracking.

Therefore, the same animal can contribute many detections over time.

### No Persistent Tracking

The current notebook performs detection only.

It does not assign persistent IDs to objects.

A future version could integrate an object tracker such as ByteTrack or BoT-SORT.

## Possible Improvements

The project can be extended in several ways.

### 1. Increase Training Data

Adding more balanced examples can improve generalization, particularly for classes with few samples.

### 2. Train for Longer

The current run uses a maximum training duration of approximately three hours.

Longer training may improve performance if the model has not converged, although longer training does not automatically guarantee better generalization.

### 3. Hyperparameter Tuning

Possible parameters to experiment with include:

```
learning rate
batch size
image size
augmentation
optimizer
weight decay
warmup
confidence threshold
IoU threshold
```

### 4. Use a Larger YOLO Model

The current model is:

```
YOLO11s
```

Larger YOLO11 variants may provide stronger detection capacity at the cost of increased computational and memory requirements.

### 5. Add Object Tracking

Tracking could provide unique IDs:

```
Animal #1
Animal #2
Animal #3
```

and allow the system to estimate unique animals instead of counting detections on every frame.

### 6. Improve Class Balance

Classes with very few samples can be improved by:

- collecting additional images,
- targeted augmentation,
- oversampling,
- improving annotation quality,
- reviewing class definitions.

### 7. Add Image and Webcam Interfaces

The trained model can be integrated into:

- Streamlit,
- Gradio,
- Flask,
- FastAPI,
- desktop applications,
- webcam applications,
- edge-device applications.

## Reproducibility Notes

The recorded experiment used:

```
Ultralytics: 8.4.162
PyTorch: 2.11.0+cu128
GPU: Tesla T4
YOLO model: YOLO11s
Image size: 640
Batch size: 16
Maximum training time: 3 hours
Workers: 2
Cache: False
```

Exact results may differ when the project is retrained because of:

- random initialization or data-order effects,
- software-version changes,
- GPU differences,
- CUDA/PyTorch differences,
- dataset changes,
- changes in preprocessing or augmentation,
- changes in the automatically generated class mapping.

The included `best.pt` should be used when the goal is to reproduce the recorded trained model rather than retrain from scratch.

## Model Usage Example

Minimal example:

```
from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Detect objects in an image
results = model.predict(
    source="image.jpg",
    conf=0.25,
    imgsz=640
)

# Save annotated result
for result in results:
    result.save(filename="annotated.jpg")
```

## Video Usage Example

```
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="video.mp4",
    conf=0.25,
    iou=0.5,
    imgsz=640,
    save=True
)
```

The annotated video will be saved by Ultralytics in its configured output directory.

## Project Outputs

The main deliverables of this project are:

```
AnimalDetectionusingYOLO11.ipynb
best.pt
```

After executing the notebook, additional generated files include:

```
yolo_animals/
runs/animals/
output/video_pred/
output/result.mp4
```

## License and Attribution

This repository contains a trained model and notebook based on a publicly available Kaggle dataset.

## Summary

This project provides a complete YOLO11-based animal detection pipeline:

```text
Dataset
   ↓
KaggleHub Download
   ↓
Dataset Inspection
   ↓
Annotation Conversion
   ↓
YOLO Dataset Generation
   ↓
YOLO11s Training
   ↓
Validation
   ↓
best.pt
   ↓
Image / Video Detection
   ↓
Annotated Output
```

The recorded model achieved:

```
mAP@50      = 0.613
mAP@50-95   = 0.527
Precision   = 0.597
Recall      = 0.626
```

on **6,505 validation images containing 7,576 annotated object instances**.

The included `best.pt` checkpoint can be loaded directly with Ultralytics for further image, video, webcam, or application-level inference.

