Emotional Detection using FER2013 – Swin Transformer Tiny with 96×96 Upscaling
This project focuses on building an emotion recognition system using the FER2013 dataset.
The goal is to improve classification accuracy on small, low‑resolution, imbalanced facial images by applying modern deep learning techniques, including Swin Transformer Tiny and image upscaling.

📌 Project Purpose
The main objectives of this project are:

To develop an automatic facial emotion detection model.

To overcome the limitations of the FER2013 dataset (small resolution, noise, imbalance).

To evaluate multiple model architectures (CNN, EfficientNet, Transformer).

To improve accuracy beyond the typical ~45–50% baseline of FER2013.

To apply data balancing and image enhancement strategies.

To identify the best-performing approach for real-world deployment.

📂 Dataset Selection
This project uses the FER2013 dataset from Kaggle:

🔗 Dataset Link:  
https://www.kaggle.com/datasets/msambare/fer2013 (kaggle.com in Bing)

Dataset Characteristics
35,887 grayscale images

Resolution: 48×48 pixels

7 emotion classes:

Angry

Disgust

Fear

Happy

Sad

Surprise

Neutral

Images are low resolution, noisy, and often misaligned.

The dataset is highly imbalanced, especially the Disgust class (~1% of total samples).

⚠️ Data Challenges
1. Very Small Resolution (48×48)
Limits feature extraction.

CNNs struggle to learn fine facial details.

Transformers (ViT) fail due to too few patches.

2. Noise and Poor Quality
Many images are blurred, occluded, or poorly lit.

Subtle emotions (Fear, Disgust, Surprise) are difficult to distinguish.

3. Severe Class Imbalance
Disgust has extremely few samples.

Causes biased predictions toward majority classes (Happy, Neutral).

🔧 Approach 1 — Data Adjustment (Balancing)
Disgust Class Augmentation
The Disgust class was increased from ~400 samples to 1271 samples using:

Rotation

Flipping

Zoom

Shifting

Contrast adjustments

Results
Accuracy improved from ~45% → 50–55%

Better stability across classes

However, the model still hit a performance ceiling due to the 48×48 resolution limit

Conclusion
Data balancing helps, but resolution remains the main bottleneck.

🔧 Approach 2 — Upscaling to 96×96 + Swin Transformer Tiny
This is the best-performing strategy in the project.

Upscaling 48×48 → 96×96
Increases pixel count 4×

Provides more meaningful patches for Transformer attention

Enhances facial structure visibility

Reduces information loss during feature extraction

Swin Transformer Tiny
Window-based attention → ideal for small images

Hierarchical structure → similar to CNN but more powerful

Patch size is small → captures fine details

Performs well on noisy and low-resolution data

More stable than ViT and lighter than larger Transformers

Best Results
Achieved 60–70% accuracy depending on training configuration

Outperformed CNNs, EfficientNetB0/B2, MobileNetV2, and MobileViT

Reduced overfitting

Works efficiently on Kaggle GPU (T4)

Conclusion
Swin Transformer Tiny + 96×96 Upscaling is the optimal solution for FER2013.

🏆 Performance Summary
Approach	Model	Input Size	Accuracy
Data Fix Only	CNN / MobileNet / EfficientNet	48×48	45–55%
Upscaling + Transformer	Swin Transformer Tiny	96×96	60–70%


📁 Project Structure
Code
FER_Project/
│
├── best_model.h5                # trained model
├── __notebook_source__.ipynb    # Kaggle notebook
├── label_map.json               # emotion class mapping (optional)
├── README.md                    # project documentation
│
└── new_dataset/                 # processed dataset (train/val/test)
    ├── train/
    ├── val/
    └── test/
🚀 Model Usage
Load the model:
python
from tensorflow.keras.models import load_model
model = load_model("best_model.h5")
Run inference:
python
import cv2
import numpy as np

img = cv2.imread("face.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (96, 96))
img = img / 255.0
img = np.expand_dims(img, axis=(0, -1))

pred = model.predict(img)
print(pred)
📌 Notes
FER2013 is a challenging dataset; Transformer-based models significantly improve performance.

Swin Tiny is ideal for small images and noisy data.

Upscaling is essential for unlocking Transformer capabilities.

This project can be extended to real-time detection or deployed as an API.