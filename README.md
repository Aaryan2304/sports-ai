# 🏆 Sports Computer Vision Analytics

> **Production-grade computer vision systems for sports tracking, analysis, and tactical insights**

A collection of end-to-end sports analytics projects leveraging modern deep learning architectures (YOLO, RT-DETR, SAM), multi-object tracking algorithms (ByteTrack, DeepSORT), and real-time inference optimization. Each project demonstrates deployment-ready pipelines from data annotation to production inference.

---

## 📊 Project Portfolio

| Sport | Focus | Key Technologies | Status | Performance Metrics |
|-------|-------|------------------|--------|---------------------|
| **[Volleyball](#volleyball-ai)** | Ball tracking + Team analytics | ONNX seq-9, YOLOv11-Pose, ByteTrack, SigLIP | ✅ Production | 100 FPS ball tracking, 87.4% F1 |
| **[Football](#football-ai)** | Player detection + Match stats | YOLOv11, Supervision, Roboflow | ✅ Complete | Multi-class detection (player/GK/ref/ball) |
| **[Basketball](#basketball-ai)** | Player tracking + Shot analysis | RT-DETR, DeepSORT, Pose estimation | 🚧 In Progress | TBD |

**Legend:**  
✅ Production-ready | 🚧 Active Development | 📋 Planned

---

## 🏐 Volleyball AI

**Real-time ball tracking and tactical analytics with hybrid CPU/GPU architecture**

### Highlights
- **100 FPS ball detection** on CPU (Intel i5-10400F) using custom ONNX seq-9 grayscale model
- **Court calibration** via YOLOv11n-Pose (10 keypoints) + RANSAC homography for metric coordinates
- **Player tracking** with ByteTrack achieving 87.3% MOTA across 20+ simultaneous players
- **Zero-shot team classification** using SigLIP embeddings + KMeans clustering
- **Automatic vertical reel generation** (9:16) with ball-centered smooth cropping

### Technical Stack
```
Ball Detection: VballNetFastV1 (ONNX seq-9 grayscale) → 87.4% F1, 88.2% Precision
Court Mapping:  YOLOv11n-Pose (10 keypoints) → 0.98 mAP50, 0.814 mAP50-95
Player Track:   YOLOv11s + ByteTrack → 97.2% Precision, 99.1% mAP50
Team Classify:  SigLIP-base + KMeans → Unsupervised clustering with temporal smoothing
```

### Key Features
- **Physics-aware analytics:** Ball speed (km/h), velocity vectors, touch detection via proximity
- **Court homography:** Pixel → metric transformation for tactical positioning analysis
- **Modular architecture:** Separate ball (CPU-optimized) and player (GPU-accelerated) pipelines
- **Live demos:** [VPS Demo](https://demo.vb-ai.ru/) | [HuggingFace Space](https://huggingface.co/spaces/asigatchov/volleyball-tracking)

### Project Structure
```
volleyball-ai/
├── models/                    # ONNX models (ball, court, player)
├── src/
│   ├── inference_onnx_seq9_gray_v2.py   # Ball detection (100 FPS)
│   ├── track_calculator.py              # Rally segmentation
│   ├── track_processor.py               # Video assembly
│   └── make_reels.py                    # Vertical reel generator
├── examples/                  # Sample outputs and GIFs
└── HYBRID_ARCHITECTURE.md     # Full technical documentation
```

**[→ View Full Documentation](./volleyball-ai/README.md)**

---

## ⚽ Football AI

**Multi-object detection and tracking for match statistics and player analytics**

### Highlights
- **Multi-class detection:** Players, goalkeepers, referees, and ball using YOLOv11
- **Team classification:** Jersey color-based clustering for team assignment
- **Advanced visualization:** Color-coded bounding boxes, confidence scores, and class labels
- **Production pipeline:** Roboflow integration for dataset management and model deployment

### Technical Stack
```
Detection Model:  YOLOv11 (Roboflow-hosted) → 4-class detection
Annotation:       Roboflow, CVAT → 300+ annotated frames
Visualization:    Supervision library → Professional overlay pipeline
Dataset:          DFL Bundesliga Data Shootout (Kaggle)
```

### Key Features
- **Supervision framework integration:** Modular annotation pipeline with BoxAnnotator, LabelAnnotator
- **Roboflow API deployment:** Cloud-hosted inference for scalable processing
- **Multi-class tracking:** Separate handling for players, goalkeepers, referees, ball
- **Jupyter notebook pipeline:** End-to-end workflow from data loading to inference

### Project Structure
```
football-ai/
├── football_ai.ipynb          # Complete pipeline notebook
├── data/                      # Sample videos and annotations
├── models/                    # YOLOv11 model configs
└── outputs/                   # Annotated videos and stats
```

**[→ View Full Documentation](./football-ai/README.md)**

---

## 🏀 Basketball AI

**Player pose estimation and shot analysis for performance metrics**

### Highlights
- **Pose keypoint detection:** 17-keypoint human pose estimation for biomechanical analysis
- **Shot trajectory tracking:** Ball arc analysis for shooting form evaluation
- **Player tracking:** DeepSORT-based multi-object tracking with re-identification
- **Real-time inference:** Optimized pipeline for live game analysis

### Technical Stack
```
Detection:  RT-DETR → Player bounding boxes
Tracking:   DeepSORT + Kalman Filter → Consistent player IDs
Pose:       HRNet/YOLOv8-Pose → 17 keypoint detection
Analytics:  Shot arc fitting, release angle calculation
```

### Key Features
- **Shooting form analysis:** Release angle, arc trajectory, follow-through metrics
- **Player movement heatmaps:** Court positioning and movement patterns
- **Multi-object tracking:** Handling occlusions and fast movements
- **Performance dashboards:** Real-time stat overlays

### Project Structure
```
basketball-ai/
├── src/
│   ├── detection.py           # RT-DETR player detection
│   ├── tracking.py            # DeepSORT tracking pipeline
│   ├── pose_estimation.py     # Keypoint extraction
│   └── shot_analysis.py       # Trajectory and form analysis
├── models/                    # Detection and pose models
└── notebooks/                 # Exploratory analysis
```

**[→ View Full Documentation](./basketball-ai/README.md)** *(Coming Soon)*

---

## 🛠️ Core Technical Capabilities

### Computer Vision Techniques
- **Object Detection:** YOLO (v5-v11), RT-DETR, Faster R-CNN
- **Segmentation:** SAM (Segment Anything Model), Mask R-CNN
- **Tracking:** ByteTrack, DeepSORT, SORT with Kalman filtering
- **Pose Estimation:** 17-keypoint human pose, custom sports-specific landmarks
- **Classical CV:** Homography transformation, RANSAC, KMeans clustering

### Model Optimization
- **ONNX Conversion:** PyTorch → ONNX for cross-platform deployment
- **Quantization:** INT8/FP16 for edge device inference
- **Temporal Models:** Sequence-based architectures (GRU, seq-N frames) for motion analysis
- **TensorRT Acceleration:** GPU optimization for real-time processing

### Data Engineering
- **Annotation Tools:** Roboflow, CVAT, Supervisely, Label Studio
- **Dataset Management:** 500+ annotated images across detection, segmentation, pose tasks
- **Data Formats:** COCO, YOLO, Pascal VOC, custom sports-specific schemas
- **Augmentation:** Albumentations, Roboflow pipelines for robust training

### Deployment & Infrastructure
- **Frameworks:** FastAPI async endpoints, Docker containerization
- **Monitoring:** Prometheus metrics, custom FPS/latency tracking
- **Cloud Platforms:** Google Colab (T4 GPU), Render, HuggingFace Spaces
- **Edge Deployment:** ONNX Runtime for CPU/ARM-optimized inference

---

## 📈 Performance Benchmarks

| Project | Metric | Value | Hardware |
|---------|--------|-------|----------|
| Volleyball Ball Detection | FPS | 100 | Intel i5-10400F (CPU) |
| Volleyball Ball Detection | F1 Score | 0.874 | - |
| Volleyball Court Detection | mAP50 | 0.98 | RTX GPU |
| Volleyball Player Tracking | MOTA | 87.3% | RTX 3050 |
| Football Multi-Class | Detection Classes | 4 (player/GK/ref/ball) | T4 GPU |
| Basketball Pose | Keypoints | 17 | RTX 3050 |

---

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.10+
python --version

# GPU (Optional but recommended)
nvidia-smi  # Verify CUDA availability
```

### Quick Start - Volleyball Project
```bash
# Clone repository
git clone https://github.com/yourusername/sports-ai.git
cd sports-ai/volleyball-ai

# Install dependencies
pip install -r requirements.txt

# Run ball detection
python src/inference_onnx_seq9_gray_v2.py \
  --video_path examples/sample.mp4 \
  --model_path models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx \
  --output_dir outputs
```

### Quick Start - Football Project
```bash
cd sports-ai/football-ai

# Open Jupyter notebook
jupyter notebook football_ai.ipynb

# Or run inference script
python src/inference.py --video_path data/sample_match.mp4
```

**[→ Full Installation Guide](./INSTALLATION.md)**

---

## 📚 Documentation

### Project-Specific Docs
- **[Volleyball Technical Architecture](./volleyball-ai/HYBRID_ARCHITECTURE.md)** - In-depth system design and algorithms
- **[Football Annotation Guide](./football-ai/ANNOTATION.md)** - Dataset preparation workflow
- **[Basketball Pose Schema](./basketball-ai/POSE_SCHEMA.md)** - Keypoint definitions *(Coming Soon)*

### General Resources
- **[Model Zoo](./docs/MODEL_ZOO.md)** - Pretrained weights and benchmarks
- **[Annotation Best Practices](./docs/ANNOTATION_GUIDE.md)** - Cross-project labeling standards
- **[Deployment Guide](./docs/DEPLOYMENT.md)** - Cloud and edge deployment strategies

---

## 🎯 Use Cases

### Sports Analytics Companies
- **Ball tracking:** Speed, trajectory, and possession analysis
- **Player performance:** Movement heatmaps, positioning metrics
- **Tactical insights:** Formation detection, play pattern recognition

### Broadcast & Media
- **Automated highlights:** Rally/play segmentation and compilation
- **Real-time overlays:** Live stats, player tracking annotations
- **Vertical content:** 9:16 reel generation for social media

### Coaching & Training
- **Biomechanics:** Pose analysis for technique improvement
- **Performance metrics:** Shot accuracy, movement efficiency
- **Video review tools:** Annotated playback with analytics

---

## 🔬 Research & Innovation

### Novel Contributions
1. **Hybrid CPU/GPU Architecture (Volleyball):** Selective GPU acceleration for 100 FPS ball tracking while adding tactical analytics
2. **Zero-Shot Team Classification:** SigLIP embeddings + KMeans for unsupervised jersey clustering
3. **Temporal Sequence Models:** 9-frame grayscale sequences for motion blur handling in fast ball tracking
4. **Physics-Aware Analytics:** Homography-based metric coordinate systems for real-world measurements

### Academic Foundations
- **ByteTrack:** Simple online and real-time tracking with ByteTrack ([Zhang et al., 2021](https://arxiv.org/abs/2110.06864))
- **YOLO Evolution:** YOLOv5-v11 architecture comparisons for sports object detection
- **Homography Estimation:** RANSAC-based robust court calibration under occlusion

---

## 🤝 Contributing

This is a personal portfolio project, but contributions, suggestions, and discussions are welcome!

### Adding New Sports
1. Create project directory: `sports-ai/<sport>-ai/`
2. Follow structure:
   ```
   <sport>-ai/
   ├── README.md          # Project-specific documentation
   ├── src/               # Source code
   ├── models/            # Trained weights (ONNX/PyTorch)
   ├── data/              # Sample datasets
   └── notebooks/         # Exploratory analysis
   ```
3. Update main README table with project details
4. Add technical specs to `docs/MODEL_ZOO.md`

### Reporting Issues
Open an issue with:
- Sport project name
- Expected vs. actual behavior
- System specs (CPU/GPU, OS)
- Minimal reproducible example

---

## 📄 License

This project is licensed under the MIT License - see individual project directories for dataset-specific licenses.

**Dataset Attributions:**
- Volleyball models trained on custom annotated data
- Football data from [DFL Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) (Kaggle)
- See individual project READMEs for complete attributions

---

## 📧 Contact

**Aaryan Kurade**  
🔗 [LinkedIn](https://linkedin.com/in/aaryan-kurade) | [GitHub](https://github.com/Aaryan2304) | [Portfolio](https://aaryankurade.vercel.app)  
📧 aaryankurade27@gmail.com

---

## 🌟 Acknowledgments

- **Supervision:** Open-source CV utilities for annotation and tracking
- **Roboflow:** Dataset management and model deployment platform
- **Ultralytics:** YOLO framework and active community
- **ByteTrack Authors:** State-of-the-art multi-object tracking algorithm

---

**Built with ❤️ for sports analytics and computer vision research**

*Last Updated: February 2026*
