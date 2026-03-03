# UrEyeBuddy

A real-time, edge-computing iOS application that helps users with low vision recognise known faces and receive instant haptic feedback — mimicking natural, human-like social recognition.

## The Problem

Navigating social interactions with low vision makes it difficult to recognise who is approaching. Cloud-based AI tools introduce network lag that disrupts the natural flow of conversation, and audio feedback can interrupt dialogue.

## The Solution

A fully on-device facial recognition pipeline running on the iPhone's Neural Engine. The system uses the front-facing camera to detect faces and immediately alerts the user via discreet vibrations when a saved contact is recognised.

```
Camera (30 fps) → Face Detection (Vision) → Embedding (Core ML) → Match → Haptic Tap
         └────────── all on-device, ~100 ms end-to-end ──────────────┘
```

## Project Structure

```
UrEyeBuddy/
├── ModelTraining/              # Phase 1 & 2: Python training + CoreML export
│   ├── model.py                # MobileNetV2 Siamese network (128-d embeddings)
│   ├── dataset.py              # Triplet dataset with augmentation
│   ├── train.py                # Training loop with triplet loss
│   ├── export_coreml.py        # Convert to quantised .mlpackage
│   └── requirements.txt
│
├── UrEyeBuddyApp/             # Phase 3: iOS application (Swift / SwiftUI)
│   ├── UrEyeBuddy.xcodeproj/
│   └── UrEyeBuddy/
│       ├── App/                # Entry point + Info.plist
│       ├── Models/             # Contact data model
│       ├── Services/           # Camera, detection, recognition, haptics, storage
│       ├── Pipeline/           # Recognition orchestrator
│       ├── ViewModels/         # MVVM view models
│       └── Views/              # SwiftUI interface
│
└── docs/
    └── ARCHITECTURE.md         # Detailed technical documentation
```

## Quick Start

### 1. Train the Model (Python)

```bash
cd ModelTraining
pip install -r requirements.txt

# Prepare a dataset: one folder per person, ≥2 images each
#   data/faces/alice/img1.jpg, img2.jpg, ...
#   data/faces/bob/img1.jpg, img2.jpg, ...

python train.py --data_dir ./data/faces --epochs 30 --batch_size 32

python export_coreml.py --checkpoint face_embedding.pth \
                        --output FaceEmbedding.mlpackage
```

### 2. Build the iOS App (Xcode)

1. Open `UrEyeBuddyApp/UrEyeBuddy.xcodeproj` in Xcode 15+.
2. Drag `FaceEmbedding.mlpackage` into the Xcode project navigator.
3. Select an iPhone as the run destination (simulator has no camera or Neural Engine).
4. Build and run.

### 3. Use the App

| Tab | What it does |
|---|---|
| **Recognise** | Live camera view — shows a banner with the person's name and fires a haptic tap when a match is found. |
| **People** | Manage enrolled contacts. Tap **+** to enroll someone new by capturing 5 face samples. |
| **Settings** | Tune similarity threshold, haptic cooldown, and processing frame rate. |

## Technical Highlights

| Requirement | How it's met |
|---|---|
| **Ultra-low latency** | `userInteractive` QoS queues, Vision framework v3, Neural Engine compute, 100 ms frame interval |
| **100 % on-device** | No network calls. All inference, storage, and haptics are local. |
| **Battery efficient** | 30 fps capture / 10 fps analysis, late-frame discard, float16 quantised model, cosine-annealing LR for smaller weights |
| **Privacy** | Face embeddings stored in the app sandbox as JSON. No images or data leave the device. |

## Architecture Overview

```
┌─────────────┐    CMSampleBuffer    ┌──────────────────┐
│ CameraService├────────────────────►│RecognitionPipeline│
│  (30 fps)   │                      │  (throttle 10fps) │
└─────────────┘                      └────────┬─────────┘
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   ▼                   ▼
                  ┌───────────────┐  ┌────────────────┐  ┌──────────────┐
                  │FaceDetection  │  │FaceRecognition │  │HapticService │
                  │Service        │  │Service         │  │              │
                  │(Vision fw)    │  │(Core ML + NNU) │  │(CoreHaptics) │
                  └───────┬───────┘  └───────┬────────┘  └──────▲───────┘
                          │                  │                   │
                   DetectedFace        MatchResult          fire haptic
                   (160×160 crop)   (name, similarity)     if match found
                          │                  │                   │
                          └──────────────────┴───────────────────┘
                                             │
                                     ┌───────▼───────┐
                                     │ ContactStore  │
                                     │ (local JSON)  │
                                     └───────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full technical deep-dive.

## Requirements

| Component | Minimum |
|---|---|
| iPhone | A12 Bionic or later (Neural Engine required) |
| iOS | 16.0+ |
| Xcode | 15.0+ |
| Python | 3.10+ (for model training) |
| PyTorch | 2.0+ |
| coremltools | 7.0+ |

## License

This project is provided for educational and accessibility purposes.
