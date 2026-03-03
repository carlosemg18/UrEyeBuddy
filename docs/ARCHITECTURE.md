# UrEyeBuddy — Architecture & Technical Documentation

Detailed technical reference for every layer of the UrEyeBuddy system: the Python model-training pipeline, the Core ML conversion step, and the iOS application.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Phase 1 — Model Training (Python)](#phase-1--model-training-python)
   - [Network Architecture](#network-architecture)
   - [Triplet Dataset](#triplet-dataset)
   - [Training Loop](#training-loop)
3. [Phase 2 — Core ML Export](#phase-2--core-ml-export)
4. [Phase 3 — iOS Application (Swift)](#phase-3--ios-application-swift)
   - [Services Layer](#services-layer)
   - [Recognition Pipeline](#recognition-pipeline)
   - [ViewModels](#viewmodels)
   - [Views](#views)
5. [End-to-End Data Flow](#end-to-end-data-flow)
6. [Threading Model](#threading-model)
7. [Configuration Reference](#configuration-reference)
8. [Privacy & Security](#privacy--security)

---

## System Overview

UrEyeBuddy is split into three sequential phases that produce a single artefact — an iOS app with a bundled Core ML model:

```
Phase 1              Phase 2                Phase 3
Python Training  →   CoreML Conversion  →   iOS App
(PyTorch model)      (.mlpackage)           (Swift / SwiftUI)
```

All inference runs on-device using Apple's Neural Engine. No network calls are made at any point during recognition.

---

## Phase 1 — Model Training (Python)

Source: `ModelTraining/`

### Network Architecture

**File:** `model.py`

`FaceEmbeddingNet` is a metric-learning model that maps a 160×160 face crop to a 128-dimensional L2-normalised embedding vector.

```
Input: (B, 3, 160, 160) RGB tensor normalised to [0, 1]
  │
  ▼
MobileNetV2.features         (pretrained on ImageNet)
  │  Output: (B, 1280, 5, 5)
  ▼
AdaptiveAvgPool2d(1)
  │  Output: (B, 1280)
  ▼
Linear(1280 → 512) + ReLU
  │
  ▼
Linear(512 → 128)
  │
  ▼
L2-normalise
  │
  ▼
Output: (B, 128) unit-length embedding
```

**Why MobileNetV2?** It is designed for mobile inference — depthwise-separable convolutions keep the parameter count low (~3.4 M) while still providing strong feature extraction from ImageNet pre-training.

**Loss function:** `TripletLoss` wraps `nn.TripletMarginLoss` with a configurable margin (default 0.3). The triplet objective pushes embeddings of the same person closer together and embeddings of different people further apart:

```
‖anchor − positive‖₂ + margin < ‖anchor − negative‖₂
```

### Triplet Dataset

**File:** `dataset.py`

`TripletFaceDataset` implements online triplet mining from a folder-per-identity layout:

```
data_root/
  person_a/
    img1.jpg
    img2.jpg
  person_b/
    img1.jpg
    ...
```

**Requirements:**
- At least 2 identities.
- Each identity must have at least 2 images (anchor + positive).

**Augmentation pipeline** (`default_transform`):

| Transform | Value | Purpose |
|---|---|---|
| `Resize` | 160×160 | Match model input |
| `RandomHorizontalFlip` | p=0.5 | Pose variation |
| `ColorJitter` | brightness=0.2, contrast=0.2 | Lighting variation |
| `ToTensor` | scales to [0, 1] | Normalisation |

A separate `eval_transform` skips augmentation (resize + tensor only).

**Triplet generation:** For each index in the flat image list the dataset returns:
1. **Anchor** — the indexed image.
2. **Positive** — a different image of the same person (random choice).
3. **Negative** — a random image of a different person (random choice).

### Training Loop

**File:** `train.py`

| Parameter | Default | CLI Flag |
|---|---|---|
| Data directory | *(required)* | `--data_dir` |
| Epochs | 30 | `--epochs` |
| Batch size | 32 | `--batch_size` |
| Learning rate | 1 × 10⁻⁴ | `--lr` |
| Triplet margin | 0.3 | `--margin` |
| Embedding dim | 128 | `--embedding_dim` |
| Output path | `face_embedding.pth` | `--output` |
| Device | CUDA if available, else CPU | `--device` |

**Optimiser:** Adam.

**Scheduler:** `CosineAnnealingLR` — smoothly decays the learning rate to near-zero by the final epoch, producing smaller weight magnitudes that benefit quantisation.

**DataLoader:** 4 workers, `pin_memory=True`, `drop_last=True`.

**Checkpointing:** The model weights are saved whenever the epoch's average loss improves over the previous best.

**Example:**

```bash
python train.py --data_dir ./data/faces --epochs 30 --batch_size 32
```

---

## Phase 2 — Core ML Export

**File:** `export_coreml.py`

Converts the trained PyTorch checkpoint into an Apple `.mlpackage` ready for Xcode.

### Conversion Steps

1. **Load weights** into `FaceEmbeddingNet` (no pretrained backbone — weights come from the checkpoint).
2. **Trace** the model with a dummy `(1, 3, 160, 160)` tensor via `torch.jit.trace`.
3. **Convert** with `coremltools.convert`:
   - **Input:** `ct.ImageType` named `"faceImage"`, shape `(1, 3, 160, 160)`, scale `1/255` (uint8 → float), zero bias.
   - **Output:** `ct.TensorType` named `"embedding"` (128-d float vector).
   - **Deployment target:** iOS 16+.
4. **Quantise** weights to float16 via `quantization_utils.quantize_weights(nbits=16)`.
5. **Save** as `.mlpackage`.

### Why Float16?

| Benefit | Detail |
|---|---|
| Smaller model | ~50 % size reduction |
| Faster inference | Neural Engine natively operates in float16 |
| Negligible accuracy loss | Embedding quality is preserved at 16-bit precision |

**Example:**

```bash
python export_coreml.py --checkpoint face_embedding.pth \
                        --output FaceEmbedding.mlpackage
```

After export, drag `FaceEmbedding.mlpackage` into Xcode. It will auto-generate a `FaceEmbedding.swift` wrapper class used by `FaceRecognitionService`.

---

## Phase 3 — iOS Application (Swift)

Source: `UrEyeBuddyApp/UrEyeBuddy/`

### Services Layer

#### CameraService

**File:** `Services/CameraService.swift`

Manages `AVCaptureSession` for the front-facing camera.

| Setting | Value | Rationale |
|---|---|---|
| Camera | `.builtInWideAngleCamera`, `.front` | Face recognition needs the front camera |
| Preset | `.hd1280x720` | Good resolution without excess GPU load |
| Pixel format | `32BGRA` | Compatible with Vision + Core Image |
| Frame rate | 30 fps (locked) | Smooth preview without excess power draw |
| Late frames | Discarded | Prevents queue backlog |
| Queue QoS | `.userInteractive` | Lowest possible latency |

Delivers frames via the `CameraServiceDelegate` protocol:

```swift
protocol CameraServiceDelegate: AnyObject {
    func cameraService(_ service: CameraService, didOutput sampleBuffer: CMSampleBuffer)
}
```

#### FaceDetectionService

**File:** `Services/FaceDetectionService.swift`

Uses Apple's Vision framework to detect face bounding boxes and produce 160×160 cropped face images.

**Algorithm:**
1. Run `VNDetectFaceRectanglesRequest` (revision 3) on the pixel buffer.
2. For each detected face:
   - Convert the normalised bounding box (Vision origin: bottom-left) to pixel coordinates.
   - Expand the rectangle by **20 % padding** on each side for better embedding accuracy.
   - Clamp to image bounds.
   - Crop using `CIContext.createCGImage` (GPU-accelerated).
   - Resize to 160×160 with high-quality interpolation via `CGContext`.
3. Return an array of `DetectedFace` values.

```swift
struct DetectedFace {
    let boundingBox: CGRect   // Normalised Vision coordinates
    let croppedFace: CGImage  // 160×160 ready for the model
}
```

#### FaceRecognitionService

**File:** `Services/FaceRecognitionService.swift`

Runs the Core ML embedding model and performs nearest-neighbour matching.

**Model loading:**
```swift
let config = MLModelConfiguration()
config.computeUnits = .all   // Neural Engine → GPU → CPU fallback
let model = try VNCoreMLModel(for: FaceEmbedding(configuration: config).model)
```

**Embedding extraction:**
1. Create a `VNCoreMLRequest` targeting the loaded model.
2. Run the request on the 160×160 `CGImage`.
3. Read the `VNCoreMLFeatureValueObservation` → `MLMultiArray`.
4. Copy to a `[Float]` array and L2-normalise.

**Matching:**
- Iterates over all enrolled contacts and their stored embeddings.
- Computes **cosine similarity** between the query embedding and each stored embedding.
- Returns the best match that exceeds `similarityThreshold` (default 0.65).

```swift
struct MatchResult {
    let contactID: String
    let contactName: String
    let similarity: Float      // 0.0 – 1.0
    let boundingBox: CGRect
}
```

**Cosine similarity formula:**

```
similarity = (a · b) / (‖a‖₂ × ‖b‖₂)
```

Since both vectors are L2-normalised, this simplifies to the dot product.

#### HapticService

**File:** `Services/HapticService.swift`

Delivers discreet tactile feedback using `CoreHaptics`.

**Haptic pattern — "Someone is here":**

| Event | Type | Time | Intensity | Sharpness |
|---|---|---|---|---|
| Tap 1 | `.hapticTransient` | 0 ms | `min(similarity + 0.3, 1.0)` | 0.6 |
| Tap 2 | `.hapticTransient` | 120 ms | `min(similarity + 0.3, 1.0)` | 0.6 |

The two quick taps feel like a subtle double-pulse that is noticeable but unobtrusive during conversation.

**Cooldown system:**
- A dictionary `[contactID: Date]` tracks the last haptic time per contact.
- If less than `cooldownInterval` (default 2.0 s) has elapsed since the last tap for that contact, the haptic is suppressed.
- Prevents continuous buzzing while a face remains in frame.
- `resetCooldowns()` clears state (called when the app returns to foreground).

**Fallback:** On devices without `CoreHaptics` support, falls back to `UINotificationFeedbackGenerator(.success)`.

#### ContactStore

**File:** `Services/ContactStore.swift`

Persists enrolled contacts and their embeddings to local JSON.

| Operation | Method |
|---|---|
| Load all | `loadContacts() → [Contact]` |
| Save all | `saveContacts([Contact])` |
| Enroll | `enroll(name:embeddings:) → Contact` |
| Delete | `delete(contactID:)` |

- **Storage file:** `Documents/contacts.json`
- **Write strategy:** Atomic (`Data.WritingOptions.atomic`)
- **Enrollment logic:** If a contact with the same name exists, new embeddings are appended; otherwise a new contact is created.
- **Access pattern:** Singleton via `ContactStore.shared`.

### Data Model

**File:** `Models/Contact.swift`

```swift
struct Contact: Identifiable, Codable {
    let id: String            // UUID string
    var name: String          // Display name ("Mom", "Dr. Smith")
    var embeddings: [[Float]] // One or more 128-d vectors
    let dateAdded: Date       // Enrollment timestamp
}
```

Each contact stores multiple embeddings captured from different angles during enrollment, improving recognition robustness.

### Recognition Pipeline

**File:** `Pipeline/RecognitionPipeline.swift`

Orchestrates the full **detect → recognise → haptic** loop.

**Published state:**
- `recognisedContacts: [MatchResult]` — currently matched faces (drives UI).
- `isRunning: Bool` — whether the pipeline is active.

**Frame throttling:**
- Camera delivers frames at 30 fps.
- The pipeline processes at most one frame every **100 ms** (~10 fps).
- An `isProcessing` flag prevents overlapping work on the processing queue.

**`processFrame(_:)` — the core loop:**
1. Extract `CVPixelBuffer` from the sample buffer.
2. `FaceDetectionService.detectFaces(in:)` — returns `[DetectedFace]`.
3. Load enrolled contacts from `ContactStore`.
4. For each detected face:
   - `FaceRecognitionService.match(face:against:)` — returns optional `MatchResult`.
   - If matched → `HapticService.fireRecognitionHaptic(for:similarity:)`.
5. Publish results to `@Published recognisedContacts` on the main thread.

### ViewModels

#### CameraViewModel

**File:** `ViewModels/CameraViewModel.swift`

Bridges the `RecognitionPipeline` to SwiftUI.

| Published property | Source |
|---|---|
| `recognisedNames: [String]` | Mapped from `pipeline.recognisedContacts` |
| `isRunning: Bool` | From `pipeline.isRunning` |
| `cameraPermissionGranted: Bool` | `AVCaptureDevice.authorizationStatus` |

Handles camera permission requests and pipeline start/stop.

#### ContactsViewModel

**File:** `ViewModels/ContactsViewModel.swift`

Loads and manages the contact list for the People tab.

### Views

#### ContentView — Root Navigation

**File:** `Views/ContentView.swift`

Three-tab layout:

| Tab | Icon | Destination |
|---|---|---|
| Recognise | `eye.fill` | `CameraView` |
| People | `person.2.fill` | `ContactsListView` |
| Settings | `gear` | `SettingsView` |

#### CameraView — Live Recognition

**File:** `Views/CameraView.swift`

- Displays the live camera preview via `CameraPreviewView`.
- Shows a translucent **recognition banner** (capsule with name) when a contact is matched.
- Animated transitions (`move + opacity`) when names appear/disappear.
- Green/red status dot indicates pipeline state.
- Falls back to a permission placeholder if camera access is denied.

#### CameraPreviewView — AVFoundation Bridge

**File:** `Views/CameraPreviewView.swift`

`UIViewRepresentable` wrapping an `AVCaptureVideoPreviewLayer`. Handles layout updates on view resize.

#### EnrollFaceView — Face Enrollment

**File:** `Views/EnrollFaceView.swift`

Multi-step enrollment flow:

1. Enter a name.
2. Start the camera.
3. Capture **5 face samples** from different angles.
4. Save the contact with collected embeddings.
5. Show a success confirmation.

Uses a progress bar and instructional text to guide the user.

#### ContactsListView — Contact Management

**File:** `Views/ContactsListView.swift`

- Lists enrolled contacts with name and embedding count.
- Swipe-to-delete support.
- "+" toolbar button presents `EnrollFaceView` as a sheet.
- Empty-state placeholder prompts the user to enroll someone.

#### SettingsView — User Preferences

**File:** `Views/SettingsView.swift`

Persisted via `@AppStorage`:

| Setting | Key | Default | Range | Unit |
|---|---|---|---|---|
| Similarity threshold | `similarityThreshold` | 0.65 | 0.40 – 0.90 | — |
| Haptic cooldown | `hapticCooldown` | 2.0 | 0.5 – 10.0 | seconds |
| Analysis FPS | `processingFPS` | 10 | 5 – 30 | frames/s |

Includes a privacy section confirming that all processing is local.

---

## End-to-End Data Flow

```
┌──────────────┐
│ Front Camera │  30 fps, 1280×720, BGRA
└──────┬───────┘
       │ CMSampleBuffer
       ▼
┌──────────────────────┐
│  RecognitionPipeline │  Throttle: process 1 frame per 100 ms
│  (CameraServiceDel.) │
└──────┬───────────────┘
       │ CVPixelBuffer
       ▼
┌──────────────────────┐
│ FaceDetectionService │  VNDetectFaceRectanglesRequest v3
│                      │  → crop + 20% pad → resize 160×160
└──────┬───────────────┘
       │ [DetectedFace]
       ▼
┌──────────────────────┐
│FaceRecognitionService│  Core ML FaceEmbedding model
│                      │  → 128-d L2-normalised embedding
│                      │  → cosine similarity vs ContactStore
│                      │  → best match > 0.65 threshold
└──────┬───────────────┘
       │ MatchResult?
       ▼
┌──────────────────────┐
│   HapticService      │  CoreHaptics double-tap pattern
│                      │  Per-contact 2 s cooldown
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  CameraView (UI)     │  Banner with contact name
│  @Published state    │  Animated transitions
└──────────────────────┘
```

**Latency budget (approximate):**

| Stage | Time |
|---|---|
| Frame capture | ~3 ms |
| Face detection (Vision) | ~8 ms |
| Crop + resize | ~2 ms |
| Embedding (Core ML / NPU) | ~5 ms |
| Cosine similarity search | < 1 ms |
| Haptic trigger | < 1 ms |
| **Total** | **~20 ms** |

The 100 ms throttle interval is conservative — actual processing per frame is much faster.

---

## Threading Model

```
┌─────────────────────────┐
│  Main Thread            │  SwiftUI rendering, @Published updates
└─────────────────────────┘

┌─────────────────────────┐
│  Camera Queue           │  "com.ureyebuddy.camera"
│  QoS: .userInteractive  │  AVCaptureSession frame delivery
└─────────────────────────┘

┌─────────────────────────┐
│  Processing Queue       │  "com.ureyebuddy.pipeline"
│  QoS: .userInteractive  │  Face detection + recognition + haptics
└─────────────────────────┘
```

- Camera frames arrive on the camera queue.
- The pipeline delegate checks the throttle, then dispatches to the processing queue.
- UI updates are marshalled back to the main thread via `DispatchQueue.main.async`.
- Haptic playback fires asynchronously from the processing queue (CoreHaptics is thread-safe).

---

## Configuration Reference

### Model & Recognition

| Parameter | Value | Location |
|---|---|---|
| Input image size | 160 × 160 px | `FaceDetectionService.faceInputSize` |
| Embedding dimensions | 128 | `FaceEmbeddingNet`, `FaceRecognitionService` |
| Similarity threshold | 0.65 (default) | `FaceRecognitionService.similarityThreshold` |
| Adjustable range | 0.40 – 0.90 | `SettingsView` |
| Compute units | `.all` (Neural Engine preferred) | `FaceRecognitionService.loadModel()` |
| Face padding | 20 % | `FaceDetectionService.detectFaces(in:)` |
| Vision request | Revision 3 | `FaceDetectionService` |

### Camera

| Parameter | Value | Location |
|---|---|---|
| Camera position | Front | `CameraService.configure()` |
| Session preset | `hd1280x720` | `CameraService` |
| Frame rate | 30 fps | `CameraService.configure()` |
| Pixel format | BGRA 32-bit | `CameraService` |
| Late frame policy | Discard | `AVCaptureVideoDataOutput.alwaysDiscardsLateVideoFrames` |

### Pipeline

| Parameter | Value | Location |
|---|---|---|
| Frame interval | 100 ms (10 fps) | `RecognitionPipeline.frameInterval` |
| Adjustable range | 5 – 30 fps | `SettingsView` |
| Queue QoS | `.userInteractive` | `RecognitionPipeline.processingQueue` |

### Haptics

| Parameter | Value | Location |
|---|---|---|
| Pattern | 2 × `.hapticTransient` at 0 ms and 120 ms | `HapticService.playCoreHaptic()` |
| Sharpness | 0.6 | `HapticService` |
| Intensity | `min(similarity + 0.3, 1.0)` | `HapticService` |
| Cooldown | 2.0 s (default) | `HapticService.cooldownInterval` |
| Adjustable range | 0.5 – 10.0 s | `SettingsView` |
| Fallback | `UINotificationFeedbackGenerator(.success)` | `HapticService.playFallbackHaptic()` |

### Training (Python)

| Parameter | Default | CLI Flag |
|---|---|---|
| Epochs | 30 | `--epochs` |
| Batch size | 32 | `--batch_size` |
| Learning rate | 1 × 10⁻⁴ | `--lr` |
| Triplet margin | 0.3 | `--margin` |
| Embedding dim | 128 | `--embedding_dim` |
| Scheduler | CosineAnnealingLR | — |
| Optimiser | Adam | — |
| DataLoader workers | 4 | — |
| Quantisation | float16 | `--quantize` |
| iOS deploy target | 16.0+ | — |

### Enrollment

| Parameter | Value | Location |
|---|---|---|
| Samples per enrollment | 5 | `EnrollFaceView.targetSamples` |
| Storage format | JSON | `ContactStore` |
| Storage file | `Documents/contacts.json` | `ContactStore.storageURL` |

---

## Privacy & Security

| Principle | Implementation |
|---|---|
| **No cloud communication** | Zero network calls in the entire codebase. No URLSession, no API keys, no analytics SDKs. |
| **On-device inference** | Core ML runs on the Neural Engine / GPU / CPU — never offloaded. |
| **Local storage only** | Face embeddings and contact names are stored in the app's sandboxed Documents directory as JSON. |
| **No raw images stored** | Only 128-d numerical embeddings are persisted — the original face images are discarded after embedding extraction. |
| **Camera permission** | Explicit `NSCameraUsageDescription` with a clear justification shown to the user. |
| **Minimal data** | Each contact stores only a name, an ID, embeddings, and a date. No photos, no metadata. |

---

## Dependencies

### iOS (system frameworks — no third-party dependencies)

| Framework | Usage |
|---|---|
| `SwiftUI` | User interface |
| `AVFoundation` | Camera capture |
| `Vision` | Face detection |
| `CoreML` | Neural network inference |
| `CoreHaptics` | Haptic feedback |
| `Combine` | Reactive data binding |

### Python

| Package | Version | Usage |
|---|---|---|
| `torch` | ≥ 2.0.0 | Model definition and training |
| `torchvision` | ≥ 0.15.0 | MobileNetV2 backbone, image transforms |
| `coremltools` | ≥ 7.0 | PyTorch → Core ML conversion + quantisation |
| `Pillow` | ≥ 10.0.0 | Image loading |
| `numpy` | ≥ 1.24.0 | Numerical operations |
| `scikit-learn` | ≥ 1.3.0 | Evaluation utilities |
| `tqdm` | ≥ 4.65.0 | Training progress bars |
