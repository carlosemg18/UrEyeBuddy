import AVFoundation
import Combine

/// Orchestrates the full camera → detect → recognise → haptic pipeline.
///
/// Processes camera frames on a dedicated queue, throttled to avoid
/// overwhelming the CPU/NPU while maintaining low latency.
final class RecognitionPipeline: NSObject, ObservableObject {

    // MARK: - Published state

    /// Currently recognised contacts in the frame (updated on main queue).
    @Published var recognisedContacts: [FaceRecognitionService.MatchResult] = []
    /// Whether the pipeline is actively running.
    @Published var isRunning = false

    // MARK: - Services

    let cameraService = CameraService()
    private let faceDetection = FaceDetectionService()
    private let faceRecognition = FaceRecognitionService()
    let hapticService = HapticService()
    private let contactStore = ContactStore.shared

    // MARK: - Throttling

    /// Process at most one frame every N milliseconds.
    /// 100 ms ≈ 10 fps analysis — keeps latency low without burning battery.
    private let frameInterval: TimeInterval = 0.1
    private var lastProcessedTime: Date = .distantPast
    private let processingQueue = DispatchQueue(
        label: "com.ureyebuddy.pipeline",
        qos: .userInteractive
    )
    private var isProcessing = false

    // MARK: - Lifecycle

    func start() {
        faceRecognition.loadModel()
        hapticService.prepare()
        cameraService.delegate = self
        cameraService.configure()
        cameraService.start()
        isRunning = true
    }

    func stop() {
        cameraService.stop()
        isRunning = false
        DispatchQueue.main.async {
            self.recognisedContacts = []
        }
    }

    // MARK: - Pipeline

    private func processFrame(_ sampleBuffer: CMSampleBuffer) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        // 1. Detect faces
        let faces = faceDetection.detectFaces(in: pixelBuffer)
        guard !faces.isEmpty else {
            DispatchQueue.main.async {
                self.recognisedContacts = []
            }
            return
        }

        // 2. Recognise against enrolled contacts
        let contacts = contactStore.loadContacts()
        guard !contacts.isEmpty else { return }

        var matches: [FaceRecognitionService.MatchResult] = []
        for face in faces {
            if let match = faceRecognition.match(face: face, against: contacts) {
                matches.append(match)
                // 3. Haptic feedback — instant!
                hapticService.fireRecognitionHaptic(
                    for: match.contactID,
                    similarity: match.similarity
                )
            }
        }

        DispatchQueue.main.async {
            self.recognisedContacts = matches
        }
    }
}

// MARK: - CameraServiceDelegate

extension RecognitionPipeline: CameraServiceDelegate {
    func cameraService(_ service: CameraService, didOutput sampleBuffer: CMSampleBuffer) {
        // Throttle: skip frames if we're still within the interval
        let now = Date()
        guard now.timeIntervalSince(lastProcessedTime) >= frameInterval,
              !isProcessing
        else { return }

        lastProcessedTime = now
        isProcessing = true

        processingQueue.async { [weak self] in
            self?.processFrame(sampleBuffer)
            self?.isProcessing = false
        }
    }
}
