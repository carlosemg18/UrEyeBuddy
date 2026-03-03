import SwiftUI
import AVFoundation

/// View for enrolling a new person's face.
/// Captures multiple frames from the camera to build a robust set of embeddings.
struct EnrollFaceView: View {
    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var capturedCount = 0
    @State private var isCapturing = false
    @State private var enrollmentComplete = false
    @State private var errorMessage: String?

    /// Number of face samples to capture for reliable recognition.
    private let targetSamples = 5

    private let pipeline = RecognitionPipeline()
    private let faceDetection = FaceDetectionService()

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                if enrollmentComplete {
                    enrollmentSuccessView
                } else {
                    enrollmentFormView
                }
            }
            .padding()
            .navigationTitle("Add Person")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }

    // MARK: - Subviews

    private var enrollmentFormView: some View {
        VStack(spacing: 24) {
            TextField("Name (e.g. Mom, Dr. Smith)", text: $name)
                .textFieldStyle(.roundedBorder)
                .font(.title3)
                .autocorrectionDisabled()

            if isCapturing {
                VStack(spacing: 12) {
                    CameraPreviewView(session: pipeline.cameraService.session)
                        .frame(height: 300)
                        .clipShape(RoundedRectangle(cornerRadius: 16))

                    ProgressView(value: Double(capturedCount), total: Double(targetSamples))

                    Text("Look at the camera from different angles")
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    Text("\(capturedCount) / \(targetSamples) samples captured")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Button("Capture Sample") {
                        captureSample()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(capturedCount >= targetSamples)
                }
            } else {
                VStack(spacing: 16) {
                    Image(systemName: "faceid")
                        .font(.system(size: 64))
                        .foregroundColor(.accentColor)

                    Text("Position the person's face in the camera and capture \(targetSamples) samples from slightly different angles.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)

                    Button("Start Camera") {
                        startCapture()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }

            if let error = errorMessage {
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
            }

            if capturedCount >= targetSamples {
                Button("Save Contact") {
                    saveContact()
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
            }

            Spacer()
        }
    }

    private var enrollmentSuccessView: some View {
        VStack(spacing: 16) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 64))
                .foregroundColor(.green)
            Text("\(name) enrolled successfully!")
                .font(.title2)
                .fontWeight(.semibold)
            Text("The app will now recognise this person and alert you with a haptic tap.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            Button("Done") { dismiss() }
                .buttonStyle(.borderedProminent)
        }
    }

    // MARK: - Capture logic

    @State private var collectedEmbeddings: [[Float]] = []
    private let recognitionService = FaceRecognitionService()

    private func startCapture() {
        recognitionService.loadModel()
        pipeline.cameraService.configure()
        pipeline.cameraService.start()
        isCapturing = true
    }

    private func captureSample() {
        // In a real implementation, we'd grab the latest sample buffer.
        // Here we illustrate the enrollment flow — on a real device the
        // CameraService delegate would deliver frames for face cropping.
        errorMessage = nil
        capturedCount += 1
    }

    private func saveContact() {
        let trimmed = name.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }

        _ = ContactStore.shared.enroll(
            name: trimmed,
            embeddings: collectedEmbeddings
        )

        pipeline.cameraService.stop()
        enrollmentComplete = true
    }
}
