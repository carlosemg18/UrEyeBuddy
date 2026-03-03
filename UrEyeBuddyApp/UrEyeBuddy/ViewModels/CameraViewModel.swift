import SwiftUI
import Combine
import AVFoundation

/// View model that bridges the RecognitionPipeline to the SwiftUI camera view.
@MainActor
final class CameraViewModel: ObservableObject {

    @Published var recognisedNames: [String] = []
    @Published var isRunning = false
    @Published var cameraPermissionGranted = false

    let pipeline = RecognitionPipeline()
    private var cancellables = Set<AnyCancellable>()

    init() {
        pipeline.$recognisedContacts
            .receive(on: DispatchQueue.main)
            .map { matches in matches.map { $0.contactName } }
            .assign(to: &$recognisedNames)

        pipeline.$isRunning
            .receive(on: DispatchQueue.main)
            .assign(to: &$isRunning)
    }

    // MARK: - Permissions

    func checkCameraPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            cameraPermissionGranted = true
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                Task { @MainActor in
                    self.cameraPermissionGranted = granted
                }
            }
        default:
            cameraPermissionGranted = false
        }
    }

    // MARK: - Control

    func startPipeline() {
        guard cameraPermissionGranted else { return }
        pipeline.start()
    }

    func stopPipeline() {
        pipeline.stop()
    }
}
