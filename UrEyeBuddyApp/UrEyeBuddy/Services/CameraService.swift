import AVFoundation
import UIKit

/// Delegate that receives camera frames for processing.
protocol CameraServiceDelegate: AnyObject {
    func cameraService(_ service: CameraService, didOutput sampleBuffer: CMSampleBuffer)
}

/// Manages the AVCaptureSession for the front-facing camera.
/// Delivers pixel buffers at 30 fps for downstream face detection.
final class CameraService: NSObject {

    weak var delegate: CameraServiceDelegate?

    let session = AVCaptureSession()
    private let outputQueue = DispatchQueue(label: "com.ureyebuddy.camera", qos: .userInteractive)

    private var isConfigured = false

    // MARK: - Setup

    func configure() {
        guard !isConfigured else { return }

        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        // Front camera for face recognition
        guard let camera = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .front
        ) else {
            print("[CameraService] No front camera available")
            session.commitConfiguration()
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: camera)
            if session.canAddInput(input) {
                session.addInput(input)
            }
        } catch {
            print("[CameraService] Camera input error: \(error)")
            session.commitConfiguration()
            return
        }

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: outputQueue)

        if session.canAddOutput(output) {
            session.addOutput(output)
        }

        // Lock to 30 fps to balance latency and battery
        if let connection = output.connection(with: .video) {
            connection.videoOrientation = .portrait
        }
        do {
            try camera.lockForConfiguration()
            camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
            camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
            camera.unlockForConfiguration()
        } catch {
            print("[CameraService] Frame rate lock failed: \(error)")
        }

        session.commitConfiguration()
        isConfigured = true
    }

    // MARK: - Session control

    func start() {
        guard isConfigured, !session.isRunning else { return }
        outputQueue.async { [weak self] in
            self?.session.startRunning()
        }
    }

    func stop() {
        guard session.isRunning else { return }
        outputQueue.async { [weak self] in
            self?.session.stopRunning()
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraService: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        delegate?.cameraService(self, didOutput: sampleBuffer)
    }
}
