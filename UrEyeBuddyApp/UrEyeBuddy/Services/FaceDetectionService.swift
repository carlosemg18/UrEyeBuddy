import Vision
import CoreImage
import UIKit

/// Result of face detection: the bounding box and the cropped face image.
struct DetectedFace {
    /// Normalised bounding box in Vision coordinates (origin bottom-left).
    let boundingBox: CGRect
    /// 160×160 cropped and aligned face ready for the embedding model.
    let croppedFace: CGImage
}

/// Uses Apple's Vision framework for fast, on-device face detection.
/// Returns cropped face images sized for the Core ML embedding model.
final class FaceDetectionService {

    /// Side length expected by the embedding model.
    static let faceInputSize = 160

    private let sequenceHandler = VNSequenceRequestHandler()

    /// Detect faces in a pixel buffer and return cropped face images.
    ///
    /// - Parameter pixelBuffer: The camera frame.
    /// - Returns: Array of detected faces with their crops.
    func detectFaces(in pixelBuffer: CVPixelBuffer) -> [DetectedFace] {
        let request = VNDetectFaceRectanglesRequest()
        request.revision = VNDetectFaceRectanglesRequestRevision3

        do {
            try sequenceHandler.perform(
                [request],
                on: pixelBuffer,
                orientation: .up
            )
        } catch {
            print("[FaceDetection] Detection failed: \(error)")
            return []
        }

        guard let results = request.results, !results.isEmpty else {
            return []
        }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        let context = CIContext(options: [.useSoftwareRenderer: false])

        return results.compactMap { observation in
            let box = observation.boundingBox

            // Convert Vision normalised rect (origin bottom-left) to pixel rect
            let x = box.origin.x * CGFloat(imageWidth)
            let y = (1 - box.origin.y - box.height) * CGFloat(imageHeight)
            let w = box.width * CGFloat(imageWidth)
            let h = box.height * CGFloat(imageHeight)

            // Add 20% padding for better embedding accuracy
            let padding: CGFloat = 0.2
            let padW = w * padding
            let padH = h * padding
            var cropRect = CGRect(
                x: x - padW,
                y: y - padH,
                width: w + padW * 2,
                height: h + padH * 2
            )

            // Clamp to image bounds
            cropRect = cropRect.intersection(
                CGRect(x: 0, y: 0, width: CGFloat(imageWidth), height: CGFloat(imageHeight))
            )

            guard !cropRect.isEmpty,
                  let croppedCG = context.createCGImage(
                      ciImage.cropped(to: cropRect),
                      from: cropRect
                  )
            else { return nil }

            // Resize to model input size (160×160)
            guard let resized = Self.resize(
                croppedCG,
                to: CGSize(
                    width: Self.faceInputSize,
                    height: Self.faceInputSize
                )
            ) else { return nil }

            return DetectedFace(boundingBox: box, croppedFace: resized)
        }
    }

    // MARK: - Helpers

    private static func resize(_ image: CGImage, to size: CGSize) -> CGImage? {
        guard let context = CGContext(
            data: nil,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(origin: .zero, size: size))
        return context.makeImage()
    }
}
