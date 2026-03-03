import CoreHaptics
import UIKit

/// Delivers discreet haptic feedback when a known face is recognised.
///
/// Uses CoreHaptics for precise, custom vibration patterns that feel like
/// a subtle tap — unobtrusive during conversation.
final class HapticService {

    private var engine: CHHapticEngine?
    private var supportsHaptics = false

    /// Minimum interval between haptic pulses for the same person (seconds).
    /// Prevents continuous buzzing while a face stays in frame.
    var cooldownInterval: TimeInterval = 2.0

    /// Tracks the last haptic time per contact to enforce cooldown.
    private var lastHapticTime: [String: Date] = [:]

    // MARK: - Setup

    func prepare() {
        guard CHHapticEngine.capabilitiesForHardware().supportsHaptics else {
            supportsHaptics = false
            return
        }
        supportsHaptics = true

        do {
            engine = try CHHapticEngine()
            engine?.resetHandler = { [weak self] in
                do {
                    try self?.engine?.start()
                } catch {
                    print("[Haptics] Engine restart failed: \(error)")
                }
            }
            try engine?.start()
        } catch {
            print("[Haptics] Engine creation failed: \(error)")
            supportsHaptics = false
        }
    }

    // MARK: - Feedback

    /// Fire a recognition haptic for a given contact, respecting cooldown.
    /// - Parameters:
    ///   - contactID: The ID of the recognised contact.
    ///   - similarity: Match confidence (0–1). Higher = stronger haptic.
    func fireRecognitionHaptic(for contactID: String, similarity: Float) {
        // Cooldown check
        if let last = lastHapticTime[contactID],
           Date().timeIntervalSince(last) < cooldownInterval {
            return
        }
        lastHapticTime[contactID] = Date()

        if supportsHaptics {
            playCoreHaptic(intensity: similarity)
        } else {
            playFallbackHaptic()
        }
    }

    /// Clear cooldown state (e.g. when the app returns to foreground).
    func resetCooldowns() {
        lastHapticTime.removeAll()
    }

    // MARK: - Core Haptics pattern

    private func playCoreHaptic(intensity: Float) {
        guard let engine else { return }

        // Two quick taps: a "someone is here" pattern
        let sharpness = CHHapticEventParameter(
            parameterID: .hapticSharpness, value: 0.6
        )
        let intensityParam = CHHapticEventParameter(
            parameterID: .hapticIntensity, value: min(intensity + 0.3, 1.0)
        )

        let tap1 = CHHapticEvent(
            eventType: .hapticTransient,
            parameters: [intensityParam, sharpness],
            relativeTime: 0
        )
        let tap2 = CHHapticEvent(
            eventType: .hapticTransient,
            parameters: [intensityParam, sharpness],
            relativeTime: 0.12
        )

        do {
            let pattern = try CHHapticPattern(events: [tap1, tap2], parameters: [])
            let player = try engine.makePlayer(with: pattern)
            try player.start(atTime: CHHapticTimeImmediate)
        } catch {
            print("[Haptics] Playback failed: \(error)")
        }
    }

    private func playFallbackHaptic() {
        let generator = UINotificationFeedbackGenerator()
        generator.notificationOccurred(.success)
    }
}
