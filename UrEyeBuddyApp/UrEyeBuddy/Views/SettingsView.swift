import SwiftUI

/// App settings for tuning recognition and haptic behaviour.
struct SettingsView: View {
    @AppStorage("similarityThreshold") private var threshold: Double = 0.65
    @AppStorage("hapticCooldown") private var cooldown: Double = 2.0
    @AppStorage("processingFPS") private var fps: Double = 10.0

    var body: some View {
        NavigationStack {
            Form {
                Section("Recognition") {
                    VStack(alignment: .leading) {
                        Text("Similarity Threshold: \(threshold, specifier: "%.2f")")
                        Slider(value: $threshold, in: 0.4...0.9, step: 0.05)
                    }
                    Text("Higher values = fewer false positives but may miss valid matches.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Section("Haptic Feedback") {
                    VStack(alignment: .leading) {
                        Text("Cooldown: \(cooldown, specifier: "%.1f")s")
                        Slider(value: $cooldown, in: 0.5...10.0, step: 0.5)
                    }
                    Text("Minimum seconds between haptic taps for the same person.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Section("Performance") {
                    VStack(alignment: .leading) {
                        Text("Analysis FPS: \(Int(fps))")
                        Slider(value: $fps, in: 5...30, step: 1)
                    }
                    Text("Higher values increase responsiveness but use more battery.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Section("Privacy") {
                    Label("All processing happens on-device", systemImage: "lock.shield.fill")
                    Label("No images leave your phone", systemImage: "iphone.gen3")
                    Label("Face data stored only locally", systemImage: "externaldrive.fill")
                }
                .foregroundColor(.secondary)

                Section("About") {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }
}
