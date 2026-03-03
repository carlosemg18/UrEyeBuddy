import SwiftUI

/// Main camera view showing the live feed with recognition overlays.
struct CameraView: View {
    @StateObject private var viewModel = CameraViewModel()

    var body: some View {
        ZStack {
            // Camera preview
            if viewModel.cameraPermissionGranted {
                CameraPreviewView(session: viewModel.pipeline.cameraService.session)
                    .ignoresSafeArea()
            } else {
                cameraPermissionPlaceholder
            }

            // Recognition overlay
            VStack {
                Spacer()
                if !viewModel.recognisedNames.isEmpty {
                    recognitionBanner
                }
            }
            .padding(.bottom, 40)

            // Status indicator
            VStack {
                HStack {
                    Spacer()
                    statusIndicator
                        .padding()
                }
                Spacer()
            }
        }
        .onAppear {
            viewModel.checkCameraPermission()
            viewModel.startPipeline()
        }
        .onDisappear {
            viewModel.stopPipeline()
        }
    }

    // MARK: - Subviews

    private var recognitionBanner: some View {
        VStack(spacing: 8) {
            ForEach(viewModel.recognisedNames, id: \.self) { name in
                HStack {
                    Image(systemName: "person.fill.checkmark")
                    Text(name)
                        .font(.title2)
                        .fontWeight(.semibold)
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 12)
                .background(.ultraThinMaterial, in: Capsule())
            }
        }
        .transition(.move(edge: .bottom).combined(with: .opacity))
        .animation(.easeInOut(duration: 0.3), value: viewModel.recognisedNames)
    }

    private var statusIndicator: some View {
        Circle()
            .fill(viewModel.isRunning ? Color.green : Color.red)
            .frame(width: 12, height: 12)
            .overlay(
                Circle()
                    .stroke(Color.white, lineWidth: 2)
            )
    }

    private var cameraPermissionPlaceholder: some View {
        VStack(spacing: 16) {
            Image(systemName: "camera.fill")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            Text("Camera access is required")
                .font(.headline)
            Text("Go to Settings > UrEyeBuddy to enable camera access.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
    }
}
