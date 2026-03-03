import SwiftUI

/// Root view with tab navigation between the live camera and contacts list.
struct ContentView: View {
    var body: some View {
        TabView {
            CameraView()
                .tabItem {
                    Label("Recognise", systemImage: "eye.fill")
                }

            ContactsListView()
                .tabItem {
                    Label("People", systemImage: "person.2.fill")
                }

            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
        }
    }
}
