import SwiftUI

/// Displays all enrolled contacts and allows deletion.
struct ContactsListView: View {
    @StateObject private var viewModel = ContactsViewModel()
    @State private var showEnrollSheet = false

    var body: some View {
        NavigationStack {
            Group {
                if viewModel.contacts.isEmpty {
                    emptyState
                } else {
                    contactsList
                }
            }
            .navigationTitle("People")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        showEnrollSheet = true
                    } label: {
                        Image(systemName: "plus")
                    }
                }
            }
            .sheet(isPresented: $showEnrollSheet) {
                viewModel.load()
            } content: {
                EnrollFaceView()
            }
            .onAppear {
                viewModel.load()
            }
        }
    }

    // MARK: - Subviews

    private var contactsList: some View {
        List {
            ForEach(viewModel.contacts) { contact in
                HStack {
                    Image(systemName: "person.crop.circle.fill")
                        .font(.title)
                        .foregroundColor(.accentColor)
                    VStack(alignment: .leading) {
                        Text(contact.name)
                            .font(.headline)
                        Text("\(contact.embeddings.count) face sample(s)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .onDelete(perform: viewModel.delete)
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "person.crop.circle.badge.plus")
                .font(.system(size: 64))
                .foregroundColor(.secondary)
            Text("No people enrolled yet")
                .font(.headline)
            Text("Tap + to add someone so the app can recognise them.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
        }
    }
}
