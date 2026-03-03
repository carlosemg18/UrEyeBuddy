import Foundation

/// A saved person the user wants the app to recognise.
struct Contact: Identifiable, Codable {
    /// Unique identifier.
    let id: String
    /// Display name (e.g. "Mom", "Dr. Smith").
    var name: String
    /// One or more 128-d face embeddings for this person.
    var embeddings: [[Float]]
    /// When this contact was first enrolled.
    let dateAdded: Date

    init(name: String, embeddings: [[Float]] = []) {
        self.id = UUID().uuidString
        self.name = name
        self.embeddings = embeddings
        self.dateAdded = Date()
    }
}
