import Foundation

/// Persists enrolled contacts and their face embeddings to local storage.
/// All data stays on-device — nothing is sent to the cloud.
final class ContactStore {

    static let shared = ContactStore()

    private let fileManager = FileManager.default
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    private var storageURL: URL {
        let docs = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("contacts.json")
    }

    private init() {}

    // MARK: - CRUD

    /// Load all enrolled contacts from disk.
    func loadContacts() -> [Contact] {
        guard fileManager.fileExists(atPath: storageURL.path) else {
            return []
        }
        do {
            let data = try Data(contentsOf: storageURL)
            return try decoder.decode([Contact].self, from: data)
        } catch {
            print("[ContactStore] Load failed: \(error)")
            return []
        }
    }

    /// Save the full contact list to disk.
    func saveContacts(_ contacts: [Contact]) {
        do {
            let data = try encoder.encode(contacts)
            try data.write(to: storageURL, options: .atomic)
        } catch {
            print("[ContactStore] Save failed: \(error)")
        }
    }

    /// Add a new contact or append embeddings to an existing one.
    func enroll(name: String, embeddings: [[Float]]) -> Contact {
        var contacts = loadContacts()

        if let idx = contacts.firstIndex(where: { $0.name == name }) {
            contacts[idx].embeddings.append(contentsOf: embeddings)
            saveContacts(contacts)
            return contacts[idx]
        } else {
            let contact = Contact(name: name, embeddings: embeddings)
            contacts.append(contact)
            saveContacts(contacts)
            return contact
        }
    }

    /// Remove a contact by ID.
    func delete(contactID: String) {
        var contacts = loadContacts()
        contacts.removeAll { $0.id == contactID }
        saveContacts(contacts)
    }
}
