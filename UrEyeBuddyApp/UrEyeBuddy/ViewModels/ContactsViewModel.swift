import SwiftUI
import Combine

/// View model for managing enrolled contacts.
@MainActor
final class ContactsViewModel: ObservableObject {

    @Published var contacts: [Contact] = []

    private let store = ContactStore.shared

    func load() {
        contacts = store.loadContacts()
    }

    func delete(at offsets: IndexSet) {
        for index in offsets {
            store.delete(contactID: contacts[index].id)
        }
        load()
    }

    func delete(contact: Contact) {
        store.delete(contactID: contact.id)
        load()
    }
}
