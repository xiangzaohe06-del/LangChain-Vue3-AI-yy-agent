const SESSION_KEY = 'dietitian.conversation.v1'
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

type SessionStore = Pick<Storage, 'getItem' | 'setItem'>

export function createId(): string {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  bytes[6] = (bytes[6]! & 0x0f) | 0x40
  bytes[8] = (bytes[8]! & 0x3f) | 0x80
  const hex = Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

export function getConversationId(storage?: SessionStore): string {
  try {
    storage ??= window.sessionStorage
    const saved = storage.getItem(SESSION_KEY)
    if (saved && UUID.test(saved)) return saved
  } catch { /* Storage can be unavailable in private browser contexts. */ }
  return createConversation(storage)
}

export function createConversation(storage?: SessionStore): string {
  const id = createId()
  try {
    storage ??= window.sessionStorage
    storage.setItem(SESSION_KEY, id)
  } catch { /* The current view still retains its in-memory conversation ID. */ }
  return id
}
