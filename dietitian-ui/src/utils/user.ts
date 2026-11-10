import {createId} from './conversation.ts'

const USER_KEY = 'dietitian.user.v1'
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
type UserStore = Pick<Storage, 'getItem' | 'setItem'>
let temporaryId: string | undefined

export function getUserId(storage?: UserStore): string {
  try {
    storage ??= window.localStorage
    const saved = storage.getItem(USER_KEY)
    if (saved && UUID.test(saved)) return saved
    const id = createId()
    storage.setItem(USER_KEY, id)
    return id
  } catch {
    temporaryId ??= createId()
    return temporaryId
  }
}

export function userHeaders() {
  return {'X-User-Id': getUserId()}
}
