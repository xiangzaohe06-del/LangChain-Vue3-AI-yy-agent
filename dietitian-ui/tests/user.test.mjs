import test from 'node:test'
import assert from 'node:assert/strict'
import {getUserId} from '../src/utils/user.ts'
import {createConversation} from '../src/utils/conversation.ts'

function store() {
  const items = new Map()
  return {getItem: key => items.get(key) ?? null, setItem: (key, value) => items.set(key, value)}
}

test('new conversations keep the same persistent user identity', () => {
  const persistent = store()
  const session = store()
  const user = getUserId(persistent)
  const first = createConversation(session)
  const second = createConversation(session)
  assert.notEqual(first, second)
  assert.equal(getUserId(persistent), user)
  assert.notEqual(getUserId(store()), user)
})

test('blocked local storage uses one in-memory identity for this page', () => {
  const broken = {getItem: () => { throw Error('disabled') }, setItem: () => { throw Error('disabled') }}
  assert.equal(getUserId(broken), getUserId(broken))
})
