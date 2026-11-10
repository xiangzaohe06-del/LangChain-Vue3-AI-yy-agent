import test from 'node:test'
import assert from 'node:assert/strict'
import {getConversationId, createConversation} from '../src/utils/conversation.ts'

function storage(initial) {
  let value = initial
  return {getItem: () => value, setItem: (_key, next) => { value = next }}
}

test('reload reuses this browser session, another session gets a different ID', () => {
  const first = storage(null)
  const second = storage(null)
  const id = getConversationId(first)
  assert.equal(getConversationId(first), id)
  assert.notEqual(getConversationId(second), id)
  assert.match(id, /^[0-9a-f-]{36}$/)
})

test('legacy shared ID is not reused and new conversation rotates only this session', () => {
  const first = storage('001')
  const other = storage(null)
  const id = getConversationId(first)
  const otherId = getConversationId(other)
  assert.notEqual(id, '001')
  const next = createConversation(first)
  assert.notEqual(next, id)
  assert.equal(getConversationId(first), next)
  assert.equal(getConversationId(other), otherId)
})

test('unavailable session storage still produces a valid conversation ID', () => {
  const broken = {getItem: () => { throw Error('disabled') }, setItem: () => { throw Error('disabled') }}
  assert.match(getConversationId(broken), /^[0-9a-f-]{36}$/)
})
