import request from '@/utils/request'
import {userHeaders} from '@/utils/user'

export interface MemoryFacts {
  allergies: string[]
  avoids: string[]
  likes: string[]
  diet: string
  goals: string[]
  budget: string
}

export interface MemoryProfile {
  facts: MemoryFacts
  revision: number
  updated_at: string | null
}

export function getMemory() {
  return request<MemoryProfile>({url: '/memory/profile', method: 'get', headers: userHeaders()})
}

export function saveMemory(facts: MemoryFacts, revision: number) {
  return request<MemoryProfile>({url: '/memory/profile', method: 'put', headers: userHeaders(), data: {facts, revision}})
}

export function clearMemory() {
  return request<MemoryProfile>({url: '/memory/profile', method: 'delete', headers: userHeaders()})
}
