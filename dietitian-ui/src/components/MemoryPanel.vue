<script setup lang="ts">
import {onMounted, reactive, ref} from 'vue'
import {clearMemory, getMemory, saveMemory, type MemoryProfile} from '@/api/memory'

const emit = defineEmits<{close: []}>()
const form = reactive({allergies: '', avoids: '', likes: '', diet: '', goals: '', budget: ''})
const revision = ref(0)
const busy = ref(false)
const message = ref('')
const failed = ref(false)

function apply(profile: MemoryProfile) {
  revision.value = profile.revision
  form.allergies = profile.facts.allergies.join('、')
  form.avoids = profile.facts.avoids.join('、')
  form.likes = profile.facts.likes.join('、')
  form.diet = profile.facts.diet
  form.goals = profile.facts.goals.join('、')
  form.budget = profile.facts.budget
}

function errorText(error: unknown) {
  const detail = (error as {response?: {data?: {detail?: unknown}}}).response?.data?.detail
  return typeof detail === 'string' ? detail : '操作失败，请检查输入或稍后重试。'
}

async function action(operation: () => Promise<MemoryProfile>, success: string) {
  if (busy.value) return
  busy.value = true
  failed.value = false
  message.value = ''
  try {
    apply(await operation())
    message.value = success
  } catch (error) {
    failed.value = true
    message.value = errorText(error)
  } finally {
    busy.value = false
  }
}

const list = (text: string) => Array.from(new Set(text.split(/[，,、\n]/).map(value => value.trim()).filter(Boolean)))
const load = () => action(getMemory, '')
const save = () => {
  const facts = {
    allergies: list(form.allergies), avoids: list(form.avoids), likes: list(form.likes),
    diet: form.diet.trim(), goals: list(form.goals), budget: form.budget.trim(),
  }
  if ([facts.allergies, facts.avoids, facts.likes, facts.goals].some(items =>
    items.length > 12 || items.some(item => item.length > 60))) {
    failed.value = true
    message.value = '每类最多12项，每项最多60个字符。'
    return
  }
  return action(() => saveMemory(facts, revision.value), '已保存，新对话会继续使用这些信息。')
}
const clear = () => action(clearMemory, '长期记忆已清空，聊天记录保留。')
onMounted(load)
</script>

<template>
  <section class="memory-panel" aria-label="长期记忆" :aria-busy="busy">
    <div class="memory-panel__heading">
      <strong>长期记忆</strong>
      <button type="button" @click="emit('close')">关闭</button>
    </div>
    <p>只保存你主动填写或明确要求记住的信息。本浏览器的新对话会继续使用，清空聊天不会删除。多项用顿号分开，每类最多12项。</p>
    <form @submit.prevent="save">
      <fieldset :disabled="busy">
        <label>过敏食材<input v-model="form.allergies" placeholder="例如：花生、虾" maxlength="720"></label>
        <label>忌口或不喜欢<input v-model="form.avoids" placeholder="例如：香菜" maxlength="720"></label>
        <label>喜欢的食材<input v-model="form.likes" placeholder="例如：豆腐、西兰花" maxlength="720"></label>
        <label>饮食方式<input v-model="form.diet" placeholder="例如：素食" maxlength="80"></label>
        <label>饮食目标<input v-model="form.goals" placeholder="填写你自己的目标" maxlength="720"></label>
        <label>常用预算<input v-model="form.budget" placeholder="例如：每餐30元" maxlength="80"></label>
      </fieldset>
      <div class="memory-panel__actions">
        <button type="submit" :disabled="busy">{{ busy ? '处理中…' : '保存记忆' }}</button>
        <button type="button" :disabled="busy" @click="load">重新加载</button>
        <button type="button" :disabled="busy" @click="clear">清空长期记忆</button>
      </div>
    </form>
    <p v-if="message" role="status" :class="{'memory-panel__error': failed}">{{ message }}</p>
  </section>
</template>

<style scoped>
.memory-panel { padding: 16px 24px; background: #f4f8f6; border-bottom: 1px solid #dce5df; max-height: 55vh; overflow: auto; }
.memory-panel__heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.memory-panel p { font-size: 13px; line-height: 1.6; margin: 8px 0; color: #52645a; }
.memory-panel fieldset { border: 0; padding: 0; margin: 12px 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.memory-panel label { display: flex; flex-direction: column; gap: 5px; font-size: 13px; }
.memory-panel input { width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid #cbd8ce; border-radius: 6px; background: white; }
.memory-panel button { border: 1px solid #cbd8ce; padding: 6px 10px; border-radius: 6px; background: white; cursor: pointer; }
.memory-panel button:disabled { opacity: .6; cursor: wait; }
.memory-panel__actions { display: flex; gap: 8px; flex-wrap: wrap; }
.memory-panel .memory-panel__error { color: #a12c2c; }
@media (max-width: 560px) { .memory-panel { padding: 12px; } .memory-panel fieldset { grid-template-columns: 1fr; } }
</style>
