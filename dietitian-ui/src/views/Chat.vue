<script setup lang="ts">
import {nextTick, onMounted, ref, watch} from 'vue'
import type {IChatMessage} from "@/types/chat";
import type {IAgentChat} from "@/types/agent";
import {chatStream, clearChatHistory, getChatHistory} from "@/api/chat";
import {getPresignedUrl, uploadCosFile} from "@/api/file.ts";
import type {IPresign} from "@/types/file.ts";
import {createConversation, getConversationId} from '@/utils/conversation';
import {getUserId} from '@/utils/user';
import MemoryPanel from '@/components/MemoryPanel.vue';

const textareaRef = ref<HTMLTextAreaElement | null>(null)

// 最大高度限制 (px)
const maxHeight = window.innerHeight * 0.5

// 动态调整高度
const adjustHeight = () => {
  const el = textareaRef.value
  if (!el) return

  // 先重置高度
  el.style.height = 'auto'

  // 根据滚动高度动态计算
  if (el.scrollHeight > maxHeight) {
    el.style.height = `${maxHeight}px`
    el.style.overflowY = 'auto'
  } else {
    el.style.height = `${el.scrollHeight}px`
    el.style.overflowY = 'hidden'
  }
}


// 聊天内容实体
const chat = ref<IAgentChat>({
  message: "",
  thread_id: getConversationId(),
  user_id: getUserId(),
  image_url: []
})
const sending = ref(false)
const loadingHistory = ref(true)
const showMemory = ref(false)

/**
 * 发送消息
 */
const sendMessage = async () => {
  if (sending.value || loadingHistory.value) return;
  if (!chat.value.message.trim() && (!chat.value.image_url || chat.value.image_url.length === 0)) return;
  if (chat.value.message.length > 8000 || (chat.value.image_url?.length ?? 0) > 4) {
    alert('每条消息最多8000字符、4张图片，请分次发送。');
    return;
  }
  sending.value = true;
  showMemory.value = false;

  // 压入用户自己的消息
  chatMessage.value.push({
    role: 'user',
    message: chat.value.message,
    image_url: chat.value.image_url
  });

  console.log(chatMessage.value)

  // 临时保存用户输入
  const userPrompt = chat.value.message;
  const userImages = chat.value.image_url?.slice() || []; // 拷贝一份
  chat.value.message = '';
  chat.value.image_url = [];

  // 预先压入 AI 空消息
  chatMessage.value.push({role: 'assistant', message: ''});
  const lastIndex = chatMessage.value.length - 1;

  // 调用流式接口，传入 image_url
  try {
    await chatStream(
      {...chat.value, message: userPrompt, image_url: userImages},
      (chunk) => {
        chatMessage.value[lastIndex].message += chunk;
      },
      () => {
        console.log('AI 回复渲染完毕');
      },
      (err) => {
        console.error('流式请求失败:', err);
        chatMessage.value[lastIndex].message = '网络出错了，请稍后再试。';
      }
    );
  } finally {
    sending.value = false;
  }
}

// 聊天消息列表
const chatMessage = ref<IChatMessage[]>([])

const chatContainerRef = ref<HTMLElement | null>(null)

// 监听自动跳转到底部
watch(() => chatMessage.value,
    async () => {
      // 等待 Vue 把新蹦出来的字渲染到 DOM 树中
      await nextTick()
      chatContainerRef.value!.scrollTo({top: chatContainerRef.value!.scrollHeight, behavior: "auto"});
    },
    {deep: true}
)

/**
 * 获取会话的历史记录
 */
const getThreadHistory = async () => {
  loadingHistory.value = true;
  const thread = chat.value.thread_id;
  try {
    const response = await getChatHistory(thread);
    if (thread === chat.value.thread_id) chatMessage.value = response.messages;
  } finally {
    loadingHistory.value = false;
  }
}

/**
 * 清空历史
 */
const clearHistory = async () => {
  if (sending.value || loadingHistory.value) return;
  loadingHistory.value = true;
  try {
    const response = await clearChatHistory(chat.value.thread_id);
    if (response.success) {
      chatMessage.value = [];
      chat.value.thread_id = createConversation();
    }
  } finally {
    loadingHistory.value = false;
  }
}

const newConversation = () => {
  if (sending.value || loadingHistory.value) return;
  chat.value.thread_id = createConversation();
  chatMessage.value = [];
  chat.value.message = '';
  chat.value.image_url = [];
}

/**
 * 图片上传
 */
const handleFile = async (event: any) => {
  const files = event.target.files

  for (let i = 0; i < files.length; i++) {
    const file = files[i]
    if ((chat.value.image_url?.length ?? 0) >= 4) {
      alert('每条消息最多4张图片。');
      break;
    }
    // 判断是否为图片
    if (file.type.startsWith('image/')) {
      const presign: IPresign = await getPresignedUrl(file.name)
      await uploadCosFile(presign.uploadUrl, file);
      chat.value.image_url?.push(presign.accessUrl)
    } else {
      alert("只能上传图片")
    }
  }

}


onMounted(() => {
  getThreadHistory()
})

</script>

<template>
  <div class="dietitian-chat">
    <div class="chat__header">
      <img class="chat__header-logo" src="@/assets/logo.png" alt="logo"/>
      <span>私人营养师</span>
      <div class="chat__header-actions">
      <button class="chat__header-clear" :disabled="sending || loadingHistory" @click="showMemory = !showMemory">长期记忆</button>
      <button class="chat__header-clear" :disabled="sending || loadingHistory" @click="newConversation">新对话</button>
      <button class="chat__header-clear" :disabled="sending || loadingHistory" @click="clearHistory">清空记录</button>
      </div>
    </div>
    <MemoryPanel v-if="showMemory" @close="showMemory = false" />
    <div class="chat__content" ref="chatContainerRef">
      <div v-for="(item,index) in chatMessage" :key="index"
           :class="`chat__content-wrapper chat__message--${item.role}`">
        <div class="chat__content--ai-avatar" v-if="item.role=='assistant'">
          <img width="100%" alt="头像" src="@/assets/logo.png">
        </div>
        <!--    内容展示        -->
        <div class="chat__content-message">
          <div v-text="item.message" class="chat__content-text"/>
          <div class="chat__content-img">
            <div class="chat__content-img-wrapper" v-for="(img,index) in item.image_url" :key="index">
              <img :src="img" alt="图片" width="100%">
            </div>
          </div>
        </div>
      </div>

      <div class="chat__content-default" v-if="chatMessage.length==0">
        <img class="chat__content-default-logo" src="@/assets/logo.png" alt="logo">
        <span class="chat__content-default-title">欢迎使用AI营养师</span>
        <span class="chat__content-default-description">我可以帮您分析食物营养、制定饮食计划、提供健康建议</span>
      </div>

    </div>
    <div class="chat__actions">
      <!--  图片列表      -->
      <div class="chat__file">
        <div class="chat__file-imgbox" v-for="(item,index) in chat.image_url" :key="index">
          <img :src="item" alt="图片" width="100%"/>
        </div>
      </div>
      <!--  内容输入部分      -->
      <div class="chat__input-wrapper">
        <button class="chat__actions-upload" @click="($refs.fileRef as HTMLInputElement).click()">
          <img width="100%" src="@/assets/add.png" alt="上传">
        </button>
        <input ref="fileRef" type="file" accept="image/*" multiple style="display: none" @change="handleFile">
        <textarea @keydown.prevent.enter="sendMessage" @input="adjustHeight" v-model="chat.message" ref="textareaRef"
                  class="chat__actions-input" placeholder="请输入消息..." rows="1"></textarea>

        <button class="chat__actions-send" :disabled="sending || loadingHistory" @click="sendMessage">
          <img width="100%" src="@/assets/send.png" alt="发送">
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
.dietitian-chat {
  background: linear-gradient(
          to bottom right,
          #f0fdf4,
          #eff6ff
  );
  display: flex;
  flex-direction: column;
  height: 100vh;
}

// 聊天头部展示
.chat__header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 15px;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1),
  0 1px 2px -1px rgba(0, 0, 0, 0.1);
  padding: 15px 20px;
  background-color: white;
  border-bottom: 1px solid #ebe6e7;
  font-size: 18px;
  font-weight: bold;
  color: #474747;
}

// 头部logo
.chat__header-logo {
  width: 30px;
}

// 清空记录按钮
.chat__header-actions {
  margin-left: auto;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.chat__header-clear {
  background-color: #10b981;
  border: none;
  color: white;
  white-space: nowrap;
  padding: 6px 15px;
  font-weight: bold;
  border-radius: 10px;
}

// 默认内容部分
.chat__content-default {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

// 默认内容的logo
.chat__content-default-logo {
  width: 100px;
}

// 默认内容标题
.chat__content-default-title {
  color: #333333;
  font-weight: bold;
  font-size: 24px;
}

// 默认内容描述
.chat__content-default-description {
  color: #6a7282;
  font-size: 16px;
  font-weight: bold;
}


// 聊天内容部分
.chat__content {
  overflow: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  align-items: flex-start;
  flex: 1;
}

.chat__content-wrapper {
  display: flex;
  gap: 15px;
}

// 聊天内容的消息
.chat__content-message {
  gap: 10px;
  padding: 15px;
  border-radius: 15px;
  font-size: 16px;
  box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1);
  max-width: 50vw;
  white-space: pre-wrap;
  word-break: break-all;
}


// 为空显示正在思考
.chat__message--assistant .chat__content-text:empty::after {
  content: "正在思考...";
  display: inline-block;
  color: #999;
  font-style: italic;
  font-size: 14px;
}

// 底部图片部分
.chat__content-img {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 10px;
  flex-wrap: wrap;
}

.chat__content-img-wrapper {
  background-color: white;
  border-radius: 5px;
  border: 1px solid #eeeeee;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  overflow: hidden;
}

// ai头像
.chat__content--ai-avatar {
  border-radius: 50%;
  width: 50px;
  height: 50px;
  line-height: 100px;
}

// 系统聊天消息
.chat__message--assistant {
  align-self: flex-start;
}

// 系统聊天消息内容
.chat__message--assistant > .chat__content-message {
  background-color: white;
}

// 用户部分聊天
.chat__message--user {
  align-self: flex-end;
}

// 用户聊天消息内容
.chat__message--user > .chat__content-message {
  color: white;
  background-image: linear-gradient(to right, #3b82f6, #2563eb);
}

// 聊天的操作部分
.chat__actions {
  display: flex;
  flex-direction: column;
  width: 90vw;
  border-radius: 15px;
  background-color: #ffffff;
  padding: 10px 20px;
  margin: 10px auto 20px;
}

// 文件部分展示
.chat__file {
  display: flex;
  align-items: flex-start;
  justify-content: flex-start;
  gap: 10px;
}

// 图片边框
.chat__file-imgbox {
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #eeeeee;
  overflow: hidden;
  width: 40px;
  height: 40px;
  border-radius: 5px;
  position: relative;
}

// 聊天内容输入
.chat__input-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 15px;
  width: 100%;
}

// 聊天输入的上传图片按钮
.chat__actions-upload {
  padding: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  border-radius: 50%;
  width: 35px;
  height: 35px;
  background-color: transparent;
  cursor: pointer;

  &:hover {
    background-color: #f5f5f5;
  }
}

// 聊天输入的输入框
.chat__actions-input {
  flex: 1;
  outline: none;
  border: none;
  resize: none;
  box-sizing: border-box;
  font-size: 16px;
  padding: 10px 0;
  line-height: 1.5;
  min-height: 38px; /* 初始高度 */
}


// 发送按钮样式
.chat__actions-send {
  padding: 10px;
  border-radius: 50%;
  background-color: #eee;
  width: 35px;
  height: 35px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  cursor: pointer;

  &:hover {
    background-color: #ddd;
  }
}


</style>
