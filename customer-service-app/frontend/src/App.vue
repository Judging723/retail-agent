<script setup>
import { computed, nextTick, reactive, ref } from 'vue'

import { createConversation, getConversation, streamChatMessage } from './api'

const customerNo = ref('DEMO001')
const displayName = ref('演示用户')
const conversation = ref(null)
const messages = ref([])
const sourcesByMessage = ref({})
const draft = ref('')
const errorMessage = ref('')
const isStarting = ref(false)
const isSending = ref(false)
const messageList = ref(null)
const failedRequest = ref(null)

const suggestions = [
  '我想买苹果，哪一种更适合直接吃？',
  '想买一件防晒衣，帮我推荐一下。',
  '给我来两碗牛肉汤。',
]

const canStart = computed(() => customerNo.value.trim() && !isStarting.value)
const canSend = computed(
  () => conversation.value && draft.value.trim() && !isSending.value,
)
const isWaitingForFirstChunk = computed(
  () =>
    isSending.value &&
    !messages.value.some((message) => message.streaming && message.content),
)

function readableError(error) {
  return error instanceof Error ? error.message : '发生了未知错误'
}

async function scrollToBottom() {
  await nextTick()
  if (messageList.value) {
    messageList.value.scrollTop = messageList.value.scrollHeight
  }
}

async function refreshHistory() {
  const detail = await getConversation(conversation.value.conversation_no)
  conversation.value = detail
  messages.value = detail.messages.map((message) => ({
    ...message,
    sources: sourcesByMessage.value[message.message_id] || [],
  }))
  await scrollToBottom()
}

async function startConversation() {
  if (!canStart.value) return

  isStarting.value = true
  errorMessage.value = ''

  try {
    conversation.value = await createConversation(
      customerNo.value.trim(),
      displayName.value.trim(),
    )
    messages.value = []
    sourcesByMessage.value = {}
  } catch (error) {
    errorMessage.value = readableError(error)
  } finally {
    isStarting.value = false
  }
}

async function sendMessage() {
  if (!canSend.value) return

  const content = draft.value.trim()
  const idempotencyKey =
    failedRequest.value?.content === content
      ? failedRequest.value.idempotencyKey
      : crypto.randomUUID()
  failedRequest.value = null
  draft.value = ''
  errorMessage.value = ''
  isSending.value = true
  const temporaryId = Date.now()
  const userMessageId = `pending-user-${temporaryId}`
  const assistantMessageId = `pending-assistant-${temporaryId}`
  messages.value.push({
    message_id: userMessageId,
    role: 'user',
    content,
    sources: [],
    pending: true,
  })
  const assistantMessage = reactive({
    message_id: assistantMessageId,
    role: 'assistant',
    content: '',
    sources: [],
    streaming: true,
  })
  messages.value.push(assistantMessage)
  await scrollToBottom()

  try {
    let completed = false

    for await (const streamEvent of streamChatMessage(
      conversation.value.conversation_no,
      content,
      idempotencyKey,
    )) {
      if (streamEvent.event === 'sources') {
        assistantMessage.sources = streamEvent.data?.sources || []
      } else if (streamEvent.event === 'delta') {
        assistantMessage.content += streamEvent.data?.content || ''
      } else if (streamEvent.event === 'error') {
        throw new Error(streamEvent.data?.detail || '模型服务暂时不可用')
      } else if (streamEvent.event === 'done') {
        completed = true
        const reply = streamEvent.data
        sourcesByMessage.value = {
          ...sourcesByMessage.value,
          [reply.message_id]: reply.sources || assistantMessage.sources,
        }
        await refreshHistory()
      }

      await scrollToBottom()
    }

    if (!completed) {
      throw new Error('流式响应意外中断，请重试')
    }
  } catch (error) {
    messages.value = messages.value.filter(
      (message) =>
        message.message_id !== userMessageId &&
        message.message_id !== assistantMessageId,
    )
    draft.value = content
    failedRequest.value = { content, idempotencyKey }
    errorMessage.value = readableError(error)
  } finally {
    isSending.value = false
  }
}

function resetConversation() {
  if (isSending.value) return

  conversation.value = null
  messages.value = []
  sourcesByMessage.value = {}
  draft.value = ''
  errorMessage.value = ''
  failedRequest.value = null
}
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <a class="brand" href="#" aria-label="Retail Agent 首页">
        <span class="brand-mark">RA</span>
        <span>
          <strong>Retail Agent</strong>
          <small>RAG-powered support</small>
        </span>
      </a>
      <div class="runtime-status">
        <span class="status-dot"></span>
        FastAPI · DeepSeek · RAG
      </div>
    </header>

    <main class="main-content">
      <section v-if="!conversation" class="welcome-card">
        <div class="eyebrow">ECOMMERCE AI SUPPORT</div>
        <h1>让每次回答，都有依据。</h1>
        <p class="welcome-copy">
          这是一个用于学习和作品展示的电商客服项目。创建会话后，你可以测试多轮对话、知识检索和来源引用。
        </p>

        <div class="feature-row" aria-label="项目能力">
          <span>多轮会话</span>
          <span>知识检索</span>
          <span>来源追踪</span>
        </div>

        <form class="start-form" @submit.prevent="startConversation">
          <label>
            客户编号
            <input
              v-model="customerNo"
              maxlength="32"
              pattern="[A-Za-z0-9_-]+"
              placeholder="例如 DEMO001"
              required
            />
          </label>
          <label>
            显示名称
            <input
              v-model="displayName"
              maxlength="100"
              placeholder="例如 演示用户"
            />
          </label>
          <button class="primary-button" type="submit" :disabled="!canStart">
            {{ isStarting ? '正在创建…' : '开始新会话' }}
          </button>
        </form>

        <p v-if="errorMessage" class="error-message" role="alert">
          {{ errorMessage }}
        </p>
      </section>

      <section v-else class="workspace">
        <aside class="conversation-card">
          <div>
            <div class="eyebrow">CURRENT SESSION</div>
            <h2>{{ displayName || '访客' }}</h2>
            <p>{{ conversation.conversation_no }}</p>
          </div>

          <dl class="session-details">
            <div>
              <dt>客户编号</dt>
              <dd>{{ conversation.customer_no }}</dd>
            </div>
            <div>
              <dt>会话状态</dt>
              <dd><span class="open-badge">{{ conversation.status }}</span></dd>
            </div>
            <div>
              <dt>消息数量</dt>
              <dd>{{ messages.length }}</dd>
            </div>
          </dl>

          <div class="knowledge-note">
            <strong>公开数据知识库</strong>
            <p>1073 条真实购物澄清案例，来源为 MIT 许可公开数据集。</p>
          </div>

          <button
            class="secondary-button"
            type="button"
            :disabled="isSending"
            @click="resetConversation"
          >
            创建其他会话
          </button>
        </aside>

        <section class="chat-card" aria-label="客服对话">
          <div class="chat-heading">
            <div>
              <div class="eyebrow">AI CUSTOMER SERVICE</div>
              <h2>智能客服</h2>
            </div>
            <span class="online-badge">在线</span>
          </div>

          <div ref="messageList" class="message-list" aria-live="polite">
            <div v-if="messages.length === 0" class="empty-state">
              <div class="empty-icon">✦</div>
              <h3>今天想咨询什么？</h3>
              <p>选择一个问题，或者直接输入你的问题。</p>
              <div class="suggestions">
                <button
                  v-for="suggestion in suggestions"
                  :key="suggestion"
                  type="button"
                  @click="draft = suggestion"
                >
                  {{ suggestion }}
                </button>
              </div>
            </div>

            <article
              v-for="message in messages"
              :key="message.message_id"
              class="message"
              :class="message.role"
            >
              <div class="message-role">
                {{ message.role === 'user' ? '你' : 'Retail Agent' }}
              </div>
              <p
                :class="{
                  'streaming-text': message.streaming && message.content,
                }"
              >
                {{ message.content }}
              </p>
              <div v-if="message.sources?.length" class="source-list">
                <span class="source-label">参考来源</span>
                <a
                  v-for="source in message.sources"
                  :key="source.source_id"
                  class="source-chip"
                  :href="source.source_url"
                  target="_blank"
                  rel="noreferrer"
                >
                  <strong>{{ source.title }}</strong>
                  <small>
                    {{ source.category }} · {{ source.license }} ·
                    {{ source.source_date }} · {{ source.revision }}
                  </small>
                </a>
              </div>
            </article>

            <div
              v-if="isWaitingForFirstChunk"
              class="typing-indicator"
              aria-label="客服正在输入"
            >
              <span></span><span></span><span></span>
            </div>
          </div>

          <div class="composer-wrap">
            <p v-if="errorMessage" class="error-message" role="alert">
              {{ errorMessage }}
            </p>
            <form class="composer" @submit.prevent="sendMessage">
              <textarea
                v-model="draft"
                rows="2"
                maxlength="4000"
                placeholder="输入你的问题…"
                @keydown.enter.exact.prevent="sendMessage"
              ></textarea>
              <button class="send-button" type="submit" :disabled="!canSend">
                <span>{{ isSending ? '接收中' : '发送' }}</span>
                <span aria-hidden="true">↗</span>
              </button>
            </form>
            <p class="disclaimer">AI 可能出错，请以实际业务规则为准。</p>
          </div>
        </section>
      </section>
    </main>
  </div>
</template>
