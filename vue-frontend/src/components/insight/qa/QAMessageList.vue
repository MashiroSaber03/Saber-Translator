<script setup lang="ts">
import { ref } from 'vue'
import type { QAMessage } from '@/stores/insightStore'
import QAMessageItem from './QAMessageItem.vue'

defineProps<{
  messages: QAMessage[]
  renderMarkdown: (content: string) => string
}>()

defineEmits<{
  (event: 'saveNote', message: QAMessage): void
  (event: 'selectPage', pageNum: number): void
}>()

const messagesEl = ref<HTMLElement | null>(null)

function scrollToBottom(): void {
  if (messagesEl.value) {
    messagesEl.value.scrollTop = messagesEl.value.scrollHeight
  }
}

defineExpose({ scrollToBottom })
</script>

<template>
  <div ref="messagesEl" class="chat-messages">
    <div v-if="messages.length === 0" class="welcome-message">
      <div class="welcome-icon">💬</div>
      <h3>智能问答</h3>
      <p>针对已分析的漫画内容提问，获取精准回答</p>
    </div>

    <QAMessageItem
      v-for="message in messages"
      :key="message.id"
      :message="message"
      :render-markdown="renderMarkdown"
      @save-note="$emit('saveNote', $event)"
      @select-page="$emit('selectPage', $event)"
    />
  </div>
</template>

<style scoped>
.chat-messages {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
}

.welcome-message {
  padding: 40px 20px;
  text-align: center;
}

.welcome-icon {
  margin-bottom: 16px;
  font-size: 48px;
}

.welcome-message h3 {
  margin-bottom: 8px;
}

.welcome-message p {
  margin-bottom: 20px;
  color: var(--insight-text-secondary);
}
</style>
