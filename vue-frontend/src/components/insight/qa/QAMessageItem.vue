<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import type { QAMessage } from '@/stores/insightStore'

defineProps<{
  message: QAMessage
  renderMarkdown: (content: string) => string
}>()

defineEmits<{
  (event: 'saveNote', message: QAMessage): void
  (event: 'selectPage', pageNum: number): void
}>()
</script>

<template>
  <div class="chat-message" :class="message.role">
    <div class="message-avatar">
      <template v-if="message.role === 'user'">
        <img src="/pic/logo.png" alt="用户" class="avatar-img">
      </template>
      <template v-else>
        🤖
      </template>
    </div>

    <div v-if="message.role === 'user'" class="message-content">
      {{ message.content }}
    </div>

    <div v-else class="message-content markdown-content">
      <div v-if="message.isLoading" class="loading-dots">
        {{ message.content }}
      </div>
      <template v-else>
        <div v-if="message.mode" class="answer-mode-badge">{{ message.mode }}</div>
        <div class="answer-text" v-html="renderMarkdown(message.content)"></div>

        <div v-if="message.citations && message.citations.length > 0" class="message-citations">
          <span>📖 引用: </span>
          <UiButton
            v-for="citation in message.citations"
            :key="citation.page"
            variant="toolbar"
            type="button"
            class="citation-item"
            :aria-label="`查看第 ${citation.page} 页`"
            @click="$emit('selectPage', citation.page)"
          >
            第{{ citation.page }}页
          </UiButton>
        </div>

        <UiButton
          v-if="message.content && !message.isLoading"
          variant="toolbar"
          class="message-save-btn"
          :class="{ saved: message.saved }"
          :disabled="message.saved"
          @click="$emit('saveNote', message)"
        >
          {{ message.saved ? '✅ 已保存' : '📝 保存为笔记' }}
        </UiButton>
      </template>
    </div>
  </div>
</template>

<style scoped>
.chat-message {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  animation: slideIn 0.3s ease;
}

.chat-message.user {
  flex-direction: row-reverse;
}

.message-avatar {
  display: flex;
  flex-shrink: 0;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  font-size: 18px;
}

.chat-message.user .message-avatar {
  overflow: hidden;
  background: transparent;
}

.avatar-img {
  display: block;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  object-fit: cover;
}

.chat-message.assistant .message-avatar {
  background: var(--insight-surface-tertiary);
}

.message-content {
  max-width: 70%;
  padding: 12px 16px;
  border-radius: 12px;
  line-height: 1.6;
}

.chat-message.user .message-content {
  border-bottom-right-radius: 4px;
  background: var(--insight-action-primary);
  color: var(--color-text-inverse);
}

.chat-message.assistant .message-content {
  border: 1px solid var(--color-border-muted);
  border-bottom-left-radius: 4px;
  background: var(--insight-surface-secondary);
}

.markdown-content {
  max-width: 70%;
}

.answer-text {
  line-height: 1.7;
}

.answer-mode-badge {
  display: inline-block;
  margin-bottom: 8px;
  padding: 2px 8px;
  border-radius: 4px;
  background: var(--insight-surface-secondary);
  color: var(--insight-text-secondary);
  font-size: 11px;
}

.message-citations {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--color-border-muted);
  color: var(--insight-text-secondary);
  font-size: 12px;
}

.citation-item {
  display: inline-block;
  margin: 2px 4px;
  padding: 2px 8px;
  border-radius: 4px;
  background: var(--insight-surface-tertiary);
  cursor: pointer;
}

.citation-item:hover {
  background: var(--insight-action-primary);
  color: var(--color-text-inverse);
}

.message-save-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-top: 12px;
  padding: 6px 12px;
  border: 1px solid var(--color-border-muted);
  border-radius: 6px;
  background: var(--insight-surface-tertiary);
  color: var(--insight-text-secondary);
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s;
}

.message-save-btn:hover {
  border-color: var(--insight-action-primary);
  background: var(--insight-action-primary);
  color: var(--color-text-inverse);
}

.message-save-btn.saved {
  border-color: var(--insight-status-success);
  background: var(--insight-status-success);
  color: var(--color-text-inverse);
  cursor: default;
}

.loading-dots {
  display: inline-block;
  color: var(--insight-text-secondary);
}

.loading-dots::after {
  animation: dots 1.5s steps(4, end) infinite;
  content: '';
}

@keyframes dots {
  0%, 20% { content: ''; }
  40% { content: '.'; }
  60% { content: '..'; }
  80%, 100% { content: '...'; }
}
</style>
