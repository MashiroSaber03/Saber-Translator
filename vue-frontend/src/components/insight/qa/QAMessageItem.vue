<script setup lang="ts">
import { computed } from 'vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductMessageBubble from '@/components/product/ProductMessageBubble.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import type { QAMessage } from '@/types/insight'

const props = defineProps<{
  message: QAMessage
  renderMarkdown: (content: string) => string
}>()

const emit = defineEmits<{
  (event: 'saveNote', message: QAMessage): void
  (event: 'selectPage', pageNum: number): void
}>()

const citationChips = computed<ProductChipItem[]>(() => {
  return props.message.citations?.map(citation => ({
    id: citation.page,
    label: `第${citation.page}页`,
    ariaLabel: `查看第 ${citation.page} 页`,
    interactive: true,
    tone: 'primary',
  })) ?? []
})

const avatarLabel = computed(() => props.message.role === 'user' ? '用户' : '智能助手')
const messageLabel = computed(() => props.message.role === 'user' ? '用户问题' : '智能回答')

function selectCitation(id: string | number): void {
  emit('selectPage', Number(id))
}
</script>

<template>
  <ProductMessageBubble
    :role="message.role"
    :aria-label="messageLabel"
    :avatar-label="avatarLabel"
    :avatar-image-src="message.role === 'user' ? '/pic/logo.png' : undefined"
    :avatar-icon-name="message.role === 'assistant' ? 'message' : undefined"
  >
    <template v-if="message.role === 'user'">
      {{ message.content }}
    </template>

    <template v-else>
      <div v-if="message.isLoading" class="qa-message-item__loading">
        {{ message.content }}
      </div>
      <template v-else>
        <div v-if="message.mode" class="qa-message-item__mode-badge">{{ message.mode }}</div>
        <div class="qa-message-item__answer-text" v-html="renderMarkdown(message.content)"></div>
      </template>
    </template>

    <template v-if="message.role === 'assistant' && citationChips.length > 0 && !message.isLoading" #footer>
      <ProductChipList
        class="qa-message-item__citations"
        aria-label="引用页码"
        label="引用:"
        label-icon-name="book-open"
        :items="citationChips"
        @select="selectCitation"
      />
    </template>

    <template v-if="message.role === 'assistant' && message.content && !message.isLoading" #actions>
      <UiButton
        variant="secondary"
        size="xs"
        :tone="message.saved ? 'success' : 'neutral'"
        :disabled="message.saved"
        :aria-label="message.saved ? '已保存' : '保存为笔记'"
        @click="emit('saveNote', message)"
      >
        <template v-if="message.saved">
          <UiIcon name="check" />
          <span>已保存</span>
        </template>
        <template v-else>
          <UiIcon name="file-text" />
          <span>保存为笔记</span>
        </template>
      </UiButton>
    </template>
  </ProductMessageBubble>
</template>

<style scoped>
.qa-message-item__answer-text {
  line-height: 1.7;
}

.qa-message-item__mode-badge {
  display: inline-block;
  margin-bottom: 8px;
  padding: 2px 8px;
  border-radius: 4px;
  background: var(--color-surface-muted);
  color: var(--color-text-supporting);
  font-size: 11px;
}

.qa-message-item__citations {
  margin: 0;
}

.qa-message-item__loading {
  display: inline-block;
  color: var(--color-text-supporting);
}

.qa-message-item__loading::after {
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
