<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import ProductScrollStack from '@/components/product/ProductScrollStack.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import type { QAMessage } from '@/types/insight'
import QAMessageItem from './QAMessageItem.vue'

const props = withDefaults(defineProps<{
  messages: QAMessage[]
  renderMarkdown: (content: string) => string
  scrollRequestId?: number
}>(), {
  scrollRequestId: 0,
})

defineEmits<{
  (event: 'saveNote', message: QAMessage): void
  (event: 'selectPage', pageNum: number): void
}>()

const messagesEl = ref<InstanceType<typeof ProductScrollStack> | null>(null)

function scrollToBottom(): void {
  messagesEl.value?.scrollToBottom()
}

watch(
  () => props.scrollRequestId,
  async (requestId, previousRequestId) => {
    if (requestId === previousRequestId) return
    await nextTick()
    scrollToBottom()
  },
)
</script>

<template>
  <ProductScrollStack
    ref="messagesEl"
    class="qa-message-list"
    role="log"
    aria-label="问答消息"
    aria-live="polite"
    gap="md"
    :empty="props.messages.length === 0"
  >
    <template #empty>
      <ProductStatusBanner
        class="qa-message-list__welcome"
        tone="neutral"
        role="note"
        icon-name="message"
        title="智能问答"
      >
        <template #icon>💬</template>
        针对已分析的漫画内容提问，获取精准回答
      </ProductStatusBanner>
    </template>

    <QAMessageItem
      v-for="message in props.messages"
      :key="message.id"
      :message="message"
      :render-markdown="props.renderMarkdown"
      @save-note="$emit('saveNote', $event)"
      @select-page="$emit('selectPage', $event)"
    />
  </ProductScrollStack>
</template>

<style scoped>
.qa-message-list {
  --product-scroll-stack-empty-justify-content: flex-start;
}

.qa-message-list__welcome {
  --product-status-banner-flex-direction: column;
  --product-status-banner-align-items: center;
  --product-status-banner-justify-content: center;
  --product-status-banner-gap: 0;
  --product-status-banner-width: 100%;
  --product-status-banner-padding: 58px 20px 40px;
  --product-status-banner-border: 0;
  --product-status-banner-background: transparent;
  --product-status-banner-text-align: center;
  --product-status-banner-icon-margin: 0 0 16px;
  --product-status-banner-icon-font-size: 48px;
  --product-status-banner-icon-transform: none;
  --product-status-banner-accent: color-mix(in srgb, var(--color-action-brand) 40%, transparent);
  --product-status-banner-title-margin-bottom: 8px;
  --product-status-banner-title-font-size: 1.17rem;
  --product-status-banner-body-color: var(--insight-text-secondary);
}
</style>
