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
        tone="neutral"
        role="note"
        icon-name="message"
        title="智能问答"
      >
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
