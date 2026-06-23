<script setup lang="ts">
import { computed } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'

const props = defineProps<{
  isStreaming: boolean
  question: string
}>()

const emit = defineEmits<{
  (event: 'submit'): void
  (event: 'update:question', value: string): void
}>()

const questionModel = computed({
  get: () => props.question,
  set: value => emit('update:question', value),
})

function handleKeydown(event: KeyboardEvent): void {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    emit('submit')
  }
}
</script>

<template>
  <div class="chat-input-wrapper">
    <UiTextarea
      v-model="questionModel"
      placeholder="输入你的问题..."
      rows="1"
      :disabled="isStreaming"
      @keydown="handleKeydown"
    />
    <UiButton
      variant="toolbar"
      class="send-btn"
      :disabled="isStreaming || !question.trim()"
      @click="$emit('submit')"
    >
      <span>发送</span>
    </UiButton>
  </div>
</template>

<style scoped>
.chat-input-wrapper {
  display: flex;
  align-items: flex-end;
  gap: 12px;
}

.send-btn {
  padding: 12px 24px;
  border: none;
  border-radius: 12px;
  background: var(--insight-action-primary);
  color: white;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.2s;
}

.send-btn:hover {
  background: var(--insight-action-primary-strong);
}

.send-btn:disabled {
  background: var(--insight-text-muted);
  cursor: not-allowed;
}
</style>
