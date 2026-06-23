<script setup lang="ts">
import { computed } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'

interface StudioGreetingOption {
  greeting_id: string
  label: string
  content: string
}

const props = defineProps<{
  greetingOpen: boolean
  promptOpen: boolean
  imageOpen: boolean
  displayGreetings: StudioGreetingOption[]
  selectedGreetingId: string
  chatMutating: boolean
  chatStreaming: boolean
  chatPromptLoading: boolean
  promptPreview: string
  promptPreviewError: string
  imageTitle: string
  imageSrc: string
}>()

const emit = defineEmits<{
  (e: 'update:greetingOpen', value: boolean): void
  (e: 'update:promptOpen', value: boolean): void
  (e: 'update:imageOpen', value: boolean): void
  (e: 'update:selectedGreetingId', value: string): void
  (e: 'confirm-greeting-selection'): void
  (e: 'copy-prompt-preview'): void
}>()

const greetingModel = computed({
  get: () => props.greetingOpen,
  set: value => emit('update:greetingOpen', value),
})

const promptModel = computed({
  get: () => props.promptOpen,
  set: value => emit('update:promptOpen', value),
})

const imageModel = computed({
  get: () => props.imageOpen,
  set: value => emit('update:imageOpen', value),
})
</script>

<template>
  <BaseModal
    v-model="greetingModel"
    title="重选开场白"
    size="large"
    custom-class="studio-chat-modal"
  >
    <div class="modal-copy">
      <p>选择一条开场白后，将归档当前会话，并以该开场白重新开启一轮新对话。</p>
    </div>
    <div v-if="displayGreetings.length === 0" class="modal-empty">当前还没有可用开场白。</div>
    <div v-else class="greeting-grid">
      <UiButton
        variant="toolbar"
        v-for="item in displayGreetings"
        :key="item.greeting_id"
        type="button"
        class="greeting-card"
        :class="{ active: selectedGreetingId === item.greeting_id }"
        @click="$emit('update:selectedGreetingId', item.greeting_id)"
      >
        <div class="greeting-card-head">
          <span class="greeting-badge">{{ item.label }}</span>
          <span v-if="selectedGreetingId === item.greeting_id" class="greeting-check">✓</span>
        </div>
        <p>{{ item.content }}</p>
      </UiButton>
    </div>
    <template #footer>
      <UiButton variant="toolbar" class="action-ghost" @click="greetingModel = false">取消</UiButton>
      <UiButton
        variant="toolbar"
        class="action-primary"
        :disabled="!selectedGreetingId || chatMutating || chatStreaming"
        @click="$emit('confirm-greeting-selection')"
      >
        确认并重新开场
      </UiButton>
    </template>
  </BaseModal>

  <BaseModal
    v-model="promptModel"
    title="本轮提示词预览"
    size="large"
    custom-class="studio-chat-modal"
  >
    <div v-if="chatPromptLoading" class="modal-loading">提示词加载中...</div>
    <div v-else-if="promptPreviewError" class="modal-empty">{{ promptPreviewError }}</div>
    <div v-else-if="promptPreview.trim()" class="prompt-preview-body">
      <div class="prompt-tools">
        <UiButton variant="toolbar" class="action-ghost" @click="$emit('copy-prompt-preview')" size="sm">
          复制内容
        </UiButton>
      </div>
      <pre>{{ promptPreview }}</pre>
    </div>
    <div v-else class="modal-empty" data-testid="prompt-preview-empty">
      请先发送至少一条消息后再查看本轮提示词。
    </div>
  </BaseModal>

  <BaseModal
    v-model="imageModel"
    :title="imageTitle"
    size="large"
    custom-class="studio-chat-modal studio-image-modal"
  >
    <div v-if="imageSrc" class="image-preview-body">
      <img :src="imageSrc" :alt="imageTitle">
    </div>
  </BaseModal>
</template>

<style scoped>
.modal-copy p,
.modal-empty,
.modal-loading {
  color: var(--studio-text-default);
  font-size: 13px;
  line-height: 1.7;
}

.greeting-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px;
  margin-top: 14px;
}

.greeting-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 14px;
  border: 1px solid var(--studio-border-default);
  border-radius: 16px;
  background: var(--character-studio-preview-workspace-surface-tint);
  text-align: left;
}

.greeting-card.active {
  border-color: var(--character-studio-preview-details-border-default);
  box-shadow: 0 0 0 2px var(--character-studio-preview-details-shadow-default);
}

.greeting-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.greeting-badge {
  color: var(--character-studio-preview-details-text-primary);
  font-weight: 700;
}

.greeting-check {
  width: 22px;
  height: 22px;
  border-radius: 999px;
  background: var(--character-studio-preview-details-surface-active);
  color: var(--color-text-inverse);
  line-height: 22px;
  text-align: center;
}

.greeting-card p {
  margin: 0;
  color: var(--studio-text-default);
  font-size: 13px;
  line-height: 1.7;
  white-space: pre-wrap;
}

.prompt-preview-body pre {
  max-height: 420px;
  margin: 10px 0 0;
  padding: 14px;
  overflow: auto;
  border: 1px solid var(--studio-border-default);
  border-radius: 16px;
  background: var(--studio-surface-soft);
  color: var(--studio-text-strong);
  font-size: 12px;
  line-height: 1.7;
  white-space: pre-wrap;
}

.prompt-tools {
  display: flex;
  justify-content: flex-end;
}

.image-preview-body {
  display: flex;
  justify-content: center;
}

.image-preview-body img {
  max-width: 100%;
  max-height: 70vh;
  border-radius: 16px;
  object-fit: contain;
}

.action-ghost,
.action-primary {
  border: none;
  border-radius: 14px;
  font-size: 13px;
  cursor: pointer;
}

.action-ghost {
  padding: 10px 14px;
  background: var(--studio-surface-muted);
  color: var(--studio-text-default);
}

.action-primary {
  padding: 11px 16px;
  background: linear-gradient(135deg, var(--character-studio-preview-details-surface-active), var(--character-studio-preview-details-surface-selected));
  box-shadow: 0 12px 24px var(--character-studio-preview-details-shadow-raised);
  color: var(--color-text-inverse);
}

.action-ghost:disabled,
.action-primary:disabled {
  cursor: not-allowed;
  box-shadow: none;
  opacity: 0.68;
}
</style>
