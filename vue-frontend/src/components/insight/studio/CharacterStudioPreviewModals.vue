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
