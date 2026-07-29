<script setup lang="ts">
import { onUnmounted, ref } from 'vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import { attachmentTypeLabel, type PendingAttachmentCard } from '../characterStudioPreviewHelpers'

defineProps<{
  chatAbortable?: boolean
  chatStreaming: boolean
}>()

const emit = defineEmits<{
  (event: 'abort-chat'): void
  (event: 'send-chat', value: { content: string; attachments: File[] }): void
}>()

const chatInput = ref('')
const pendingFiles = ref<PendingAttachmentCard[]>([])
const attachmentInput = ref<InstanceType<typeof UiFileInput> | null>(null)

function pickAttachments() {
  attachmentInput.value?.click()
}

function handleAttachmentChange(files: File[]) {
  if (files.length === 0) return
  pendingFiles.value = [
    ...pendingFiles.value,
    ...files.map(file => ({
      id: `pending-${Date.now()}-${Math.random().toString(16).slice(2, 6)}`,
      file,
      previewUrl: URL.createObjectURL(file),
    })),
  ]
  attachmentInput.value?.clear()
}

function removePendingFile(index: number) {
  const removed = pendingFiles.value[index]
  if (removed) URL.revokeObjectURL(removed.previewUrl)
  pendingFiles.value.splice(index, 1)
}

function sendChat() {
  const content = chatInput.value.trim()
  if (!content && pendingFiles.value.length === 0) return
  emit('send-chat', { content, attachments: pendingFiles.value.map(item => item.file) })
  chatInput.value = ''
  pendingFiles.value.forEach(item => URL.revokeObjectURL(item.previewUrl))
  pendingFiles.value = []
}

onUnmounted(() => {
  pendingFiles.value.forEach(item => URL.revokeObjectURL(item.previewUrl))
})
</script>

<template>
  <div class="studio-chat-composer">
    <div v-if="pendingFiles.length > 0" class="studio-chat-composer__pending-files">
      <div
        v-for="(file, index) in pendingFiles"
        :key="file.id"
        class="studio-chat-composer__pending-card"
      >
        <div class="studio-chat-composer__pending-thumb">
          <img class="studio-chat-composer__pending-image" :src="file.previewUrl" :alt="file.file.name">
        </div>
        <div class="studio-chat-composer__pending-copy">
          <strong class="studio-chat-composer__pending-name">{{ file.file.name }}</strong>
          <span class="studio-chat-composer__pending-type">{{ attachmentTypeLabel(file.file.type || 'application/octet-stream') }}</span>
        </div>
        <UiIconButton
          variant="plain"
          type="button"
          size="xs"
          class="studio-chat-composer__pending-remove"
          :label="`移除附件：${file.file.name}`"
          @click="removePendingFile(index)"
        >
          <UiIcon name="x" size="14" />
        </UiIconButton>
      </div>
    </div>
    <div class="studio-chat-composer__main">
      <UiTextarea
        v-model="chatInput"
        class="studio-chat-composer__input"
        variant="studio"
        rows="1"
        aria-label="聊天消息内容"
        placeholder="输入消息，或添加图片后让角色结合画面继续聊天。"
      />
      <div class="studio-chat-composer__actions">
        <UiIconButton
          variant="soft"
          size="lg"
          data-testid="chat-upload-trigger"
          type="button"
          label="添加图片"
          :disabled="chatStreaming"
          @click="pickAttachments"
        >
          <UiIcon name="plus" size="18" />
        </UiIconButton>
        <UiIconButton
          v-if="chatStreaming"
          variant="danger"
          size="lg"
          data-testid="chat-abort-trigger"
          type="button"
          :label="chatAbortable ? '中止本次生成' : '正在创建后端操作'"
          :disabled="!chatAbortable"
          @click="$emit('abort-chat')"
        >
          <UiIcon name="square" size="18" />
        </UiIconButton>
        <UiIconButton
          v-else
          variant="primary"
          size="lg"
          data-testid="chat-send-trigger"
          type="button"
          label="发送消息"
          :disabled="!chatInput.trim() && pendingFiles.length === 0"
          @click="sendChat"
        >
          <UiIcon name="send" size="18" />
        </UiIconButton>
      </div>
    </div>
    <UiFileInput ref="attachmentInput" hidden accept="image/*" multiple @files-change="handleAttachmentChange" />
  </div>
</template>

<style scoped>
.studio-chat-composer {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 2px;
  padding: 12px;
  border: 1px solid var(--studio-border-default);
  border-radius: 20px;
  background: color-mix(in srgb, var(--studio-surface-soft) 92%, var(--color-surface-card));
  box-shadow: inset 0 1px 0 color-mix(in srgb, var(--color-surface-card) 64%, transparent);
}

.studio-chat-composer__main {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: stretch;
  gap: 10px;
}

.studio-chat-composer__input {
  min-height: 72px;
  resize: vertical;
}

.studio-chat-composer__actions {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  justify-content: flex-end;
  gap: 6px;
}

.studio-chat-composer__pending-files {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 180px), 1fr));
  gap: 10px;
}

.studio-chat-composer__pending-card {
  position: relative;
  display: grid;
  grid-template-columns: 48px minmax(0, 1fr);
  gap: 8px;
  align-items: center;
  padding: 8px 28px 8px 8px;
  border: 1px solid var(--studio-border-default);
  border-radius: 14px;
  background: color-mix(in srgb, var(--color-surface-card) 88%, transparent);
  text-align: left;
}

.studio-chat-composer__pending-thumb {
  width: 48px;
  height: 48px;
  overflow: hidden;
  border-radius: 10px;
  background: var(--studio-surface-soft);
}

.studio-chat-composer__pending-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.studio-chat-composer__pending-copy {
  min-width: 0;
}

.studio-chat-composer__pending-name {
  display: block;
  overflow: hidden;
  color: var(--studio-text-strong);
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.studio-chat-composer__pending-type {
  color: var(--studio-text-muted);
  font-size: 11px;
}

.studio-chat-composer__pending-remove {
  position: absolute;
  top: 6px;
  right: 8px;
}

@media (--breakpoint-preview-down) {
  .studio-chat-composer__main {
    grid-template-columns: 1fr;
  }

  .studio-chat-composer__actions {
    flex-direction: row;
  }
}
</style>
