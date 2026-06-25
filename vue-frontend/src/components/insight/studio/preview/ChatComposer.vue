<script setup lang="ts">
import { onUnmounted, ref } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import { attachmentTypeLabel, type PendingAttachmentCard } from '../characterStudioPreviewHelpers'

defineProps<{
  chatStreaming: boolean
}>()

const emit = defineEmits<{
  (event: 'send-chat', value: { content: string; attachments: File[] }): void
}>()

const chatInput = ref('')
const pendingFiles = ref<PendingAttachmentCard[]>([])
const attachmentInput = ref<HTMLInputElement | null>(null)

function pickAttachments() {
  attachmentInput.value?.click()
}

function handleAttachmentChange(event: Event) {
  const target = event.target as HTMLInputElement
  const files = Array.from(target.files || [])
  pendingFiles.value = [
    ...pendingFiles.value,
    ...files.map(file => ({
      id: `pending-${Date.now()}-${Math.random().toString(16).slice(2, 6)}`,
      file,
      previewUrl: URL.createObjectURL(file),
    })),
  ]
  target.value = ''
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
  <div class="composer-card">
    <div v-if="pendingFiles.length > 0" class="pending-files">
      <div
        v-for="(file, index) in pendingFiles"
        :key="file.id"
        class="pending-image-card"
      >
        <div class="pending-image-thumb">
          <img :src="file.previewUrl" :alt="file.file.name">
        </div>
        <div class="pending-image-copy">
          <strong>{{ file.file.name }}</strong>
          <span>{{ attachmentTypeLabel(file.file.type || 'application/octet-stream') }}</span>
        </div>
        <UiButton
          variant="toolbar"
          type="button"
          class="pending-remove"
          :aria-label="`移除附件：${file.file.name}`"
          @click="removePendingFile(index)"
        >
          ×
        </UiButton>
      </div>
    </div>
    <div class="composer-main">
      <UiTextarea
        v-model="chatInput"
        class="chat-composer-input"
        rows="1"
        placeholder="输入消息，或添加图片后让角色结合画面继续聊天。"
      />
      <div class="composer-actions compact-actions">
        <UiButton
          variant="toolbar"
          data-testid="chat-upload-trigger"
          class="action-ghost icon-btn"
          type="button"
          title="添加图片"
          aria-label="添加图片"
          :disabled="chatStreaming"
          @click="pickAttachments"
        >
          +
        </UiButton>
        <UiButton
          variant="toolbar"
          data-testid="chat-send-trigger"
          class="action-primary icon-btn"
          type="button"
          :title="chatStreaming ? '回复生成中...' : '发送消息'"
          :aria-label="chatStreaming ? '回复生成中...' : '发送消息'"
          :disabled="chatStreaming || (!chatInput.trim() && pendingFiles.length === 0)"
          @click="sendChat"
        >
          {{ chatStreaming ? '…' : '↗' }}
        </UiButton>
      </div>
    </div>
    <UiFileInput ref="attachmentInput" hidden accept="image/*" multiple @change="handleAttachmentChange" />
  </div>
</template>

<style scoped>
.composer-card {
  --ui-textarea-border: 1px solid var(--studio-border-strong);
  --ui-textarea-background: var(--studio-surface-soft);
  --ui-textarea-radius: 14px;
  --ui-textarea-padding: 10px 12px;
  --ui-textarea-color: var(--studio-text-strong);
  --ui-textarea-font-size: 13px;

  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 2px;
  padding: 10px 12px;
  border: 1px solid var(--studio-border-default);
  border-radius: 20px;
  background: var(--character-studio-preview-composer-background);
}

.composer-main {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: stretch;
  gap: 10px;
}

.chat-composer-input {
  min-height: 64px;
  resize: vertical;
}

.composer-actions {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  justify-content: space-between;
}

.compact-actions {
  flex-direction: column;
  align-items: stretch;
  justify-content: flex-end;
  gap: 6px;
}

.icon-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  min-width: 44px;
  height: 44px;
  padding: 0;
  font-size: 22px;
  line-height: 1;
}

.pending-files {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  flex-wrap: wrap;
  gap: 10px;
}

.pending-image-card {
  position: relative;
  display: grid;
  grid-template-columns: 48px minmax(0, 1fr);
  gap: 8px;
  align-items: center;
  padding: 8px 28px 8px 8px;
  border: 1px solid var(--studio-border-default);
  border-radius: 14px;
  background: var(--character-studio-preview-pending-attachment-background);
  text-align: left;
}

.pending-image-thumb {
  width: 48px;
  height: 48px;
  overflow: hidden;
  border-radius: 10px;
  background: var(--studio-surface-soft);
}

.pending-image-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.pending-image-copy {
  min-width: 0;
}

.pending-image-copy strong {
  display: block;
  overflow: hidden;
  color: var(--studio-text-strong);
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.pending-image-copy span {
  color: var(--studio-text-muted);
  font-size: 11px;
}

.pending-remove {
  position: absolute;
  top: 6px;
  right: 8px;
  padding: 2px 4px;
  color: var(--studio-text-muted);
  cursor: pointer;
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
  background: linear-gradient(135deg, var(--character-studio-preview-primary-action-background-start), var(--character-studio-preview-primary-action-background-end));
  box-shadow: 0 12px 24px var(--character-studio-preview-primary-action-shadow);
  color: var(--color-text-inverse);
}

.action-ghost:disabled,
.action-primary:disabled {
  cursor: not-allowed;
  box-shadow: none;
  opacity: 0.68;
}

@media (--breakpoint-preview-down) {
  .composer-main {
    grid-template-columns: 1fr;
  }

  .compact-actions {
    flex-direction: row;
  }
}
</style>
