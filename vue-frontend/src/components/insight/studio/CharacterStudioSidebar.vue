<template>
  <div class="sidebar-shell">
    <div class="sidebar-toolbar">
      <div class="toolbar-copy">
        <div class="kicker">导航与资源</div>
        <h2>当前书籍角色工坊</h2>
        <p>从分析候选锁定角色名，再用 AI 补全整卡；也可以直接空白新建或导入外部角色卡。候选仅预填角色名，不再直接抽离压缩后的分析字段。</p>
      </div>

      <div class="toolbar-actions">
        <UiInput
          :value="search"
          class="search-input"
          placeholder="搜索角色 / 标签 / 来源"
          type="text"
          :disabled="workspaceLoading"
          @input="$emit('update:search', ($event.target as HTMLInputElement).value)"
        />
        <div class="action-row">
          <UiButton variant="toolbar" class="action-primary" :disabled="creatingManual || importingFile" @click="$emit('create-manual')">
            {{ creatingManual ? '新建中...' : '空白新建' }}
          </UiButton>
          <UiButton variant="toolbar" class="action-ghost" :disabled="creatingManual || importingFile" @click="pickImport">
            {{ importingFile ? '导入中...' : '导入' }}
          </UiButton>
        </div>
      </div>

      <UiFileInput ref="fileInput" hidden accept=".json,.png,.jpg,.jpeg,.webp,.gif,.bmp" @change="handleFileSelect" />
    </div>

    <div class="sidebar-content">
      <DocumentListPane
        :documents="documents"
        :current-document-id="currentDocumentId"
        :opening-document-id="openingDocumentId"
        @open="$emit('open-document', $event)"
      />

      <CandidateListPane
        :candidates="candidates"
        :has-timeline="hasTimeline"
        :creating-candidate-name="creatingCandidateName"
        @create="$emit('create-from-candidate', $event)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { ref } from 'vue'
import type { CharacterStudioCandidate, CharacterStudioSummary } from '@/types/characterStudio'
import DocumentListPane from './DocumentListPane.vue'
import CandidateListPane from './CandidateListPane.vue'

defineProps<{
  documents: CharacterStudioSummary[]
  candidates: CharacterStudioCandidate[]
  search: string
  currentDocumentId: string
  hasTimeline: boolean
  workspaceLoading: boolean
  creatingManual: boolean
  importingFile: boolean
  openingDocumentId: string
  creatingCandidateName: string
}>()

const emit = defineEmits<{
  (e: 'update:search', value: string): void
  (e: 'open-document', docId: string): void
  (e: 'create-manual'): void
  (e: 'create-from-candidate', candidateName: string): void
  (e: 'import-file', file: File): void
}>()

const fileInput = ref<HTMLInputElement | null>(null)

function pickImport() {
  fileInput.value?.click()
}

function handleFileSelect(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return
  emit('import-file', file)
  target.value = ''
}
</script>

<style scoped>
.sidebar-shell {
  --character-studio-sidebar-toolbar-background-start: rgba(79, 136, 240, .12);
  --character-studio-sidebar-toolbar-background-end: rgba(246, 249, 254, .88);
  --character-studio-sidebar-primary-action-shadow: rgba(37, 99, 199, .2);
  --character-studio-sidebar-shell-background: rgba(252, 253, 255, .88);
  --character-studio-sidebar-search-background: rgba(255, 255, 255, .92);
  --character-studio-sidebar-primary-action-background-start: #2563c7;
  --character-studio-sidebar-primary-action-background-end: #4d86ee;
  --character-studio-sidebar-kicker-text: #6f84a2;
  --character-studio-sidebar-title-text: #102741;
  --ui-input-padding: 12px 14px;
  --ui-input-border: 1px solid var(--studio-border-strong);
  --ui-input-radius: 14px;
  --ui-input-background: var(--character-studio-sidebar-search-background);
  --ui-input-color: var(--studio-text-strong);
  --ui-input-font-size: 13px;
  --ui-input-focus-border: var(--color-border-brand);
  --ui-input-focus-shadow: var(--color-focus-brand-soft);

  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  width: 100%;
  border-radius: 26px;
  overflow: hidden;
  background: var(--character-studio-sidebar-shell-background);
  border: 1px solid var(--studio-border-default);
  box-shadow: 0 24px 40px var(--studio-shadow-floating);
}

.sidebar-toolbar {
  flex-shrink: 0;
  padding: 18px 18px 16px;
  border-bottom: 1px solid var(--studio-border-default);
  background:
    linear-gradient(180deg, var(--character-studio-sidebar-toolbar-background-start), var(--character-studio-sidebar-toolbar-background-end));
}

.kicker {
  font-size: 11px;
  letter-spacing: 0;
  text-transform: uppercase;
  color: var(--character-studio-sidebar-kicker-text);
}

.toolbar-copy h2 {
  margin: 8px 0 0;
  font-size: 22px;
  line-height: 1.24;
  color: var(--character-studio-sidebar-title-text);
}

.toolbar-copy p {
  margin: 10px 0 0;
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.7;
}

.toolbar-actions {
  margin-top: 16px;
}

.search-input {
  width: 100%;
}

.action-row {
  display: flex;
  gap: 10px;
  margin-top: 12px;
}

.action-primary,
.action-ghost {
  border: none;
  border-radius: 14px;
  cursor: pointer;
}

.action-primary {
  flex: 1;
  padding: 11px 16px;
  background: linear-gradient(135deg, var(--character-studio-sidebar-primary-action-background-start), var(--character-studio-sidebar-primary-action-background-end));
  color: var(--color-text-inverse);
  box-shadow: 0 12px 24px var(--character-studio-sidebar-primary-action-shadow);
}

.action-ghost {
  padding: 11px 14px;
  background: var(--studio-surface-muted);
  color: var(--studio-text-default);
}

.action-primary:disabled,
.action-ghost:disabled {
  opacity: 0.68;
  cursor: not-allowed;
  box-shadow: none;
}

.sidebar-content {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding: 18px;
  overflow: auto;
}
</style>
