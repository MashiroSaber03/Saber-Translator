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
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  width: 100%;
  border-radius: 26px;
  overflow: hidden;
  background: var(--character-studio-sidebar-surface-base);
  border: 1px solid var(--color-border-studio);
  box-shadow: 0 24px 40px var(--shadow-studio-floating);
}

.sidebar-toolbar {
  flex-shrink: 0;
  padding: 18px 18px 16px;
  border-bottom: 1px solid var(--color-border-studio);
  background:
    linear-gradient(180deg, var(--character-studio-sidebar-accent-primary), var(--character-studio-sidebar-accent-secondary));
}

.kicker {
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--character-studio-sidebar-text-primary);
}

.toolbar-copy h2 {
  margin: 8px 0 0;
  font-size: 22px;
  line-height: 1.24;
  color: var(--character-studio-sidebar-text-secondary);
}

.toolbar-copy p {
  margin: 10px 0 0;
  color: var(--color-text-studio-muted);
  font-size: 13px;
  line-height: 1.7;
}

.toolbar-actions {
  margin-top: 16px;
}

.search-input {
  width: 100%;
  border: 1px solid var(--color-border-studio-strong);
  background: var(--character-studio-sidebar-surface-raised);
  border-radius: 14px;
  padding: 12px 14px;
  color: var(--color-text-studio-strong);
  font-size: 13px;
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
  background: linear-gradient(135deg, var(--character-studio-sidebar-surface-muted), var(--character-studio-sidebar-surface-subtle));
  color: var(--color-text-inverse);
  box-shadow: 0 12px 24px var(--character-studio-sidebar-shadow-default);
}

.action-ghost {
  padding: 11px 14px;
  background: var(--color-surface-studio-muted);
  color: var(--color-text-studio);
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
