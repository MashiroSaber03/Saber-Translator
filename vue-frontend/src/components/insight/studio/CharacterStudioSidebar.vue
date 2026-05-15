<template>
  <aside class="sidebar-shell">
    <div class="sidebar-toolbar">
      <div class="toolbar-copy">
        <div class="kicker">导航与资源</div>
        <h2>当前书籍角色工坊</h2>
        <p>从分析候选创建角色，或直接空白新建、导入外部角色卡。</p>
      </div>

      <div class="toolbar-actions">
        <input
          :value="search"
          class="search-input"
          placeholder="搜索角色 / 标签 / 来源"
          type="text"
          @input="$emit('update:search', ($event.target as HTMLInputElement).value)"
        >
        <div class="action-row">
          <button class="primary-btn" @click="$emit('create-manual')">空白新建</button>
          <button class="ghost-btn" @click="pickImport">导入</button>
        </div>
      </div>

      <input ref="fileInput" hidden type="file" accept=".json,.png,.jpg,.jpeg,.webp,.gif,.bmp" @change="handleFileSelect">
    </div>

    <div class="sidebar-content">
      <DocumentListPane
        :documents="documents"
        :current-document-id="currentDocumentId"
        @open="$emit('open-document', $event)"
      />

      <CandidateListPane
        :candidates="candidates"
        :has-timeline="hasTimeline"
        @create="$emit('create-from-candidate', $event)"
      />
    </div>
  </aside>
</template>

<script setup lang="ts">
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
  min-height: 0;
  width: 100%;
  border-radius: 26px;
  overflow: hidden;
  background: rgba(252, 253, 255, 0.88);
  border: 1px solid rgba(28, 55, 94, 0.08);
  box-shadow: 0 24px 40px rgba(20, 46, 82, 0.08);
}

.sidebar-toolbar {
  padding: 18px 18px 16px;
  border-bottom: 1px solid rgba(28, 55, 94, 0.08);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(245, 249, 255, 0.82));
}

.kicker {
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #6f84a2;
}

.toolbar-copy h2 {
  margin: 8px 0 0;
  font-size: 22px;
  line-height: 1.24;
  color: #102741;
}

.toolbar-copy p {
  margin: 10px 0 0;
  color: #607794;
  font-size: 13px;
  line-height: 1.7;
}

.toolbar-actions {
  margin-top: 16px;
}

.search-input {
  width: 100%;
  border: 1px solid rgba(28, 55, 94, 0.12);
  background: rgba(255, 255, 255, 0.92);
  border-radius: 14px;
  padding: 12px 14px;
  color: #183351;
  font-size: 13px;
}

.action-row {
  display: flex;
  gap: 10px;
  margin-top: 12px;
}

.primary-btn,
.ghost-btn {
  border: none;
  border-radius: 14px;
  cursor: pointer;
}

.primary-btn {
  flex: 1;
  padding: 11px 16px;
  background: linear-gradient(135deg, #2563c7, #4d86ee);
  color: #fff;
  box-shadow: 0 12px 24px rgba(37, 99, 199, 0.2);
}

.ghost-btn {
  padding: 11px 14px;
  background: rgba(20, 56, 106, 0.07);
  color: #234977;
}

.sidebar-content {
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding: 18px;
  overflow: auto;
}
</style>
