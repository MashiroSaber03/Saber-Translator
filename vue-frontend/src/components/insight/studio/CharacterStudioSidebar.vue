<template>
  <div class="character-studio-sidebar">
    <div class="character-studio-sidebar__toolbar">
      <div class="character-studio-sidebar__toolbar-copy">
        <div class="character-studio-sidebar__kicker">导航与资源</div>
        <h2 class="character-studio-sidebar__title">当前书籍角色工坊</h2>
        <p class="character-studio-sidebar__description">
          从分析候选锁定角色名，再用 AI 补全整卡；也可以直接空白新建或导入外部角色卡。
        </p>
      </div>

      <div class="character-studio-sidebar__actions">
        <ProductSearchField
          :model-value="search"
          class="character-studio-sidebar__search"
          aria-label="搜索角色资源"
          placeholder="搜索角色 / 标签 / 来源"
          :show-icon="false"
          :disabled="workspaceLoading"
          @update:model-value="$emit('update:search', $event)"
          @clear="$emit('update:search', '')"
        />
        <ProductActionRow
          appearance="accent"
          class="character-studio-sidebar__action-row"
          aria-label="角色资源操作"
          justify="start"
        >
          <UiButton
            variant="primary"
            class="character-studio-sidebar__create-action"
            :disabled="creatingManual || importingFile"
            @click="$emit('create-manual')"
          >
            {{ creatingManual ? '新建中...' : '空白新建' }}
          </UiButton>
          <UiButton
            variant="secondary"
            class="character-studio-sidebar__import-action"
            :disabled="creatingManual || importingFile"
            @click="pickImport"
          >
            {{ importingFile ? '导入中...' : '导入' }}
          </UiButton>
        </ProductActionRow>
      </div>

      <UiFileInput
        ref="fileInput"
        hidden
        accept=".json,.png,.jpg,.jpeg,.webp,.gif,.bmp"
        @files-change="handleFileSelect"
      />
    </div>

    <div class="character-studio-sidebar__content">
      <DocumentListPane
        :documents="documents"
        :current-document-id="currentDocumentId"
        :opening-document-id="openingDocumentId"
        @open="$emit('open-document', $event)"
      />

      <CandidateListPane
        :candidates="candidates"
        :has-timeline="hasTimeline"
        :creating-candidate-id="creatingCandidateId"
        @create="$emit('create-from-candidate', $event)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductSearchField from '@/components/product/ProductSearchField.vue'
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
  creatingCandidateId: string
}>()

const emit = defineEmits<{
  (e: 'update:search', value: string): void
  (e: 'open-document', docId: string): void
  (e: 'create-manual'): void
  (e: 'create-from-candidate', candidateId: string): void
  (e: 'import-file', file: File): void
}>()

const fileInput = ref<InstanceType<typeof UiFileInput> | null>(null)

function pickImport() {
  fileInput.value?.click()
}

function handleFileSelect(files: File[]) {
  const file = files[0]
  if (!file) return
  emit('import-file', file)
  fileInput.value?.clear()
}
</script>

<style scoped>
.character-studio-sidebar {
  --character-studio-sidebar-toolbar-background-start: color-mix(
    in srgb,
    var(--color-action-brand) 12%,
    transparent
  );
  --character-studio-sidebar-toolbar-background-end: color-mix(
    in srgb,
    var(--color-surface-raised) 88%,
    transparent
  );
  --character-studio-sidebar-shell-background: color-mix(
    in srgb,
    var(--color-surface-card) 88%,
    transparent
  );
  --character-studio-sidebar-kicker-text: var(--color-text-muted);
  --character-studio-sidebar-title-text: var(--color-text-brand);
  --character-studio-sidebar-description-text: var(--studio-text-subtle);

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

.character-studio-sidebar__toolbar {
  flex-shrink: 0;
  padding: 18px 18px 16px;
  border-bottom: 1px solid var(--studio-border-default);
  background: linear-gradient(
    180deg,
    var(--character-studio-sidebar-toolbar-background-start),
    var(--character-studio-sidebar-toolbar-background-end)
  );
}

.character-studio-sidebar__kicker {
  font-size: 11px;
  letter-spacing: 0;
  text-transform: uppercase;
  color: var(--character-studio-sidebar-kicker-text);
}

.character-studio-sidebar__title {
  margin: 8px 0 0;
  font-size: 22px;
  line-height: 1.24;
  color: var(--character-studio-sidebar-title-text);
}

.character-studio-sidebar__description {
  margin: 10px 0 0;
  color: var(--character-studio-sidebar-description-text);
  font-size: 13px;
  line-height: 1.7;
}

.character-studio-sidebar__actions {
  margin-top: 16px;
}

.character-studio-sidebar__search {
  --product-search-field-input-padding: 12px 14px;
  --product-search-field-radius: 14px;
  --product-search-field-background: color-mix(in srgb, var(--color-surface-base) 92%, transparent);
  --product-search-field-border: 1px solid
    color-mix(in srgb, var(--color-text-link-strong) 12%, transparent);

  width: 100%;
}

.character-studio-sidebar__action-row {
  margin-top: 12px;
}

.character-studio-sidebar__create-action {
  --ui-button-padding: 11px 16px;
  --ui-button-radius: 14px;
  --ui-button-primary-background: linear-gradient(
    135deg,
    var(--color-action-brand),
    var(--color-action-brand-strong)
  );
  --ui-button-primary-shadow: 0 12px 24px
    color-mix(in srgb, var(--color-action-brand) 20%, transparent);

  flex: 1 1 150px;
}

.character-studio-sidebar__import-action {
  --ui-button-padding: 11px 14px;
  --ui-button-radius: 14px;
  --ui-button-secondary-background: color-mix(
    in srgb,
    var(--color-text-link-strong) 7%,
    transparent
  );
  --ui-button-secondary-border: none;
  --ui-button-secondary-color: var(--color-text-link-strong);

  flex: 0 0 auto;
}

.character-studio-sidebar__content {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 18px;
  padding: 18px;
  overflow: auto;
}
</style>
