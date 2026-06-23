<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import { computed, ref, watch } from 'vue'

import BaseModal from '@/components/common/BaseModal.vue'
import { useFolderTree } from '@/composables/useFolderTree'
import { useThumbnailSelection } from '@/composables/useThumbnailSelection'
import { useImageStore } from '@/stores/imageStore'
import { clampPageSelection, createPageSelectionSummary, normalizePageSelection } from '@/utils/pageSelection'

const props = defineProps<{
  modelValue: boolean
  selectedPages: number[]
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'confirm', pages: number[]): void
}>()

const imageStore = useImageStore()

const images = computed(() => imageStore.images)
const totalImages = computed(() => images.value.length)

const {
  useTreeMode,
  breadcrumbs,
  currentSubfolders,
  currentImages,
  currentFolderPath,
  enterFolder,
  goUp,
  navigateTo,
  getFolderImageCount,
  resetToRoot,
} = useFolderTree(images)

const {
  getImageGlobalIndex,
  getStatusType,
  getThumbnailTitle,
  isTranslated,
  failedPages,
  completedPages,
  pendingPages,
  labeledPages,
} = useThumbnailSelection(images)

const draftSelectedPages = ref<number[]>([])

watch(
  () => props.modelValue,
  (isOpen) => {
    if (isOpen) {
      draftSelectedPages.value = clampPageSelection(props.selectedPages, totalImages.value)
      resetToRoot()
    }
  },
  { immediate: true }
)

watch(totalImages, (count) => {
  draftSelectedPages.value = clampPageSelection(draftSelectedPages.value, count)
})

const normalizedDraftSelection = computed(() => normalizePageSelection(draftSelectedPages.value))
const selectedCount = computed(() => normalizedDraftSelection.value.length)
const draftSummary = computed(() => createPageSelectionSummary(normalizedDraftSelection.value))

function closeModal(): void {
  emit('update:modelValue', false)
}

function confirmSelection(): void {
  emit('confirm', normalizedDraftSelection.value)
  closeModal()
}

function togglePage(page: number): void {
  const next = new Set(normalizedDraftSelection.value)
  if (next.has(page)) {
    next.delete(page)
  } else {
    next.add(page)
  }
  draftSelectedPages.value = [...next]
}

function isSelected(page: number): boolean {
  return normalizedDraftSelection.value.includes(page)
}

function selectAllPages(): void {
  draftSelectedPages.value = Array.from({ length: totalImages.value }, (_, index) => index + 1)
}

function clearSelection(): void {
  draftSelectedPages.value = []
}

function replaceSelection(pages: number[]): void {
  draftSelectedPages.value = clampPageSelection(pages, totalImages.value)
}

function handleThumbnailClick(index: number): void {
  togglePage(index + 1)
}

function handleFolderClick(folderPath: string): void {
  enterFolder(folderPath)
}

function handleBreadcrumbClick(path: string): void {
  navigateTo(path)
}
</script>

<template>
  <BaseModal
    :model-value="modelValue"
    title="指定翻译页码"
    size="full"
    custom-class="page-selection-modal"
    width="min(1180px, 95vw)"
    height="min(88vh, 920px)"
    background="var(--color-surface-base)"
    border="1px solid #dbe4ef"
    border-radius="14px"
    box-shadow="0 20px 50px rgba(28, 45, 72, .18)"
    header-padding="18px 22px"
    header-border="1px solid #e2e9f2"
    title-color="#20314f"
    title-font-size="22px"
    title-font-weight="700"
    body-padding-value="18px 20px"
    body-background="#f4f7f9"
    footer-padding="16px 20px"
    footer-border="1px solid #e2e9f2"
    footer-background="var(--color-surface-base)"
    @update:model-value="emit('update:modelValue', $event)"
    @close="closeModal"
  >
    <div class="page-selection-shell">
      <section class="page-selection-summary-card">
        <div class="page-selection-summary-main">
          <div class="page-selection-summary-title">页码选择</div>
          <div class="page-selection-summary-text">{{ draftSummary }}</div>
        </div>
        <div class="page-selection-summary-meta">
          <span class="page-selection-chip">共 {{ totalImages }} 张</span>
          <span class="page-selection-chip">已选 {{ selectedCount }} 张</span>
        </div>
      </section>

      <section class="page-selection-toolbar-card">
        <div class="page-selection-toolbar-row">
          <UiButton variant="toolbar" type="button" class="page-selection-toolbar-btn" @click="selectAllPages">全选</UiButton>
          <UiButton variant="toolbar" type="button" class="page-selection-toolbar-btn" @click="clearSelection">清空</UiButton>
          <UiButton variant="toolbar" type="button" class="page-selection-toolbar-btn page-selection-filter-failed" @click="replaceSelection(failedPages)">失败页</UiButton>
          <UiButton variant="toolbar" type="button" class="page-selection-toolbar-btn" @click="replaceSelection(pendingPages)">未翻译页</UiButton>
          <UiButton variant="toolbar" type="button" class="page-selection-toolbar-btn" @click="replaceSelection(completedPages)">已翻译页</UiButton>
          <UiButton variant="toolbar" type="button" class="page-selection-toolbar-btn" @click="replaceSelection(labeledPages)">手动标注页</UiButton>
        </div>
      </section>

      <section class="page-selection-browser-card">
        <template v-if="useTreeMode">
          <div class="page-selection-breadcrumb-nav">
            <template v-for="(crumb, idx) in breadcrumbs" :key="crumb.path">
              <span
                class="page-selection-breadcrumb-item"
                :class="{ active: idx === breadcrumbs.length - 1 }"
                @click="idx < breadcrumbs.length - 1 && handleBreadcrumbClick(crumb.path)"
              >
                {{ idx === 0 ? '📁' : '' }}{{ crumb.name }}
              </span>
              <span v-if="idx < breadcrumbs.length - 1" class="page-selection-breadcrumb-sep">/</span>
            </template>
          </div>

          <div
            v-if="currentFolderPath"
            class="page-selection-folder-back-btn"
            @click="goUp"
          >
            <span class="back-icon">⬅️</span>
            <span>返回上级</span>
          </div>

          <div class="page-selection-grid">
            <div
              v-for="subfolder in currentSubfolders"
              :key="subfolder.path"
              class="page-selection-folder-item"
              @click="handleFolderClick(subfolder.path)"
            >
              <span class="folder-icon">📁</span>
              <div class="folder-info">
                <span class="folder-name" :title="subfolder.name">{{ subfolder.name }}</span>
                <span class="folder-count">({{ getFolderImageCount(subfolder) }})</span>
              </div>
            </div>

            <UiButton
              variant="toolbar"
              v-for="image in currentImages"
              :key="image.id"
              type="button"
              class="page-selection-thumbnail"
              :class="{ active: isSelected(getImageGlobalIndex(image) + 1) }"
              :title="getThumbnailTitle(image)"
              @click="handleThumbnailClick(getImageGlobalIndex(image))"
            >
              <img
                v-if="image.originalDataURL"
                :src="image.originalDataURL"
                :alt="image.fileName"
                class="thumbnail-image"
              >
              <span class="page-number-indicator">{{ getImageGlobalIndex(image) + 1 }}</span>
              <span v-if="isTranslated(image)" class="translated-indicator">✓</span>
              <span v-if="getStatusType(image) === 'failed'" class="translation-failed-indicator">!</span>
              <span v-else-if="getStatusType(image) === 'labeled'" class="labeled-indicator">✏️</span>
              <div v-if="getStatusType(image) === 'processing'" class="thumbnail-processing-indicator">⟳</div>
              <div v-if="isSelected(getImageGlobalIndex(image) + 1)" class="page-selection-selected-badge">已选</div>
            </UiButton>
          </div>
        </template>

        <template v-else>
          <div class="page-selection-grid">
            <UiButton
              variant="toolbar"
              v-for="(image, index) in images"
              :key="image.id"
              type="button"
              class="page-selection-thumbnail"
              :class="{ active: isSelected(index + 1) }"
              :title="getThumbnailTitle(image)"
              @click="handleThumbnailClick(index)"
            >
              <img
                v-if="image.originalDataURL"
                :src="image.originalDataURL"
                :alt="image.fileName"
                class="thumbnail-image"
              >
              <span class="page-number-indicator">{{ index + 1 }}</span>
              <span v-if="isTranslated(image)" class="translated-indicator">✓</span>
              <span v-if="getStatusType(image) === 'failed'" class="translation-failed-indicator">!</span>
              <span v-else-if="getStatusType(image) === 'labeled'" class="labeled-indicator">✏️</span>
              <div v-if="getStatusType(image) === 'processing'" class="thumbnail-processing-indicator">⟳</div>
              <div v-if="isSelected(index + 1)" class="page-selection-selected-badge">已选</div>
            </UiButton>
          </div>
        </template>
      </section>
    </div>

    <template #footer>
      <UiButton variant="secondary" type="button" class="page-selection-footer-btn" @click="closeModal">
        取消
      </UiButton>
      <UiButton variant="primary" type="button" class="page-selection-footer-btn page-selection-confirm-btn" @click="confirmSelection">
        确定
      </UiButton>
    </template>
  </BaseModal>
</template>

<style scoped>
.page-selection-shell {
  --page-selection-modal-border-default: #dbe4ef;
  --page-selection-modal-border-strong: #e2e9f2;
  --page-selection-modal-border-muted: #d3deed;
  --page-selection-modal-border-subtle: #d8e3f1;
  --page-selection-modal-border-hover: #cfdcec;
  --page-selection-modal-border-active: #4a82ce;
  --page-selection-modal-border-focus: #f0d78c;
  --page-selection-modal-shadow-raised: rgba(28, 45, 72, .07);
  --page-selection-modal-shadow-floating: rgba(74, 130, 206, .12);
  --page-selection-modal-shadow-strong: rgba(240, 215, 140, .4);
  --page-selection-modal-shadow-soft: rgba(0, 0, 0, .1);
  --page-selection-modal-shadow-focus: rgba(52, 152, 219, .45);
  --page-selection-modal-shadow-glow: rgba(0, 0, 0, .25);
  --page-selection-modal-shadow-inset: rgba(0, 0, 0, .3);
  --page-selection-modal-surface-raised: #f4f8fd;
  --page-selection-modal-surface-muted: #f5f8fd;
  --page-selection-modal-surface-subtle: #fbfdff;
  --page-selection-modal-surface-hover: #4a82ce;
  --page-selection-modal-surface-active: #3f74bc;
  --page-selection-modal-surface-selected: #e8f4fd;
  --page-selection-modal-surface-overlay: #d4e8f8;
  --page-selection-modal-surface-inverse: #ffe8b8;
  --page-selection-modal-surface-contrast: #f2f8ff;
  --page-selection-modal-surface-tint: rgba(74, 130, 206, .92);
  --page-selection-modal-surface-soft: rgba(53, 152, 219, .8);
  --page-selection-modal-surface-strong: rgba(255, 0, 0, .8);
  --page-selection-modal-surface-stronger: rgba(0, 123, 255, .8);
  --page-selection-modal-surface-highlight: rgba(0, 0, 0, .6);
  --page-selection-modal-surface-highlight-strong: rgba(34, 197, 94, .9);
  --page-selection-modal-text-secondary: #273959;
  --page-selection-modal-text-muted: #51637f;
  --page-selection-modal-text-subtle: #5b6f8e;
  --page-selection-modal-text-supporting: #243552;
  --page-selection-modal-text-disabled: #94a3b8;
  --page-selection-modal-text-inverse: #5a4a00;
  --page-selection-modal-text-brand: #8a7a30;

  display: flex;
  flex-direction: column;
  gap: 14px;
  min-height: 100%;
}

.page-selection-summary-card,
.page-selection-toolbar-card,
.page-selection-browser-card {
  background: var(--color-surface-base);
  border: 1px solid var(--page-selection-modal-border-default);
  border-radius: 14px;
  box-shadow: 0 8px 20px var(--page-selection-modal-shadow-raised);
}

.page-selection-summary-card {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  padding: 16px 18px;
}

.page-selection-summary-title {
  color: var(--page-selection-modal-text-secondary);
  font-size: 16px;
  font-weight: 700;
}

.page-selection-summary-text {
  margin-top: 4px;
  color: var(--page-selection-modal-text-muted);
  font-size: 14px;
}

.page-selection-summary-meta {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.page-selection-chip {
  display: inline-flex;
  align-items: center;
  padding: 5px 10px;
  border: 1px solid var(--page-selection-modal-border-muted);
  border-radius: 999px;
  background: var(--page-selection-modal-surface-raised);
  color: var(--page-selection-modal-text-subtle);
  font-size: 12px;
  font-weight: 600;
}

.page-selection-toolbar-card {
  padding: 12px;
  background: var(--page-selection-modal-surface-muted);
  border-color: var(--page-selection-modal-border-subtle);
}

.page-selection-toolbar-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.page-selection-toolbar-btn,
.page-selection-footer-btn {
  min-height: 38px;
  padding: 9px 14px;
  border-radius: 10px;
  border: 1px solid var(--page-selection-modal-border-hover);
  background: var(--page-selection-modal-surface-subtle);
  color: var(--page-selection-modal-text-supporting);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
}

.page-selection-toolbar-btn:hover,
.page-selection-footer-btn.secondary:hover {
  border-color: var(--page-selection-modal-border-active);
  box-shadow: 0 0 0 3px var(--page-selection-modal-shadow-floating);
}

.page-selection-footer-btn.primary {
  border-color: var(--page-selection-modal-border-active);
  background: var(--page-selection-modal-surface-hover);
  color: var(--color-text-inverse);
}

.page-selection-footer-btn.primary:hover {
  background: var(--page-selection-modal-surface-active);
}

.page-selection-browser-card {
  flex: 1;
  min-height: 0;
  padding: 14px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.page-selection-breadcrumb-nav {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 2px;
  padding: 8px 10px;
  background: var(--color-surface-quiet);
  border-radius: 6px;
  margin-bottom: 10px;
  font-size: 12px;
  line-height: 1.4;
}

.page-selection-breadcrumb-item {
  color: var(--color-text-link);
  cursor: pointer;
  word-break: break-word;
}

.page-selection-breadcrumb-item.active {
  color: var(--color-text-heading);
  font-weight: 600;
  cursor: default;
}

.page-selection-breadcrumb-sep {
  color: var(--page-selection-modal-text-disabled);
  margin: 0 2px;
}

.page-selection-folder-back-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  background: linear-gradient(135deg, var(--page-selection-modal-surface-selected) 0%, var(--page-selection-modal-surface-overlay) 100%);
  border-radius: 8px;
  cursor: pointer;
  margin-bottom: 12px;
  font-size: 13px;
  color: var(--color-text-link);
  transition: all 0.2s;
}

.page-selection-grid {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 14px;
  padding-right: 4px;
}

.page-selection-folder-item {
  position: relative;
  min-height: 88px;
  padding: 12px;
  background: linear-gradient(135deg, var(--color-surface-warning-subtle) 0%, var(--color-surface-warning-warm) 100%);
  border: 1px solid var(--page-selection-modal-border-focus);
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s;
}

.page-selection-folder-item:hover {
  background: linear-gradient(135deg, var(--color-surface-warning-warm) 0%, var(--page-selection-modal-surface-inverse) 100%);
  transform: translateY(-1px);
  box-shadow: 0 2px 8px var(--page-selection-modal-shadow-strong);
}

.page-selection-folder-item .folder-icon {
  position: absolute;
  top: -4px;
  right: -4px;
  font-size: 14px;
  background: var(--color-surface-base);
  border-radius: 4px;
  padding: 1px 2px;
  box-shadow: 0 1px 3px var(--page-selection-modal-shadow-soft);
}

.page-selection-folder-item .folder-name {
  font-size: 13px;
  font-weight: 500;
  color: var(--page-selection-modal-text-inverse);
  word-break: break-word;
  line-height: 1.4;
}

.page-selection-folder-item .folder-count {
  font-size: 11px;
  color: var(--page-selection-modal-text-brand);
  margin-left: 4px;
}

.page-selection-thumbnail {
  position: relative;
  width: 100%;
  padding: 5px;
  border: 2px solid var(--color-border-muted);
  border-radius: 10px;
  background: var(--color-surface-base);
  cursor: pointer;
  overflow: hidden;
  transition: all 0.3s ease;
}

.page-selection-thumbnail:hover,
.page-selection-thumbnail.active {
  border-color: var(--color-border-accent);
  box-shadow: 0 0 8px var(--page-selection-modal-shadow-focus);
  transform: translateY(-2px);
}

.page-selection-thumbnail.active {
  background: linear-gradient(180deg, var(--color-surface-plain) 0%, var(--page-selection-modal-surface-contrast) 100%);
}

.page-selection-thumbnail .thumbnail-image {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 6px;
}

.page-selection-selected-badge {
  position: absolute;
  top: 28px;
  left: 3px;
  padding: 2px 6px;
  border-radius: 999px;
  background: var(--page-selection-modal-surface-tint);
  color: var(--color-text-inverse);
  font-size: 11px;
  font-weight: 700;
  z-index: var(--z-local-control);
  box-shadow: 0 1px 3px var(--page-selection-modal-shadow-glow);
}

.thumbnail-processing-indicator {
  position: absolute;
  top: 5px;
  right: 5px;
  background-color: var(--page-selection-modal-surface-soft);
  color: white;
  width: 15px;
  height: 15px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  z-index: var(--z-local-control);
  animation: pageSelectionPulse 1.5s infinite;
}

.translation-failed-indicator,
.labeled-indicator,
.page-number-indicator,
.translated-indicator {
  position: absolute;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 1px 3px var(--page-selection-modal-shadow-inset);
}

.translation-failed-indicator {
  bottom: 3px;
  right: 3px;
  background-color: var(--page-selection-modal-surface-strong);
  color: white;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  font-size: 12px;
  font-weight: bold;
  z-index: var(--z-local-toolbar-raised);
}

.labeled-indicator {
  bottom: 3px;
  right: 3px;
  background-color: var(--page-selection-modal-surface-stronger);
  color: white;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  font-size: 12px;
  z-index: var(--z-local-toolbar);
}

.page-number-indicator {
  bottom: 3px;
  left: 3px;
  background-color: var(--page-selection-modal-surface-highlight);
  color: white;
  min-width: 18px;
  height: 18px;
  padding: 0 4px;
  border-radius: 3px;
  font-size: 11px;
  font-weight: 500;
  z-index: var(--z-local-raised);
}

.translated-indicator {
  top: 3px;
  left: 3px;
  background-color: var(--page-selection-modal-surface-highlight-strong);
  color: white;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  font-size: 12px;
  font-weight: bold;
  z-index: var(--z-local-control);
}

@keyframes pageSelectionPulse {
  0%,
  100% {
    opacity: 1;
  }

  50% {
    opacity: 0.55;
  }
}

@media (--breakpoint-lg-down) {
  .page-selection-summary-card {
    flex-direction: column;
    align-items: flex-start;
  }

  .page-selection-grid {
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  }
}
</style>
