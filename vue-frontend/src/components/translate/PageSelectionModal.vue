<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductBreadcrumbTrail from '@/components/product/ProductBreadcrumbTrail.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductFolderCard from '@/components/product/ProductFolderCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import ProductThumbnailGrid from '@/components/product/ProductThumbnailGrid.vue'
import type { ProductThumbnailGridItem } from '@/components/product/ProductThumbnailGrid.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import { computed, ref, watch } from 'vue'

import BaseModal from '@/components/common/BaseModal.vue'
import { useFolderTree } from '@/composables/useFolderTree'
import { useThumbnailSelection } from '@/composables/useThumbnailSelection'
import { useImageStore } from '@/stores/imageStore'
import type { ImageData } from '@/types/image'
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
const summaryChipItems = computed<ProductChipItem[]>(() => [
  { id: 'total', label: `共 ${totalImages.value} 张`, tone: 'neutral' },
  { id: 'selected', label: `已选 ${selectedCount.value} 张`, tone: 'primary' },
])
const flatThumbnailItems = computed<ProductThumbnailGridItem[]>(() => {
  return images.value.map((image, index) => buildThumbnailItem(image, index))
})
const currentFolderThumbnailItems = computed<ProductThumbnailGridItem[]>(() => {
  return currentImages.value.map((image) => buildThumbnailItem(image, getImageGlobalIndex(image)))
})

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

function handleThumbnailSelect(id: string | number): void {
  if (typeof id !== 'number') return
  handleThumbnailClick(id)
}

function handleFolderClick(folderPath: string): void {
  enterFolder(folderPath)
}

function handleBreadcrumbClick(path: string): void {
  navigateTo(path)
}

function buildThumbnailItem(image: ImageData, index: number): ProductThumbnailGridItem {
  const page = index + 1
  const selected = isSelected(page)
  const statusType = getStatusType(image)
  return {
    id: index,
    src: image.sourceAssetUrl,
    alt: image.fileName,
    label: String(page),
    selected,
    selectedBadge: selected ? '已选' : undefined,
    marked: isTranslated(image),
    cornerLabel: statusType === 'failed'
      ? '!'
      : statusType === 'labeled'
        ? '标'
        : statusType === 'processing'
          ? '处理中'
          : undefined,
    fallbackLabel: String(page),
    ariaLabel: `${selected ? '取消选择' : '选择'}第 ${page} 页：${image.fileName}`,
    disabledTitle: getThumbnailTitle(image),
  }
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
    body-padding="compact"
    scroll-mode="contained"
    @update:model-value="emit('update:modelValue', $event)"
    @close="closeModal"
  >
    <div class="page-selection-shell">
      <ProductStatusBanner
        class="page-selection-summary-banner"
        tone="neutral"
        role="note"
        title="页码选择"
      >
        {{ draftSummary }}
        <template #actions>
          <ProductChipList
            aria-label="页码选择统计"
            :items="summaryChipItems"
          />
        </template>
      </ProductStatusBanner>

      <ProductActionRow
        class="page-selection-shortcuts"
        aria-label="页码选择快捷操作"
        justify="start"
      >
        <UiButton variant="secondary" size="sm" type="button" @click="selectAllPages">全选</UiButton>
        <UiButton variant="secondary" size="sm" type="button" @click="clearSelection">清空</UiButton>
        <UiButton variant="danger" size="sm" type="button" @click="replaceSelection(failedPages)">失败页</UiButton>
        <UiButton variant="secondary" size="sm" type="button" @click="replaceSelection(pendingPages)">未翻译页</UiButton>
        <UiButton variant="secondary" size="sm" type="button" @click="replaceSelection(completedPages)">已翻译页</UiButton>
        <UiButton variant="secondary" size="sm" type="button" @click="replaceSelection(labeledPages)">手动标注页</UiButton>
      </ProductActionRow>

      <section class="page-selection-browser-card">
        <template v-if="useTreeMode">
          <ProductBreadcrumbTrail
            class="page-selection-breadcrumb"
            :items="breadcrumbs"
            @select="handleBreadcrumbClick"
          />

          <UiButton
            v-if="currentFolderPath"
            variant="secondary"
            tone="primary"
            size="sm"
            block
            type="button"
            class="page-selection-folder-back-button"
            @click="goUp"
          >
            <UiIcon class="page-selection-folder-back-icon" name="chevron-right" size="14" />
            <span>返回上级</span>
          </UiButton>

          <div v-if="currentSubfolders.length > 0" class="page-selection-folder-grid">
            <ProductFolderCard
              v-for="subfolder in currentSubfolders"
              :key="subfolder.path"
              class="page-selection-folder-card"
              :count="getFolderImageCount(subfolder)"
              :count-id="subfolder.path"
              :folder-name="subfolder.name"
              :aria-label="`打开文件夹 ${subfolder.name}`"
              @select="handleFolderClick(subfolder.path)"
            />
          </div>

          <ProductThumbnailGrid
            v-if="currentFolderThumbnailItems.length > 0"
            class="page-selection-thumbnail-grid"
            aria-label="选择翻译页码"
            :items="currentFolderThumbnailItems"
            @select="handleThumbnailSelect"
          />
        </template>

        <template v-else>
          <ProductThumbnailGrid
            class="page-selection-thumbnail-grid"
            aria-label="选择翻译页码"
            :items="flatThumbnailItems"
            @select="handleThumbnailSelect"
          />
        </template>
      </section>
    </div>

    <template #footer>
      <ProductActionRow
        variant="dialog"
        aria-label="指定翻译页码操作"
      >
        <UiButton variant="secondary" type="button" @click="closeModal">取消</UiButton>
        <UiButton
          variant="primary"
          type="button"
          data-testid="confirm-page-selection-button"
          @click="confirmSelection"
        >
          确定
        </UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<style scoped>
.page-selection-shell {
  --page-selection-modal-border-default: var(--color-border-muted);
  --page-selection-modal-border-subtle: var(--color-border-soft);
  --page-selection-modal-border-focus: var(--color-status-warning);
  --page-selection-modal-shadow-raised: var(--shadow-soft);
  --page-selection-modal-shadow-strong: var(--shadow-action-success);
  --page-selection-modal-surface-warning: var(--color-status-warning-surface-soft);
  --page-selection-modal-surface-warning-raised: var(--color-status-warning-surface-raised);

  display: flex;
  flex-direction: column;
  gap: 14px;
  min-height: 100%;
}

.page-selection-browser-card {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-height: 0;
  padding: 14px;
  overflow: hidden;
  background: var(--color-surface-base);
  border: 1px solid var(--page-selection-modal-border-default);
  border-radius: 10px;
  box-shadow: 0 8px 20px var(--page-selection-modal-shadow-raised);
}

.page-selection-shortcuts {
  padding: 12px;
  border: 1px solid var(--page-selection-modal-border-subtle);
  border-radius: 10px;
  background: var(--color-surface-muted);
}

.page-selection-breadcrumb {
  margin-bottom: 10px;
}

.page-selection-folder-back-button {
  margin-bottom: 12px;
  justify-content: flex-start;
}

.page-selection-folder-back-icon {
  transform: rotate(180deg);
}

.page-selection-folder-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 14px;
  margin-bottom: 14px;
}

.page-selection-thumbnail-grid {
  --product-thumbnail-grid-min-size: 150px;
  --product-thumbnail-grid-aspect-ratio: 3 / 4;

  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding-right: 4px;
}

.page-selection-folder-card {
  --product-record-card-background: var(--page-selection-modal-surface-warning);
  --product-record-card-border: var(--page-selection-modal-border-focus);
  --product-record-card-accent: var(--color-status-warning);
  --product-record-card-shadow-hover: 0 2px 8px var(--page-selection-modal-shadow-strong);
  --product-record-card-padding: 12px;
  --product-record-card-radius: 10px;
  --product-record-card-gap: 8px;

  min-height: 88px;
  transition: transform 0.2s ease;
}

.page-selection-folder-card:hover {
  --product-record-card-background: var(--page-selection-modal-surface-warning-raised);

  transform: translateY(-1px);
}

@media (--breakpoint-lg-down) {
  .page-selection-folder-grid,
  .page-selection-thumbnail-grid {
    --product-thumbnail-grid-min-size: 120px;

    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  }
}
</style>
