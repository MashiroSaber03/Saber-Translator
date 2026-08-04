<template>
  <BaseModal
    :model-value="visible"
    :show-header="false"
    custom-class="reference-image-selector-modal"
    frame-variant="outlined"
    body-padding="none"
    scroll-mode="contained"
    width="min(1120px, calc(100vw - 48px))"
    max-height="min(88vh, 980px)"
    body-display="flex"
    body-direction="column"
    body-min-height="0"
    @update:model-value="value => { if (!value) handleCancel() }"
  >
    <div class="reference-image-selector">
      <div class="reference-image-selector__header">
        <h3 class="reference-image-selector__title">选择参考图 ({{ selectedCount }}/{{ maxCount }})</h3>
        <ProductActionRow
          class="reference-image-selector__batch-actions"
          aria-label="参考图批量操作"
          justify="start"
        >
          <UiButton variant="secondary" size="sm" @click="autoSelectLast">
            自动选择最后{{ maxCount }}张
          </UiButton>
          <UiButton variant="secondary" size="sm" @click="clearSelection">
            清空
          </UiButton>
        </ProductActionRow>
        <ProductActionRow
          class="reference-image-selector__dialog-actions"
          aria-label="参考图选择器操作"
          justify="end"
          variant="dialog"
        >
          <UiButton variant="secondary" size="sm" @click="handleCancel">取消</UiButton>
          <UiButton variant="primary" size="sm" @click="handleConfirm">确定</UiButton>
        </ProductActionRow>
        <UiIconButton
          class="reference-image-selector__close"
          label="关闭参考图选择器"
          title="关闭"
          variant="plain"
          size="sm"
          shape="circle"
          @click="handleCancel"
        >
          <UiIcon name="x" size="18" />
        </UiIconButton>
      </div>

      <div
        v-if="mode === 'image' && (characterForms.length > 0 || hasMoreCharacterForms)"
        class="reference-image-selector__character-section"
      >
        <div class="reference-image-selector__section-label">
          <span>角色档案</span>
          <span class="reference-image-selector__section-hint">（自动添加，不计入选择数量）</span>
        </div>
        <ProductThumbnailGrid
          v-if="characterForms.length > 0"
          class="reference-image-selector__character-thumbnail-grid"
          aria-label="角色档案参考图"
          :items="characterThumbnailItems"
        />
        <UiButton
          v-if="hasMoreCharacterForms"
          class="reference-image-selector__load-more-forms"
          variant="secondary"
          size="sm"
          block
          :disabled="loadingMoreCharacterForms"
          @click="$emit('load-more-character-forms')"
        >
          {{ loadingMoreCharacterForms ? '加载中...' : '加载更多角色参考图' }}
        </UiButton>
      </div>

      <div class="reference-image-selector__manga-section">
        <div class="reference-image-selector__section-label">
          <span>漫画图片</span>
        </div>
        <UiButton
          v-if="hasOlderOriginalImages"
          class="reference-image-selector__load-older"
          variant="secondary"
          size="sm"
          :disabled="loadingOlderOriginalImages"
          @click="$emit('load-older-originals')"
        >
          {{ loadingOlderOriginalImages ? '加载中...' : '加载更早的原作页面' }}
        </UiButton>
        <VirtualThumbnailGrid
          ref="thumbnailsGrid"
          class="reference-image-selector__scroll reference-image-selector__thumbnail-grid"
          aria-label="漫画参考图选择"
          :items="selectableThumbnailItems"
          :max-height="560"
          :min-item-width="110"
          @select="toggleSelectionByToken"
        />
      </div>
    </div>
  </BaseModal>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import BaseModal from '@/components/common/BaseModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductThumbnailGrid from '@/components/product/ProductThumbnailGrid.vue'
import type { ProductThumbnailGridItem } from '@/components/product/ProductThumbnailGrid.vue'
import VirtualThumbnailGrid from '@/components/virtual/VirtualThumbnailGrid.vue'
import { ref, computed, watch, nextTick } from 'vue'
import type { MangaImageInfo, CharacterFormInfo } from '@/api/continuation'
import * as insightApi from '@/api/insight'

const props = defineProps<{
  visible: boolean
  mode: 'script' | 'image'
  maxCount: number
  originalImages: MangaImageInfo[]
  continuationImages: MangaImageInfo[]
  characterForms: CharacterFormInfo[]
  initialSelection: string[]
  bookId: string
  hasOlderOriginalImages?: boolean
  loadingOlderOriginalImages?: boolean
  hasMoreCharacterForms?: boolean
  loadingMoreCharacterForms?: boolean
}>()

const emit = defineEmits<{
  'update:visible': [value: boolean]
  'confirm': [selectedTokens: string[]]
  'cancel': []
  'load-older-originals': []
  'load-more-character-forms': []
}>()

const selectedTokens = ref<string[]>([])
const thumbnailsGrid = ref<InstanceType<typeof VirtualThumbnailGrid> | null>(null)
const selectedCount = computed(() => selectedTokens.value.length)
const characterThumbnailItems = computed<ProductThumbnailGridItem[]>(() => {
  return props.characterForms.map(form => {
    const label = `${form.character_name} - ${form.form_name}`

    return {
      id: form.token || `${form.character_name}-${form.form_id}`,
      alt: label,
      fallbackLabel: '角色图缺失',
      interactive: false,
      label,
      src: form.has_image && form.path ? getImageUrl(form.path) : '',
    }
  })
})
const selectableThumbnailItems = computed<ProductThumbnailGridItem[]>(() => {
  return [
    ...props.originalImages.map(img => createOriginalThumbnailItem(img)),
    ...(props.mode === 'image' ? props.continuationImages.map(img => createContinuationThumbnailItem(img)) : []),
  ]
})

watch(() => props.visible, (newVisible) => {
  if (newVisible) {
    const availableTokens = new Set(
      [
        ...props.originalImages,
        ...(props.mode === 'image' ? props.continuationImages : []),
      ]
        .map(img => img.token)
        .filter(Boolean)
    )

    if (props.initialSelection && props.initialSelection.length > 0) {
      selectedTokens.value = props.initialSelection.filter(token => availableTokens.has(token))
      if (selectedTokens.value.length === 0) {
        autoSelectLast()
      }
    } else {
      autoSelectLast()
    }

    nextTick(() => {
      scrollToBottom()
    })
  }
}, { immediate: true })

function getImageIdentifier(img: MangaImageInfo): string {
  return img.token || ''
}

function isSelected(img: MangaImageInfo): boolean {
  const identifier = getImageIdentifier(img)
  return identifier ? selectedTokens.value.includes(identifier) : false
}

function getSelectionIndex(img: MangaImageInfo): number {
  const identifier = getImageIdentifier(img)
  const index = selectedTokens.value.indexOf(identifier)
  return index >= 0 ? index + 1 : 0
}

function isThumbnailDisabled(img: MangaImageInfo): boolean {
  return !isSelected(img) && selectedCount.value >= props.maxCount
}

function getThumbnailActionLabel(img: MangaImageInfo, source: string): string {
  const action = isSelected(img) ? '取消选择' : '选择'
  return `${action}${source}第${img.page_number}页参考图`
}

function toggleSelectionByToken(identifierValue: string | number): void {
  const identifier = String(identifierValue)
  if (!identifier) return

  const index = selectedTokens.value.indexOf(identifier)
  if (index >= 0) {
    selectedTokens.value.splice(index, 1)
  } else {
    if (selectedTokens.value.length < props.maxCount) {
      selectedTokens.value.push(identifier)
    }
  }
}

function createOriginalThumbnailItem(img: MangaImageInfo): ProductThumbnailGridItem {
  return createThumbnailItem(img, '原作', {
    alt: `第${img.page_number}页`,
    fallbackLabel: '原作页缺失',
    src: img.has_image ? getOriginalThumbnailUrl(img.page_number) : '',
  })
}

function createContinuationThumbnailItem(img: MangaImageInfo): ProductThumbnailGridItem {
  return createThumbnailItem(img, '续写', {
    alt: `第${img.page_number}页续写图`,
    cornerLabel: '续写',
    fallbackLabel: '占位页',
    src: img.has_image && img.path ? getImageUrl(img.path) : '',
  })
}

function createThumbnailItem(
  img: MangaImageInfo,
  source: string,
  options: {
    alt: string
    cornerLabel?: string
    fallbackLabel: string
    src: string
  }
): ProductThumbnailGridItem {
  const identifier = getImageIdentifier(img)
  const disabled = isThumbnailDisabled(img)

  return {
    id: identifier || `${source}-${img.page_number}`,
    alt: options.alt,
    ariaLabel: getThumbnailActionLabel(img, source),
    cornerLabel: options.cornerLabel,
    disabled,
    disabledTitle: disabled ? '已达到最大数量，请先取消其他选择' : undefined,
    fallbackLabel: options.fallbackLabel,
    label: String(img.page_number),
    selected: isSelected(img),
    selectedBadge: isSelected(img) ? String(getSelectionIndex(img)) : undefined,
    src: options.src,
  }
}

function autoSelectLast(): void {
  selectedTokens.value = []

  const validImages = [
    ...props.originalImages,
    ...(props.mode === 'image' ? props.continuationImages : []),
  ]
    .filter(img => img.token && img.has_image && img.path)
    .sort((left, right) => left.page_number - right.page_number)

  const lastN = validImages.slice(-props.maxCount)
  selectedTokens.value = lastN.map(img => img.token)

  nextTick(() => {
    scrollToBottom()
  })
}

function clearSelection(): void {
  selectedTokens.value = []
}

function scrollToBottom(): void {
  thumbnailsGrid.value?.scrollToEnd()
}

function getOriginalThumbnailUrl(pageNum: number): string {
  if (!props.bookId) return ''
  return insightApi.getThumbnailUrl(props.bookId, pageNum)
}

function getImageUrl(path: string): string {
  return path
}

function handleConfirm(): void {
  emit('confirm', [...selectedTokens.value])
  emit('update:visible', false)
}

function handleCancel(): void {
  emit('cancel')
  emit('update:visible', false)
}
</script>

<style scoped>
.reference-image-selector {
  --reference-image-selector-border-muted: var(--color-status-warning);
  --reference-image-selector-surface-header: var(--color-surface-muted);
  --reference-image-selector-surface-section: var(--color-status-warning-surface-soft);
  --reference-image-selector-text-character: var(--color-text-strong);
  --reference-image-selector-text-muted: var(--color-text-supporting);
  --reference-image-selector-text-section: var(--color-text-secondary);

  display: flex;
  flex: 1;
  min-height: 0;
  flex-direction: column;
}

.reference-image-selector__header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  padding: 16px 20px;
  background: var(--reference-image-selector-surface-header);
  border-bottom: 1px solid var(--color-border-default);
  gap: 12px;
  flex-shrink: 0;
}

.reference-image-selector__title {
  flex: 1 1 180px;
  min-width: 0;
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  overflow-wrap: anywhere;
}

.reference-image-selector__batch-actions {
  display: flex;
  flex: 1 1 240px;
  flex-wrap: wrap;
  gap: 8px;
  min-width: 0;
  margin-left: 16px;
}

.reference-image-selector__dialog-actions {
  display: flex;
  flex: 0 1 auto;
  flex-wrap: wrap;
  gap: 8px;
  min-width: 0;
  margin-left: auto;
}

.reference-image-selector__close {
  color: var(--color-text-secondary);
  margin-left: 8px;
}

.reference-image-selector__close:hover {
  color: var(--color-text-default);
}

.reference-image-selector__character-section {
  padding: 12px 20px;
  background: var(--reference-image-selector-surface-section);
  border-bottom: 1px solid var(--reference-image-selector-border-muted);
  flex-shrink: 0;
}

.reference-image-selector__section-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--reference-image-selector-text-character);
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.reference-image-selector__section-hint {
  font-weight: 400;
  font-size: 12px;
  color: var(--reference-image-selector-text-muted);
}

.reference-image-selector__manga-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  padding: 16px 20px;
}

.reference-image-selector__manga-section .reference-image-selector__section-label {
  color: var(--reference-image-selector-text-section);
  margin-bottom: 12px;
  flex-shrink: 0;
}

.reference-image-selector__scroll {
  flex: 1;
  min-height: 0;
  padding-right: 4px;
}

.reference-image-selector__load-older {
  flex: 0 0 auto;
  width: 100%;
  margin-bottom: 10px;
}

.reference-image-selector__thumbnail-grid {
  --product-thumbnail-grid-min-size: 110px;
  --product-thumbnail-grid-aspect-ratio: 55 / 77;

  gap: 10px;
}

.reference-image-selector__character-thumbnail-grid {
  --product-thumbnail-grid-min-size: 90px;
  --product-thumbnail-grid-aspect-ratio: 5 / 7;

  gap: 10px;
}

.reference-image-selector__load-more-forms {
  margin-top: 10px;
}

@media (--breakpoint-lg-down) {
  .reference-image-selector__header {
    flex-wrap: wrap;
    gap: 8px;
  }

  .reference-image-selector__batch-actions {
    margin-left: 0;
    order: 3;
    width: 100%;
  }

  .reference-image-selector__thumbnail-grid {
    --product-thumbnail-grid-min-size: 85px;
  }
}
</style>
