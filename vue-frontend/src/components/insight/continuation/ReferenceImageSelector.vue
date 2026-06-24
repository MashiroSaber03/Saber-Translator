<template>
  <BaseModal
    :model-value="visible"
    :show-header="false"
    custom-class="reference-selector-modal"
    body-padding="none"
    scroll-mode="contained"
    width="min(1120px, calc(100vw - 48px))"
    max-height="min(88vh, 980px)"
    body-display="flex"
    body-direction="column"
    body-min-height="0"
    @update:model-value="value => { if (!value) handleCancel() }"
  >
    <div class="reference-selector-content">
      <div class="modal-header">
        <h3>选择参考图 ({{ selectedCount }}/{{ maxCount }})</h3>
        <div class="header-actions">
          <UiButton class="reference-selector-modal__button reference-selector-modal__button--secondary" variant="secondary" @click="autoSelectLast">
            自动选择最后{{ maxCount }}张
          </UiButton>
          <UiButton class="reference-selector-modal__button reference-selector-modal__button--secondary" variant="secondary" @click="clearSelection">
            清空
          </UiButton>
        </div>
        <div class="header-right">
          <UiButton class="reference-selector-modal__button reference-selector-modal__button--secondary" variant="secondary" @click="handleCancel">取消</UiButton>
          <UiButton class="reference-selector-modal__button reference-selector-modal__button--primary" variant="primary" @click="handleConfirm">确定</UiButton>
        </div>
        <UiButton variant="toolbar" class="close-btn" @click="handleCancel">&times;</UiButton>
      </div>

      <div v-if="mode === 'image' && characterForms.length > 0" class="character-section">
        <div class="section-label">
          <span>角色档案</span>
          <span class="section-hint">（自动添加，不计入选择数量）</span>
        </div>
        <div class="thumbnails-row">
          <div
            v-for="form in characterForms"
            :key="form.token || `${form.character_name}-${form.form_id}`"
            class="thumbnail character-thumbnail"
          >
            <img
              v-if="form.has_image && form.path"
              :src="getImageUrl(form.path)"
              :alt="`${form.character_name} - ${form.form_name}`"
              loading="lazy"
              @error="handleImageError"
            />
            <div v-else class="placeholder-card">
              <span>角色图缺失</span>
            </div>
            <div class="character-label">{{ form.character_name }} - {{ form.form_name }}</div>
          </div>
        </div>
      </div>

      <div class="manga-section">
        <div class="section-label">
          <span>漫画图片</span>
        </div>
        <div class="thumbnails-grid" ref="thumbnailsGrid">
          <UiButton
            v-for="img in originalImages"
            :key="`original-${img.page_number}`"
            variant="toolbar"
            type="button"
            class="thumbnail"
            :class="{
              selected: isSelected(img),
              disabled: isThumbnailDisabled(img)
            }"
            :aria-label="getThumbnailActionLabel(img, '原作')"
            :aria-pressed="String(isSelected(img))"
            :disabled="isThumbnailDisabled(img)"
            @click="toggleSelection(img)"
          >
            <img
              v-if="img.has_image"
              :src="getOriginalThumbnailUrl(img.page_number)"
              :alt="`第${img.page_number}页`"
              loading="lazy"
              @error="handleImageError"
            />
            <div v-else class="placeholder-card">
              <span>原作页缺失</span>
            </div>
            <div v-if="isSelected(img)" class="selection-badge">
              {{ getSelectionIndex(img) }}
            </div>
            <div class="page-badge">{{ img.page_number }}</div>
            <div
              v-if="isThumbnailDisabled(img)"
              class="disabled-overlay"
              title="已达到最大数量，请先取消其他选择"
            ></div>
          </UiButton>

          <UiButton
            v-for="img in continuationImages"
            :key="`continuation-${img.page_number}`"
            variant="toolbar"
            type="button"
            class="thumbnail continuation-thumbnail"
            :class="{
              selected: isSelected(img),
              disabled: isThumbnailDisabled(img)
            }"
            :aria-label="getThumbnailActionLabel(img, '续写')"
            :aria-pressed="String(isSelected(img))"
            :disabled="isThumbnailDisabled(img)"
            @click="toggleSelection(img)"
          >
            <img
              v-if="img.has_image && img.path"
              :src="getImageUrl(img.path)"
              :alt="`第${img.page_number}页续写图`"
              loading="lazy"
              @error="handleImageError"
            />
            <div v-else class="placeholder-card">
              <span>占位页</span>
            </div>
            <div v-if="isSelected(img)" class="selection-badge">
              {{ getSelectionIndex(img) }}
            </div>
            <div class="page-badge">{{ img.page_number }}</div>
            <div class="continuation-badge">续写</div>
            <div
              v-if="isThumbnailDisabled(img)"
              class="disabled-overlay"
              title="已达到最大数量，请先取消其他选择"
            ></div>
          </UiButton>
        </div>
      </div>
    </div>
  </BaseModal>
</template>

<script setup lang="ts">
import './ReferenceImageSelector.global.styles.css'
import UiButton from '@/components/ui/UiButton.vue'
import BaseModal from '@/components/common/BaseModal.vue'
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
}>()

const emit = defineEmits<{
  'update:visible': [value: boolean]
  'confirm': [selectedTokens: string[]]
  'cancel': []
}>()

const selectedTokens = ref<string[]>([])
const thumbnailsGrid = ref<HTMLElement | null>(null)
const selectedCount = computed(() => selectedTokens.value.length)

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

function toggleSelection(img: MangaImageInfo): void {
  const identifier = getImageIdentifier(img)
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
  if (thumbnailsGrid.value) {
    thumbnailsGrid.value.scrollTop = thumbnailsGrid.value.scrollHeight
  }
}

function getOriginalThumbnailUrl(pageNum: number): string {
  if (!props.bookId) return ''
  return insightApi.getThumbnailUrl(props.bookId, pageNum)
}

function getImageUrl(path: string): string {
  if (!path) return ''
  return `/api/manga-insight/file?path=${encodeURIComponent(path)}`
}

function handleImageError(event: Event): void {
  const img = event.target as HTMLImageElement
  img.style.display = 'none'
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
.reference-selector-content {
  --reference-image-selector-border-active: #d1d5db;
  --reference-image-selector-border-focus: #9ca3af;
  --reference-image-selector-border-hover: #409eff;
  --reference-image-selector-border-muted: #fcd34d;
  --reference-image-selector-border-strong: #cbd5e1;
  --reference-image-selector-border-subtle: #e5e7eb;
  --reference-image-selector-shadow-floating: rgba(64, 158, 255, .2);
  --reference-image-selector-shadow-raised: rgba(64, 158, 255, .25);
  --reference-image-selector-shadow-strong: rgba(0, 0, 0, .25);
  --reference-image-selector-surface-active: rgba(59, 130, 246, .9);
  --reference-image-selector-surface-header: #f8f9fa;
  --reference-image-selector-surface-overlay: rgba(255, 255, 255, .6);
  --reference-image-selector-surface-primary: #409eff;
  --reference-image-selector-surface-primary-strong: #337ecc;
  --reference-image-selector-surface-scrim: rgba(0, 0, 0, .75);
  --reference-image-selector-surface-section: #fef3c7;
  --reference-image-selector-text-character: #92400e;
  --reference-image-selector-text-muted: #b45309;
  --reference-image-selector-text-placeholder: #6b7280;
  --reference-image-selector-text-section: #4b5563;
  --reference-image-selector-text-supporting: #374151;

  display: flex;
  flex: 1;
  min-height: 0;
  flex-direction: column;
}

.modal-header {
  display: flex;
  align-items: center;
  padding: 16px 20px;
  background: var(--reference-image-selector-surface-header);
  border-bottom: 1px solid var(--color-border-default);
  gap: 12px;
  flex-shrink: 0;
}

.modal-header h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  white-space: nowrap;
}

.header-actions {
  display: flex;
  gap: 8px;
  margin-left: 16px;
}

.header-right {
  display: flex;
  gap: 8px;
  margin-left: auto;
}

.close-btn {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
  color: var(--color-text-secondary);
  padding: 0;
  line-height: 1;
  margin-left: 8px;
}

.close-btn:hover {
  color: var(--color-text-default);
}

.placeholder-card {
  width: 100%;
  height: 100%;
  min-height: 132px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, var(--color-surface-muted), var(--color-surface-hover));
  color: var(--reference-image-selector-text-placeholder);
  font-size: 13px;
  font-weight: 600;
  border: 1px dashed var(--reference-image-selector-border-strong);
}

.character-section {
  padding: 12px 20px;
  background: var(--reference-image-selector-surface-section);
  border-bottom: 1px solid var(--reference-image-selector-border-muted);
  flex-shrink: 0;
}

.section-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--reference-image-selector-text-character);
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.section-hint {
  font-weight: 400;
  font-size: 12px;
  color: var(--reference-image-selector-text-muted);
}

.thumbnails-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.manga-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  padding: 16px 20px;
}

.manga-section .section-label {
  color: var(--reference-image-selector-text-section);
  margin-bottom: 12px;
  flex-shrink: 0;
}

.thumbnails-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, 110px);
  gap: 10px;
  overflow-y: auto;
  flex: 1;
  padding-right: 4px;
  justify-content: start;
}

.thumbnail {
  position: relative;
  width: 110px;
  height: 154px;
  border: 2px solid var(--reference-image-selector-border-subtle);
  border-radius: 6px;
  overflow: hidden;
  cursor: pointer;
  background: var(--color-surface-base);
  transition: all 0.15s ease;
  flex-shrink: 0;
}

.thumbnail:hover {
  border-color: var(--reference-image-selector-border-hover);
  box-shadow: 0 2px 12px var(--reference-image-selector-shadow-raised);
  transform: translateY(-2px);
}

.thumbnail.selected {
  border-color: var(--reference-image-selector-border-hover);
  box-shadow: 0 0 0 3px var(--reference-image-selector-shadow-floating);
}

.thumbnail.disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.thumbnail img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.selection-badge {
  position: absolute;
  top: 6px;
  left: 6px;
  width: 26px;
  height: 26px;
  background: var(--reference-image-selector-surface-primary);
  color: var(--color-text-inverse);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: bold;
  box-shadow: 0 2px 6px var(--reference-image-selector-shadow-strong);
  animation: badgePop 0.15s ease;
}

@keyframes badgePop {
  from { transform: scale(0.8); }
  to { transform: scale(1); }
}

.page-badge {
  position: absolute;
  bottom: 4px;
  right: 4px;
  background: var(--reference-image-selector-surface-scrim);
  color: var(--color-text-inverse);
  padding: 3px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
}

.disabled-overlay {
  position: absolute;
  inset: 0;
  background: var(--reference-image-selector-surface-overlay);
  cursor: not-allowed;
}

.continuation-badge {
  position: absolute;
  top: 6px;
  right: 6px;
  background: var(--reference-image-selector-surface-active);
  color: var(--color-text-inverse);
  padding: 2px 6px;
  border-radius: 999px;
  font-size: 10px;
  font-weight: 600;
}

.character-thumbnail {
  width: 90px;
  height: 126px;
  cursor: default;
  flex-shrink: 0;
}

.character-thumbnail:hover {
  border-color: var(--reference-image-selector-border-muted);
  box-shadow: none;
  transform: none;
}

.character-label {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: var(--reference-image-selector-surface-scrim);
  color: var(--color-text-inverse);
  padding: 4px 6px;
  font-size: 10px;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.reference-selector-modal__button {
  padding: 7px 14px;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
}

.reference-selector-modal__button--primary {
  background: var(--reference-image-selector-surface-primary);
  color: var(--color-text-inverse);
}

.reference-selector-modal__button--primary:hover {
  background: var(--reference-image-selector-surface-primary-strong);
}

.reference-selector-modal__button--secondary {
  background: var(--color-surface-base);
  color: var(--reference-image-selector-text-supporting);
  border: 1px solid var(--reference-image-selector-border-active);
}

.reference-selector-modal__button--secondary:hover {
  background: var(--color-surface-muted);
  border-color: var(--reference-image-selector-border-focus);
}

@media (--breakpoint-lg-down) {
  .modal-header {
    flex-wrap: wrap;
    gap: 8px;
  }

  .header-actions {
    margin-left: 0;
    order: 3;
    width: 100%;
  }

  .thumbnails-grid {
    grid-template-columns: repeat(auto-fill, minmax(85px, 1fr));
  }
}
</style>
