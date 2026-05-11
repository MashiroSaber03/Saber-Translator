<template>
  <div
    v-if="visible"
    ref="overlayRef"
    class="reference-selector-overlay"
    @mousedown.self="handleOverlayMouseDown"
  >
    <div class="reference-selector-modal">
      <!-- 标题栏 -->
      <div class="modal-header">
        <h3>选择参考图 ({{ selectedCount }}/{{ maxCount }})</h3>
        <div class="header-actions">
          <button class="btn secondary" @click="autoSelectLast">
            自动选择最后{{ maxCount }}张
          </button>
          <button class="btn secondary" @click="clearSelection">
            清空
          </button>
        </div>
        <div class="header-right">
          <button class="btn secondary" @click="handleCancel">取消</button>
          <button class="btn primary" @click="handleConfirm">确定</button>
        </div>
        <button class="close-btn" @click="handleCancel">&times;</button>
      </div>

      <!-- 角色档案区域（仅生图场景显示） -->
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

      <!-- 漫画图片区域 -->
      <div class="manga-section">
        <div class="section-label">
          <span>漫画图片</span>
        </div>
        <div class="thumbnails-grid" ref="thumbnailsGrid">
          <div
            v-for="img in originalImages"
            :key="`original-${img.page_number}`"
            class="thumbnail"
            :class="{
              selected: isSelected(img),
              disabled: !isSelected(img) && selectedCount >= maxCount
            }"
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
            <!-- 选中标记 -->
            <div v-if="isSelected(img)" class="selection-badge">
              {{ getSelectionIndex(img) }}
            </div>
            <!-- 页码徽章 -->
            <div class="page-badge">{{ img.page_number }}</div>
            <!-- 禁用遮罩 -->
            <div
              v-if="!isSelected(img) && selectedCount >= maxCount"
              class="disabled-overlay"
              title="已达到最大数量，请先取消其他选择"
            ></div>
          </div>

          <div
            v-for="img in continuationImages"
            :key="`continuation-${img.page_number}`"
            class="thumbnail continuation-thumbnail"
            :class="{
              selected: isSelected(img),
              disabled: !isSelected(img) && selectedCount >= maxCount
            }"
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
              v-if="!isSelected(img) && selectedCount >= maxCount"
              class="disabled-overlay"
              title="已达到最大数量，请先取消其他选择"
            ></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'
import type { MangaImageInfo, CharacterFormInfo } from '@/api/continuation'
import * as insightApi from '@/api/insight'
import { useOverlayDismiss } from '@/composables/useOverlayDismiss'

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

// 选中的参考图 token 列表（按选择顺序）
const selectedTokens = ref<string[]>([])
const { overlayRef, handleOverlayMouseDown } = useOverlayDismiss(handleCancel, {
  enabled: () => props.visible,
})

// 缩略图网格引用
const thumbnailsGrid = ref<HTMLElement | null>(null)

// 计算选中数量
const selectedCount = computed(() => selectedTokens.value.length)

// 监听可见性变化，初始化选择状态
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

    // 恢复之前的选择状态，或自动预选最后N张
    if (props.initialSelection && props.initialSelection.length > 0) {
      selectedTokens.value = props.initialSelection.filter(token => availableTokens.has(token))
      if (selectedTokens.value.length === 0) {
        autoSelectLast()
      }
    } else {
      // 自动预选最后N张
      autoSelectLast()
    }
    // 滚动到底部
    nextTick(() => {
      scrollToBottom()
    })
  }
}, { immediate: true })

// 获取图片的唯一标识符（使用 token）
function getImageIdentifier(img: MangaImageInfo): string {
  return img.token || ''
}

// 检查图片是否被选中
function isSelected(img: MangaImageInfo): boolean {
  const identifier = getImageIdentifier(img)
  return identifier ? selectedTokens.value.includes(identifier) : false
}

// 获取选中序号
function getSelectionIndex(img: MangaImageInfo): number {
  const identifier = getImageIdentifier(img)
  const index = selectedTokens.value.indexOf(identifier)
  return index >= 0 ? index + 1 : 0
}

// 切换选择状态
function toggleSelection(img: MangaImageInfo): void {
  const identifier = getImageIdentifier(img)
  if (!identifier) return

  const index = selectedTokens.value.indexOf(identifier)
  if (index >= 0) {
    // 取消选择
    selectedTokens.value.splice(index, 1)
  } else {
    // 添加选择（检查是否达到上限）
    if (selectedTokens.value.length < props.maxCount) {
      selectedTokens.value.push(identifier)
    }
  }
}

// 自动选择最后N张
function autoSelectLast(): void {
  selectedTokens.value = []

  const validImages = [
    ...props.originalImages,
    ...(props.mode === 'image' ? props.continuationImages : []),
  ]
    .filter(img => img.token)
    .sort((left, right) => left.page_number - right.page_number)

  // 取最后N张
  const lastN = validImages.slice(-props.maxCount)
  selectedTokens.value = lastN.map(img => img.token)

  // 滚动到底部
  nextTick(() => {
    scrollToBottom()
  })
}

// 清空选择
function clearSelection(): void {
  selectedTokens.value = []
}

// 滚动到底部
function scrollToBottom(): void {
  if (thumbnailsGrid.value) {
    thumbnailsGrid.value.scrollTop = thumbnailsGrid.value.scrollHeight
  }
}

// 获取原作图片缩略图URL（使用缩略图接口，性能更好）
function getOriginalThumbnailUrl(pageNum: number): string {
  if (!props.bookId) return ''
  return insightApi.getThumbnailUrl(props.bookId, pageNum)
}

// 获取其他图片URL（角色档案等）
function getImageUrl(path: string): string {
  if (!path) return ''
  // 通过后端文件服务接口获取图片
  return `/api/manga-insight/file?path=${encodeURIComponent(path)}`
}

// 图片加载失败处理
function handleImageError(event: Event): void {
  const img = event.target as HTMLImageElement
  img.style.display = 'none'
}

// 确认选择
function handleConfirm(): void {
  emit('confirm', [...selectedTokens.value])
  emit('update:visible', false)
}

// 取消选择
function handleCancel(): void {
  emit('cancel')
  emit('update:visible', false)
}
</script>

<style scoped>
.reference-selector-overlay {
  position: fixed;
  inset: 0;
  background: rgb(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: var(--z-overlay);
  animation: fadeIn 0.2s ease;
}

.reference-selector-modal {
  width: min(1120px, calc(100vw - 48px));
  max-height: min(88vh, 980px);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: #fff;
  border: 1px solid #dbe3ef;
  border-radius: 18px;
  box-shadow: 0 24px 64px rgb(15, 23, 42, 0.28);
  animation: scaleIn 0.2s ease;
}

@keyframes scaleIn {
  from { transform: scale(0.9); opacity: 0; }
  to { transform: scale(1); opacity: 1; }
}

.modal-header {
  display: flex;
  align-items: center;
  padding: 16px 20px;
  background: #f8f9fa;
  border-bottom: 1px solid #e0e0e0;
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
  color: #666;
  padding: 0;
  line-height: 1;
  margin-left: 8px;
}

.close-btn:hover {
  color: #333;
}

.placeholder-card {
  width: 100%;
  height: 100%;
  min-height: 132px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #f3f4f6, #e5e7eb);
  color: #6b7280;
  font-size: 13px;
  font-weight: 600;
  border: 1px dashed #cbd5e1;
}

/* 角色档案区域 */
.character-section {
  padding: 12px 20px;
  background: #fef3c7;
  border-bottom: 1px solid #fcd34d;
  flex-shrink: 0;
}

.section-label {
  font-size: 13px;
  font-weight: 600;
  color: #92400e;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.section-hint {
  font-weight: 400;
  font-size: 12px;
  color: #b45309;
}

.thumbnails-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

/* 漫画图片区域 */
.manga-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  padding: 16px 20px;
}

.manga-section .section-label {
  color: #4b5563;
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
  border: 2px solid #e5e7eb;
  border-radius: 6px;
  overflow: hidden;
  cursor: pointer;
  background: white;
  transition: all 0.15s ease;
  flex-shrink: 0;
}

.thumbnail:hover {
  border-color: #409eff;
  box-shadow: 0 2px 12px rgb(64, 158, 255, 0.25);
  transform: translateY(-2px);
}

.thumbnail.selected {
  border-color: #409eff;
  box-shadow: 0 0 0 3px rgb(64, 158, 255, 0.2);
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
  background: #409eff;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: bold;
  box-shadow: 0 2px 6px rgb(0, 0, 0, 0.25);
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
  background: rgb(0, 0, 0, 0.75);
  color: white;
  padding: 3px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 600;
}

.disabled-overlay {
  position: absolute;
  inset: 0;
  background: rgb(255, 255, 255, 0.6);
  cursor: not-allowed;
}

.continuation-badge {
  position: absolute;
  top: 6px;
  right: 6px;
  background: rgb(59, 130, 246, 0.9);
  color: white;
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
  border-color: #fcd34d;
  box-shadow: none;
  transform: none;
}

.character-label {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgb(0, 0, 0, 0.75);
  color: white;
  padding: 4px 6px;
  font-size: 10px;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.btn {
  padding: 7px 14px;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
}

.btn.primary {
  background: #409eff;
  color: white;
}

.btn.primary:hover {
  background: #337ecc;
}

.btn.secondary {
  background: white;
  color: #374151;
  border: 1px solid #d1d5db;
}

.btn.secondary:hover {
  background: #f3f4f6;
  border-color: #9ca3af;
}

/* 响应式适配 */
@media (width <= 900px) {
  .reference-selector-modal {
    width: 95vw;
    max-height: 90vh;
  }

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
