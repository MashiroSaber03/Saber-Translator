<template>
  <BaseModal
    :model-value="visible"
    :show-header="false"
    custom-class="reference-selector-modal"
    @update:model-value="value => { if (!value) handleCancel() }"
  >
    <!-- 标题栏 -->
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

// 选中的参考图 token 列表（按选择顺序）
const selectedTokens = ref<string[]>([])

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
    .filter(img => img.token && img.has_image && img.path)
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
