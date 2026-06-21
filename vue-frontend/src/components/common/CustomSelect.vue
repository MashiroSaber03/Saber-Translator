<!--
  自定义下拉选择器组件
  替代原生select，提供更好的样式控制
-->
<template>
  <div 
    class="custom-select" 
    :class="[
      `custom-select--${variant}`,
      { open: isOpen, disabled: disabled, 'custom-select--fit': fit }
    ]"
    ref="selectRef"
  >
    <!-- 选择框显示区域 -->
    <div 
      class="custom-select-trigger" 
      @click="toggleDropdown"
      :title="title"
    >
      <span class="custom-select-value">{{ displayValue }}</span>
      <span class="custom-select-arrow">
        <svg viewBox="0 0 12 12" width="12" height="12">
          <path d="M2 4l4 4 4-4" stroke="currentColor" stroke-width="1.5" fill="none" />
        </svg>
      </span>
    </div>
    
    <!-- 下拉选项列表 -->
    <Teleport to="body">
      <div 
        v-if="isOpen"
        ref="dropdownRef" 
        class="custom-select-dropdown"
        :style="dropdownStyle"
      >
        <div class="custom-select-options">
          <template v-if="hasGroups">
            <div 
              v-for="group in groupedOptions" 
              :key="group.label" 
              class="custom-select-group"
            >
              <div class="custom-select-group-label">{{ group.label }}</div>
              <div
                v-for="option in group.options"
                :key="option.value"
                class="custom-select-option"
                :class="{ selected: option.value === modelValue }"
                @click="selectOption(option.value)"
              >
                {{ option.label }}
              </div>
            </div>
          </template>
          <template v-else>
            <div
              v-for="option in flatOptions"
              :key="option.value"
              class="custom-select-option"
              :class="{ selected: option.value === modelValue }"
              @click="selectOption(option.value)"
            >
              {{ option.label }}
            </div>
          </template>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'

// 类型定义
// 注意：Vue 的 :key 需要 PropertyKey (string | number | symbol)，boolean 不兼容
type SelectValue = string | number

interface SelectOption {
  label: string
  value: SelectValue
}

interface SelectGroup {
  label: string
  options: SelectOption[]
}

// Props
const props = withDefaults(defineProps<{
  /** 当前选中的值 */
  modelValue: SelectValue
  /** 选项数组 (平铺模式) */
  options?: SelectOption[]
  /** 分组选项数组 (分组模式) */
  groups?: SelectGroup[]
  /** 占位文本 */
  placeholder?: string
  /** 是否禁用 */
  disabled?: boolean
  /** 标题提示 */
  title?: string
  /** 视觉尺寸/场景变体 */
  variant?: 'default' | 'compact' | 'workflow'
  /** 是否跟随父容器宽度 */
  fit?: boolean
}>(), {
  options: () => [],
  groups: () => [],
  placeholder: '请选择',
  disabled: false,
  title: '',
  variant: 'default',
  fit: false
})

// Emits
const emit = defineEmits<{
  (e: 'update:modelValue', value: SelectValue): void
  (e: 'change', value: SelectValue): void
}>()

// 状态
const isOpen = ref(false)
const selectRef = ref<HTMLElement | null>(null)
const dropdownRef = ref<HTMLElement | null>(null)
const dropdownStyle = ref<Record<string, string>>({})

const VIEWPORT_PADDING = 12
const DROPDOWN_GAP = 6
const MAX_DROPDOWN_HEIGHT = 360

// 是否使用分组模式
const hasGroups = computed(() => props.groups && props.groups.length > 0)

// 分组选项
const groupedOptions = computed(() => props.groups)

// 平铺选项
const flatOptions = computed(() => props.options)

// 获取所有选项（用于查找当前选中项的标签）
const allOptions = computed(() => {
  if (hasGroups.value) {
    return props.groups.flatMap(g => g.options)
  }
  return props.options
})

// 当前显示的值
const displayValue = computed(() => {
  const option = allOptions.value.find(o => o.value === props.modelValue)
  return option ? option.label : props.placeholder
})

// 切换下拉框
function toggleDropdown(): void {
  if (props.disabled) return
  
  if (!isOpen.value) {
    isOpen.value = true
    // 先渲染下拉，再根据视口和内容动态定位
    nextTick(() => {
      updatePosition()
      requestAnimationFrame(() => updatePosition())
    })
  } else {
    isOpen.value = false
  }
}

function getOptionCount(): number {
  if (hasGroups.value) {
    return props.groups.reduce((count, group) => count + group.options.length + 1, 0)
  }
  return props.options.length
}

// 更新下拉框位置
function updatePosition() {
  if (!selectRef.value) return

  const rect = selectRef.value.getBoundingClientRect()
  const viewportWidth = window.innerWidth
  const viewportHeight = window.innerHeight
  const fallbackHeight = Math.min(MAX_DROPDOWN_HEIGHT, Math.max(44, getOptionCount() * 40))
  const renderedHeight = dropdownRef.value?.scrollHeight ?? fallbackHeight
  const desiredHeight = Math.min(MAX_DROPDOWN_HEIGHT, Math.max(44, renderedHeight))

  const spaceBelow = viewportHeight - rect.bottom - VIEWPORT_PADDING
  const spaceAbove = rect.top - VIEWPORT_PADDING
  const shouldOpenAbove = spaceBelow < Math.min(desiredHeight, 220) && spaceAbove > spaceBelow

  const availableHeight = shouldOpenAbove ? spaceAbove : spaceBelow
  const maxHeight = Math.min(desiredHeight, Math.max(availableHeight - DROPDOWN_GAP, 44))
  const width = Math.min(rect.width, viewportWidth - VIEWPORT_PADDING * 2)
  const left = Math.min(
    Math.max(rect.left, VIEWPORT_PADDING),
    viewportWidth - VIEWPORT_PADDING - width
  )

  const rawTop = shouldOpenAbove
    ? rect.top - maxHeight - DROPDOWN_GAP
    : rect.bottom + DROPDOWN_GAP
  const top = Math.min(
    Math.max(rawTop, VIEWPORT_PADDING),
    viewportHeight - VIEWPORT_PADDING - maxHeight
  )

  dropdownStyle.value = {
    top: `${Math.round(top)}px`,
    left: `${Math.round(left)}px`,
    width: `${Math.round(width)}px`,
    minWidth: '160px',
    maxHeight: `${Math.round(maxHeight)}px`
  }
}

// 选择选项
function selectOption(value: SelectValue): void {
  emit('update:modelValue', value)
  emit('change', value)
  isOpen.value = false
}

// 点击外部关闭
function handleClickOutside(event: MouseEvent): void {
  // 检查点击是否在触发器上
  if (selectRef.value && selectRef.value.contains(event.target as Node)) {
    return
  }
  
  // 检查点击是否在下拉菜单内部
  if (dropdownRef.value && dropdownRef.value.contains(event.target as Node)) {
    return
  }

  isOpen.value = false
}

// 监听页面滚动和调整大小，更新位置或关闭
function handleScrollOrResize() {
  if (isOpen.value) {
    // 简单起见，滚动时更新位置
    updatePosition()
  }
}

// 生命周期
onMounted(() => {
  document.addEventListener('click', handleClickOutside)
  window.addEventListener('scroll', handleScrollOrResize, true) // 捕获模式以监听所有滚动
  window.addEventListener('resize', handleScrollOrResize)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
  window.removeEventListener('scroll', handleScrollOrResize, true)
  window.removeEventListener('resize', handleScrollOrResize)
})
</script>

<style scoped>
.custom-select {
  position: relative;
  min-width: 160px;
  font-size: 14px;
  color: var(--custom-select-text-primary);
}

.custom-select--fit {
  width: 100%;
  min-width: 0;
}

.custom-select-trigger {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 40px;
  padding: 0 12px;
  border: 1px solid var(--custom-select-border-default);
  border-radius: 8px;
  background: var(--color-surface-plain);
  color: var(--custom-select-text-primary);
  cursor: pointer;
  transition: border-color 0.15s, box-shadow 0.15s;
}

.custom-select--compact {
  min-width: 70px;
  flex: 0 0 auto;
}

.custom-select--compact .custom-select-trigger {
  height: 38px;
  padding: 0 10px;
}

.custom-select--workflow .custom-select-trigger {
  min-height: 42px;
  border-color: var(--custom-select-border-strong);
  border-radius: 10px;
}

.custom-select-trigger:hover {
  border-color: var(--custom-select-border-muted);
}

.custom-select.open .custom-select-trigger {
  border-color: var(--custom-select-border-subtle);
  box-shadow: 0 0 0 2px var(--custom-select-shadow-default);
}

.custom-select.disabled .custom-select-trigger {
  opacity: 0.6;
  cursor: not-allowed;
}

.custom-select-value {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--custom-select-text-primary);
}

.custom-select-arrow {
  margin-left: 8px;
  color: var(--color-text-secondary);
  transition: transform 0.2s;
}

.custom-select.open .custom-select-arrow {
  transform: rotate(180deg);
}

.custom-select-dropdown {
  position: fixed; /* 改为 fixed 以配合 Teleport */
  /* top, left, width 由 JS 动态计算 */
  margin-top: 0; /* JS计算位置时已包含偏移 */
  background: var(--color-surface-plain);
  border: 1px solid var(--color-border-default);
  border-radius: 10px;
  box-shadow: 0 12px 26px var(--custom-select-shadow-raised);
  z-index: var(--z-popover);
  max-height: 360px;
  overflow-y: auto;
  overscroll-behavior: contain;
  color: var(--custom-select-text-primary);
}

.custom-select-options {
  padding: 6px 0;
  background: var(--color-surface-plain);
  color: var(--custom-select-text-primary);
}

.custom-select-group {
  margin-bottom: 4px;
  background: var(--color-surface-plain);
}

.custom-select-group:last-child {
  margin-bottom: 0;
}

.custom-select-group-label {
  padding: 8px 12px 4px;
  font-size: 11px;
  font-weight: 600;
  color: var(--color-text-secondary);
  background: var(--color-surface-subtle);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.custom-select-option {
  padding: 9px 12px;
  cursor: pointer;
  color: var(--custom-select-text-primary);
  background: var(--color-surface-plain);
  font-size: 14px;
  line-height: 1.4;
  transition: background 0.15s;
}

.custom-select-option:hover {
  background: var(--custom-select-surface-base);
  color: var(--custom-select-text-primary);
}

.custom-select-option.selected {
  background: var(--custom-select-surface-raised);
  color: var(--custom-select-text-secondary);
  font-weight: 500;
}

</style>
