<!--
  自定义下拉选择器组件
  替代原生select，解决暗色模式下option颜色问题
-->
<template>
  <div 
    class="custom-select" 
    :class="{ open: isOpen, disabled: disabled }"
    ref="selectRef"
    style="color: #1f2430;"
  >
    <!-- 选择框显示区域 -->
    <div 
      class="custom-select-trigger" 
      @click="toggleDropdown"
      :title="title"
      style="background: #fff; color: #1f2430;"
    >
      <span class="custom-select-value" style="color: #1f2430;">{{ displayValue }}</span>
      <span class="custom-select-arrow" style="color: #666;">
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
import { ref, computed, onMounted, onUnmounted } from 'vue'

// 类型定义
type SelectValue = string | number | boolean

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
}>(), {
  options: () => [],
  groups: () => [],
  placeholder: '请选择',
  disabled: false,
  title: ''
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
    // 打开前计算位置
    updatePosition()
    isOpen.value = true
  } else {
    isOpen.value = false
  }
}

// 更新下拉框位置
function updatePosition() {
  if (selectRef.value) {
    const rect = selectRef.value.getBoundingClientRect()
    dropdownStyle.value = {
      top: `${rect.bottom + 4}px`,
      left: `${rect.left}px`,
      width: `${rect.width}px`,
      minWidth: '160px' // 保持最小宽度
    }
  }
}

// 选择选项
function selectOption(value: string): void {
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

<style>
/* 不使用scoped，直接使用全局样式确保不被覆盖 */
.custom-select {
  position: relative;
  min-width: 160px;
  font-size: 13px !important;
  color: #1f2430 !important;
}

.custom-select-trigger {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 36px;
  padding: 0 10px;
  border: 1px solid #cfd6e4;
  border-radius: 8px;
  background: #ffffff !important;
  color: #1f2430 !important;
  cursor: pointer;
  transition: border-color 0.15s, box-shadow 0.15s;
}

.custom-select-trigger:hover {
  border-color: #8aa0f6;
}

.custom-select.open .custom-select-trigger {
  border-color: #5b73f2;
  box-shadow: 0 0 0 2px rgba(88, 125, 255, 0.18);
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
  color: #1f2430 !important;
}

.custom-select-arrow {
  margin-left: 8px;
  color: #666666 !important;
  transition: transform 0.2s;
}

.custom-select.open .custom-select-arrow {
  transform: rotate(180deg);
}

.custom-select-dropdown {
  position: fixed; /* 改为 fixed 以配合 Teleport */
  /* top, left, width 由 JS 动态计算 */
  margin-top: 0; /* JS计算位置时已包含偏移 */
  background: #ffffff !important;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 20000; /* 确保高于模态框 (10002) */
  max-height: 300px;
  overflow-y: auto;
  color: #1f2430 !important;
}

.custom-select-options {
  padding: 4px 0;
  background: #ffffff !important;
  color: #1f2430 !important;
}

.custom-select-group {
  margin-bottom: 4px;
  background: #ffffff !important;
}

.custom-select-group:last-child {
  margin-bottom: 0;
}

.custom-select-group-label {
  padding: 8px 12px 4px;
  font-size: 11px;
  font-weight: 600;
  color: #666666 !important;
  background: #f5f5f5 !important;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.custom-select-option {
  padding: 8px 12px;
  cursor: pointer;
  color: #1f2430 !important;
  background: #ffffff !important;
  transition: background 0.15s;
}

.custom-select-option:hover {
  background: #e3f2fd !important;
  color: #1f2430 !important;
}

.custom-select-option.selected {
  background: #e8edff !important;
  color: #3040c2 !important;
  font-weight: 500;
}

</style>
