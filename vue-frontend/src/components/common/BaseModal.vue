<template>
  <Teleport to="body">
    <Transition name="modal-fade">
      <div v-if="modelValue" class="modal-overlay" @click.self="handleOverlayClick">
        <div
          ref="modalContainerRef"
          class="modal-container"
          :class="[sizeClass, customClass]"
          :style="customStyle"
          role="dialog"
          aria-modal="true"
          :aria-labelledby="resolvedAriaLabelledby"
          :aria-describedby="ariaDescribedby"
          tabindex="-1"
        >
          <!-- 模态框头部 -->
          <div v-if="showHeader" class="modal-header">
            <h3 :id="internalTitleId" class="modal-title">
              <slot name="title">{{ title }}</slot>
            </h3>
            <button
              v-if="showCloseButton"
              type="button"
              class="modal-close-btn"
              title="关闭"
              aria-label="关闭对话框"
              @click="close"
            >
              ✕
            </button>
          </div>

          <!-- 模态框内容 -->
          <div class="modal-body">
            <slot></slot>
          </div>

          <!-- 模态框底部 -->
          <div v-if="$slots.footer" class="modal-footer">
            <slot name="footer"></slot>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, watch, onMounted, onUnmounted, nextTick, ref, getCurrentInstance } from 'vue'

// Props 定义
interface Props {
  /** 控制模态框显示/隐藏（可选，默认为 true） */
  modelValue?: boolean
  /** 模态框标题 */
  title?: string
  /** 是否显示头部 */
  showHeader?: boolean
  /** 是否显示关闭按钮 */
  showCloseButton?: boolean
  /** 点击遮罩层是否关闭 */
  closeOnOverlay?: boolean
  /** 按 ESC 键是否关闭 */
  closeOnEsc?: boolean
  /** 模态框尺寸 */
  size?: 'small' | 'medium' | 'large' | 'full'
  /** 自定义类名 */
  customClass?: string
  /** 自定义样式 */
  customStyle?: Record<string, string>
  /** 可选：显式指定 aria-labelledby */
  ariaLabelledby?: string
  /** 可选：显式指定 aria-describedby */
  ariaDescribedby?: string
  /** 可选：打开后优先聚焦的选择器（在模态框内部查询） */
  initialFocusSelector?: string
}

const props = withDefaults(defineProps<Props>(), {
  modelValue: true,  // 默认显示，当组件被渲染时
  title: '',
  showHeader: true,
  showCloseButton: true,
  closeOnOverlay: true,
  closeOnEsc: true,
  size: 'medium',
  customClass: '',
  customStyle: () => ({}),
  ariaLabelledby: '',
  ariaDescribedby: '',
  initialFocusSelector: ''
})

// Emits 定义
const emit = defineEmits<{
  /** 更新显示状态 */
  'update:modelValue': [value: boolean]
  /** 关闭事件 */
  close: []
  /** 打开事件 */
  open: []
}>()

// 计算尺寸类名
const sizeClass = computed(() => {
  return `modal-${props.size}`
})

const modalContainerRef = ref<HTMLElement | null>(null)
const lastActiveElement = ref<HTMLElement | null>(null)
const instanceUid = getCurrentInstance()?.uid ?? Math.floor(Math.random() * 100000)
const internalTitleId = `base-modal-title-${instanceUid}`
const ariaDescribedby = computed(() => props.ariaDescribedby || undefined)
const resolvedAriaLabelledby = computed(() => {
  if (props.ariaLabelledby) return props.ariaLabelledby
  return props.showHeader ? internalTitleId : undefined
})

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'area[href]',
  'input:not([disabled]):not([type="hidden"])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  'button:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
  '[contenteditable="true"]'
].join(',')

function getFocusableElements(): HTMLElement[] {
  if (!modalContainerRef.value) return []
  return Array.from(modalContainerRef.value.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR))
    .filter(el => !el.hasAttribute('disabled') && el.getAttribute('aria-hidden') !== 'true')
}

function focusFirstElement(): void {
  const container = modalContainerRef.value
  if (!container) return

  if (props.initialFocusSelector) {
    const preferred = container.querySelector<HTMLElement>(props.initialFocusSelector)
    if (preferred) {
      preferred.focus()
      return
    }
  }

  const focusables = getFocusableElements()
  if (focusables.length > 0) {
    focusables[0]?.focus()
  } else {
    container.focus()
  }
}

function restoreFocus(): void {
  const previous = lastActiveElement.value
  if (previous && typeof previous.focus === 'function') {
    previous.focus()
  }
}

function trapFocus(event: KeyboardEvent): void {
  const focusables = getFocusableElements()
  const container = modalContainerRef.value
  if (!container) return

  if (focusables.length === 0) {
    event.preventDefault()
    container.focus()
    return
  }

  const first = focusables[0]
  const last = focusables[focusables.length - 1]
  const active = document.activeElement as HTMLElement | null

  if (event.shiftKey) {
    if (active === first || active === container) {
      event.preventDefault()
      last?.focus()
    }
  } else if (active === last) {
    event.preventDefault()
    first?.focus()
  }
}

// 关闭模态框
const close = () => {
  emit('update:modelValue', false)
  emit('close')
}

// 处理遮罩层点击
const handleOverlayClick = () => {
  if (props.closeOnOverlay) {
    close()
  }
}

// 处理键盘事件
const handleKeydown = (event: KeyboardEvent) => {
  if (!props.modelValue) return

  if (event.key === 'Escape' && props.closeOnEsc && props.modelValue) {
    close()
    return
  }
  if (event.key === 'Tab') {
    trapFocus(event)
  }
}

// 监听显示状态变化
watch(
  () => props.modelValue,
  (newValue) => {
    if (newValue) {
      lastActiveElement.value = document.activeElement as HTMLElement | null
      emit('open')
      // 打开时禁止背景滚动
      document.body.style.overflow = 'hidden'
      nextTick(() => {
        focusFirstElement()
      })
    } else {
      // 关闭时恢复背景滚动
      document.body.style.overflow = ''
      restoreFocus()
    }
  },
  { immediate: true }
)

// 生命周期
onMounted(() => {
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
  // 确保恢复背景滚动
  document.body.style.overflow = ''
  restoreFocus()
})
</script>

<style scoped>
/* 遮罩层 */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgb(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: var(--z-overlay);
}

/* 模态框容器 */
.modal-container {
  background: var(--modal-bg, #fff);
  border-radius: 12px;
  box-shadow: 0 4px 20px rgb(0, 0, 0, 0.15);
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* 尺寸变体 */
.modal-small {
  width: 400px;
  max-width: 90vw;
}

.modal-medium {
  width: 600px;
  max-width: 90vw;
}

.modal-large {
  width: 900px;
  max-width: 95vw;
}

.modal-full {
  width: 95vw;
  height: 90vh;
}

/* 模态框头部 */
.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-color, #e0e0e0);
}

.modal-title {
  margin: 0;
  font-size: 1.2em;
  font-weight: 600;
  color: var(--text-color, #2c3e50);
}

.modal-close-btn {
  background: none;
  border: none;
  font-size: 1.2em;
  cursor: pointer;
  color: var(--text-secondary, #666);
  padding: 4px 8px;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.modal-close-btn:hover {
  background-color: rgb(0, 0, 0, 0.1);
  color: var(--text-color, #2c3e50);
}

/* 模态框内容 */
.modal-body {
  padding: 20px;
  overflow-y: auto;
  flex: 1;
}

/* 模态框底部 */
.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 16px 20px;
  border-top: 1px solid var(--border-color, #e0e0e0);
}

/* 过渡动画 */
.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.2s ease;
}

.modal-fade-enter-active .modal-container,
.modal-fade-leave-active .modal-container {
  transition: transform 0.2s ease;
}

.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}

.modal-fade-enter-from .modal-container,
.modal-fade-leave-to .modal-container {
  transform: scale(0.95);
}
</style>
