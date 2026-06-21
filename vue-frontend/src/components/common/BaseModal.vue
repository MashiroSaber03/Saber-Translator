<template>
  <Teleport to="body">
    <Transition name="modal-fade">
      <div
        v-if="modelValue"
        ref="overlayRef"
        class="ui-modal__overlay"
        :class="overlayClass"
        @mousedown.self="handleOverlayMouseDown"
      >
        <div
          class="ui-modal__container"
          :class="[uiSizeClass, uiChromeClass, customClass]"
          :style="customStyle"
        >
          <!-- 模态框头部 -->
          <div v-if="showHeader" class="ui-modal__header">
            <h3 class="ui-modal__title">
              <slot name="title">{{ title }}</slot>
            </h3>
            <UiButton
              variant="toolbar"
              v-if="showCloseButton"
              class="ui-modal__close"
              title="关闭"
              @click="close"
            >
              ✕
            </UiButton>
          </div>

          <!-- 模态框内容 -->
          <div class="ui-modal__body" :class="[uiBodyPaddingClass, uiBodyScrollClass, bodyClass]">
            <slot></slot>
          </div>

          <!-- 模态框底部 -->
          <div v-if="$slots.footer" class="ui-modal__footer" :class="footerClass">
            <slot name="footer"></slot>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import { computed, watch, onMounted, onUnmounted } from 'vue'
import { useOverlayDismiss } from '@/composables/useOverlayDismiss'

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
  /** 遮罩层自定义类名，仅用于明确的 Teleport 布局定制 */
  overlayClass?: string
  /** 内容区自定义类名 */
  bodyClass?: string
  /** 底部自定义类名 */
  footerClass?: string
  /** 内容区 padding 策略 */
  bodyPadding?: 'default' | 'none' | 'compact'
  /** 内容区滚动策略 */
  scrollMode?: 'auto' | 'contained' | 'none'
  /** 弹窗 chrome 视觉变体 */
  chromeVariant?: 'default' | 'compact' | 'plain'
  /** 自定义样式 */
  customStyle?: Record<string, string>
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
  overlayClass: '',
  bodyClass: '',
  footerClass: '',
  bodyPadding: 'default',
  scrollMode: 'auto',
  chromeVariant: 'default',
  customStyle: () => ({})
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

const uiSizeClass = computed(() => {
  return `ui-modal__container--${props.size}`
})

const uiChromeClass = computed(() => {
  return `ui-modal__container--chrome-${props.chromeVariant}`
})

const uiBodyPaddingClass = computed(() => {
  return `ui-modal__body--padding-${props.bodyPadding}`
})

const uiBodyScrollClass = computed(() => {
  return `ui-modal__body--scroll-${props.scrollMode}`
})

// 关闭模态框
const close = () => {
  emit('update:modelValue', false)
  emit('close')
}

const { overlayRef, handleOverlayMouseDown, resetOverlayDismissState } = useOverlayDismiss(close, {
  enabled: () => props.closeOnOverlay && props.modelValue,
})

// 处理键盘事件
const handleKeydown = (event: KeyboardEvent) => {
  if (event.key === 'Escape' && props.closeOnEsc && props.modelValue) {
    close()
  }
}

// 监听显示状态变化
watch(
  () => props.modelValue,
  (newValue) => {
    if (newValue) {
      emit('open')
      // 打开时禁止背景滚动
      document.body.style.overflow = 'hidden'
    } else {
      resetOverlayDismissState()
      // 关闭时恢复背景滚动
      document.body.style.overflow = ''
    }
  }
)

// 生命周期
onMounted(() => {
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
  // 确保恢复背景滚动
  document.body.style.overflow = ''
})
</script>

<style scoped>
/* 遮罩层 */
.ui-modal__overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: var(--base-modal-surface-base);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: var(--z-overlay);
}

/* 模态框容器 */
.ui-modal__container {
  background: var(--modal-bg, var(--color-surface-base));
  border-radius: 12px;
  box-shadow: 0 4px 20px var(--base-modal-shadow-default);
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* 尺寸变体 */
.ui-modal__container--small {
  width: 400px;
  max-width: 90vw;
}

.ui-modal__container--medium {
  width: 600px;
  max-width: 90vw;
}

.ui-modal__container--large {
  width: 900px;
  max-width: 95vw;
}

.ui-modal__container--full {
  width: 95vw;
  height: 90vh;
}

/* 模态框头部 */
.ui-modal__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid var(--color-border-muted, var(--color-border-default));
}

.ui-modal__title {
  margin: 0;
  font-size: 1.2em;
  font-weight: 600;
  color: var(--color-text-strong, var(--color-text-heading));
}

.ui-modal__close {
  background: none;
  border: none;
  font-size: 1.2em;
  cursor: pointer;
  color: var(--color-text-supporting, var(--color-text-secondary));
  padding: 4px 8px;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.ui-modal__close:hover {
  background-color: var(--base-modal-surface-raised);
  color: var(--color-text-strong, var(--color-text-heading));
}

/* 模态框内容 */
.ui-modal__body {
  padding: 20px;
  overflow-y: auto;
  flex: 1;
}

.ui-modal__body--padding-none {
  padding: 0;
}

.ui-modal__body--padding-compact {
  padding: 12px;
}

.ui-modal__body--scroll-contained {
  overflow: hidden;
}

.ui-modal__body--scroll-none {
  overflow: visible;
}

/* 模态框底部 */
.ui-modal__footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 16px 20px;
  border-top: 1px solid var(--color-border-muted, var(--color-border-default));
}

.ui-modal__container--chrome-compact .ui-modal__header {
  padding: 12px 16px;
}

.ui-modal__container--chrome-compact .ui-modal__footer {
  padding: 12px 16px;
}

.ui-modal__container--chrome-plain {
  box-shadow: none;
}

/* 过渡动画 */
.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.2s ease;
}

.modal-fade-enter-active .ui-modal__container,
.modal-fade-leave-active .ui-modal__container {
  transition: transform 0.2s ease;
}

.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}

.modal-fade-enter-from .ui-modal__container,
.modal-fade-leave-to .ui-modal__container {
  transform: scale(0.95);
}
</style>
