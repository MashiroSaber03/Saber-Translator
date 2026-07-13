<template>
  <Teleport to="body">
    <Transition name="modal-fade">
      <div
        v-if="modelValue"
        ref="overlayRef"
        class="ui-modal__overlay"
        :class="[uiPlacementClass, uiBackdropClass, uiOverlayLayerClass, uiBackdropEffectClass, overlayClass]"
        data-testid="base-dialog-overlay"
        @mousedown.self="handleOverlayMouseDown"
      >
        <div
          ref="dialogRef"
          class="ui-modal__container"
          :class="[uiSizeClass, uiChromeClass, uiFrameClass, uiMobilePresentationClass, customClass]"
          :style="dialogStyle"
          role="dialog"
          tabindex="-1"
          aria-modal="true"
          :aria-labelledby="showHeader ? titleId : undefined"
          :aria-label="!showHeader && title ? title : undefined"
          data-testid="base-dialog-container"
        >
          <div
            v-if="showHeader"
            class="ui-modal__header"
            :class="[uiHeaderVariantClass, uiHeaderDividerClass]"
          >
            <h3 :id="titleId" class="ui-modal__title">
              <slot name="title">{{ title }}</slot>
            </h3>
            <UiIconButton
              v-if="showCloseButton"
              class="ui-modal__close"
              label="关闭"
              title="关闭"
              variant="plain"
              size="sm"
              shape="circle"
              data-testid="base-dialog-close"
              @click="close"
            >
              <UiIcon name="x" size="16" />
            </UiIconButton>
          </div>

          <div
            class="ui-modal__body"
            :class="[uiBodyPaddingClass, uiBodyScrollClass, bodyClass]"
            data-testid="base-dialog-body"
          >
            <slot></slot>
          </div>

          <div
            v-if="$slots.footer"
            class="ui-modal__footer"
            :class="[uiFooterDividerClass, uiFooterToneClass, footerClass]"
            data-testid="base-dialog-footer"
          >
            <slot name="footer"></slot>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script lang="ts">
let openModalCount = 0
let previousBodyOverflow: string | null = null

function lockBodyScroll() {
  if (openModalCount === 0) {
    previousBodyOverflow = document.body.style.overflow
  }
  openModalCount += 1
  document.body.style.overflow = 'hidden'
}

function unlockBodyScroll() {
  if (openModalCount === 0) return
  openModalCount -= 1
  if (openModalCount === 0) {
    document.body.style.overflow = previousBodyOverflow ?? ''
    previousBodyOverflow = null
  }
}
</script>

<script setup lang="ts">
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import { computed, ref, toRef, watch, onMounted, onUnmounted, useId } from 'vue'
import { useOverlayDismiss } from '@/composables/useOverlayDismiss'
import { useDialogLifecycle } from '@/composables/useDialogLifecycle'

let hasLockedBodyScroll = false

function ensureBodyScrollLocked() {
  if (hasLockedBodyScroll) return
  lockBodyScroll()
  hasLockedBodyScroll = true
}

function releaseBodyScrollLock() {
  if (!hasLockedBodyScroll) return
  unlockBodyScroll()
  hasLockedBodyScroll = false
}

interface Props {
  modelValue?: boolean
  title?: string
  showHeader?: boolean
  showCloseButton?: boolean
  closeOnOverlay?: boolean
  closeOnEsc?: boolean
  size?: 'small' | 'medium' | 'large' | 'full'
  placement?: 'center' | 'top-end'
  backdrop?: 'default' | 'strong'
  overlayLayer?: 'default' | 'popover'
  backdropEffect?: 'none' | 'blur-sm'
  mobilePresentation?: 'default' | 'fullscreen'
  headerVariant?: 'default' | 'brand'
  frameVariant?: 'default' | 'soft' | 'floating' | 'outlined' | 'warning'
  dividerVariant?: 'default' | 'none' | 'soft'
  footerTone?: 'default' | 'muted'
  customClass?: string
  overlayClass?: string
  bodyClass?: string
  footerClass?: string
  bodyPadding?: 'default' | 'none' | 'compact' | 'spacious'
  scrollMode?: 'auto' | 'contained' | 'none'
  chromeVariant?: 'default' | 'compact' | 'plain' | 'inverse'
  width?: string
  height?: string
  minHeight?: string
  maxWidth?: string
  maxHeight?: string
  headerPadding?: string
  bodyDisplay?: string
  bodyDirection?: string
  bodyMinHeight?: string
  bodyPaddingValue?: string
  bodyTextAlign?: string
  footerGap?: string
  footerPadding?: string
  footerJustify?: string
  footerWrap?: string
}

const props = withDefaults(defineProps<Props>(), {
  modelValue: true,
  title: '',
  showHeader: true,
  showCloseButton: true,
  closeOnOverlay: true,
  closeOnEsc: true,
  size: 'medium',
  placement: 'center',
  backdrop: 'default',
  overlayLayer: 'default',
  backdropEffect: 'none',
  mobilePresentation: 'default',
  headerVariant: 'default',
  frameVariant: 'default',
  dividerVariant: 'default',
  footerTone: 'default',
  customClass: '',
  overlayClass: '',
  bodyClass: '',
  footerClass: '',
  bodyPadding: 'default',
  scrollMode: 'auto',
  chromeVariant: 'default',
  width: '',
  height: '',
  minHeight: '',
  maxWidth: '',
  maxHeight: '',
  headerPadding: '',
  bodyDisplay: '',
  bodyDirection: '',
  bodyMinHeight: '',
  bodyPaddingValue: '',
  bodyTextAlign: '',
  footerGap: '',
  footerPadding: '',
  footerJustify: '',
  footerWrap: '',
})

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  close: []
  open: []
}>()
const dialogRef = ref<HTMLElement | null>(null)

const uiSizeClass = computed(() => {
  return `ui-modal__container--${props.size}`
})

const titleId = useId()

const uiChromeClass = computed(() => {
  return `ui-modal__container--chrome-${props.chromeVariant}`
})

const uiFrameClass = computed(() => {
  return `ui-modal__container--frame-${props.frameVariant}`
})

const uiPlacementClass = computed(() => {
  return `ui-modal__overlay--placement-${props.placement}`
})

const uiBackdropClass = computed(() => {
  return `ui-modal__overlay--backdrop-${props.backdrop}`
})

const uiOverlayLayerClass = computed(() => {
  return `ui-modal__overlay--layer-${props.overlayLayer}`
})

const uiBackdropEffectClass = computed(() => {
  return `ui-modal__overlay--effect-${props.backdropEffect}`
})

const uiMobilePresentationClass = computed(() => {
  return `ui-modal__container--mobile-${props.mobilePresentation}`
})

const uiHeaderVariantClass = computed(() => {
  return `ui-modal__header--${props.headerVariant}`
})

const uiHeaderDividerClass = computed(() => {
  return `ui-modal__header--divider-${props.dividerVariant}`
})

const uiFooterDividerClass = computed(() => {
  return `ui-modal__footer--divider-${props.dividerVariant}`
})

const uiFooterToneClass = computed(() => {
  return `ui-modal__footer--tone-${props.footerTone}`
})

const uiBodyPaddingClass = computed(() => {
  return `ui-modal__body--padding-${props.bodyPadding}`
})

const uiBodyScrollClass = computed(() => {
  return `ui-modal__body--scroll-${props.scrollMode}`
})

const dialogStyle = computed(() => {
  const usesResponsivePresentation = props.mobilePresentation !== 'default'
  const responsiveValue = (name: string, value: string) => value ? `var(${name}, ${value})` : ''
  const layoutEntries: Array<[string, string]> = usesResponsivePresentation
    ? [
        ['width', responsiveValue('--ui-dialog-mobile-width', props.width)],
        ['height', responsiveValue('--ui-dialog-mobile-height', props.height)],
        ['minHeight', responsiveValue('--ui-dialog-mobile-min-height', props.minHeight)],
        ['maxWidth', responsiveValue('--ui-dialog-mobile-max-width', props.maxWidth)],
        ['maxHeight', responsiveValue('--ui-dialog-mobile-max-height', props.maxHeight)],
      ]
    : [
        ['width', props.width],
        ['height', props.height],
        ['minHeight', props.minHeight],
        ['maxWidth', props.maxWidth],
        ['maxHeight', props.maxHeight],
      ]

  const entries: Array<[string, string]> = [
    ...layoutEntries,
    ['--ui-dialog-max-height', props.maxHeight],
    ['--ui-dialog-header-padding', props.headerPadding],
    ['--ui-dialog-body-display', props.bodyDisplay],
    ['--ui-dialog-body-direction', props.bodyDirection],
    ['--ui-dialog-body-min-height', props.bodyMinHeight],
    ['--ui-dialog-body-padding', props.bodyPaddingValue],
    ['--ui-dialog-body-text-align', props.bodyTextAlign],
    ['--ui-dialog-actions-gap', props.footerGap],
    ['--ui-dialog-actions-padding', props.footerPadding],
    ['--ui-dialog-actions-justify', props.footerJustify],
    ['--ui-dialog-actions-wrap', props.footerWrap],
  ]

  return Object.fromEntries(entries.filter(([, value]) => value !== ''))
})

const close = () => {
  emit('update:modelValue', false)
  emit('close')
}

const { overlayRef, handleOverlayMouseDown, resetOverlayDismissState } = useOverlayDismiss(close, {
  enabled: () => props.closeOnOverlay && props.modelValue,
})

useDialogLifecycle({
  open: toRef(props, 'modelValue'),
  container: dialogRef,
  close,
  closeOnEscape: () => props.closeOnEsc,
})

watch(
  () => props.modelValue,
  (newValue) => {
    if (newValue) {
      emit('open')
      ensureBodyScrollLocked()
    } else {
      resetOverlayDismissState()
      releaseBodyScrollLock()
    }
  }
)

onMounted(() => {
  if (props.modelValue) {
    ensureBodyScrollLocked()
  }
})

onUnmounted(() => {
  releaseBodyScrollLock()
})
</script>

<style scoped>
.ui-modal__overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: var(--base-modal-overlay-background);
  display: flex;
  z-index: var(--z-overlay);
}

.ui-modal__overlay--placement-center {
  justify-content: center;
  align-items: center;
}

.ui-modal__overlay--placement-top-end {
  justify-content: flex-end;
  align-items: flex-start;
  padding: 64px 16px 16px;
}

.ui-modal__overlay--backdrop-strong {
  background: var(--color-overlay-backdrop-strong);
}

.ui-modal__overlay--layer-popover {
  z-index: var(--z-popover);
}

.ui-modal__overlay--effect-blur-sm {
  backdrop-filter: blur(4px);
}

.ui-modal__container {
  background: var(--color-surface-base);
  border: 0;
  border-radius: 12px;
  box-shadow: 0 4px 20px var(--base-modal-container-shadow-color);
  max-height: var(--ui-dialog-max-height, 90vh);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

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

.ui-modal__container--frame-soft {
  border-radius: 16px;
}

.ui-modal__container--frame-floating {
  box-shadow: 0 20px 60px var(--base-modal-container-shadow-color);
}

.ui-modal__container--frame-outlined {
  border: 1px solid var(--color-border-default);
  border-radius: 18px;
  box-shadow: 0 24px 64px var(--shadow-medium);
}

.ui-modal__container--frame-warning {
  border: 2px solid var(--color-status-warning);
  border-radius: 16px;
  box-shadow: 0 25px 80px var(--color-overlay-backdrop-strong);
}

.ui-modal__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: var(--ui-dialog-header-padding, 16px 20px);
  border-bottom: 1px solid var(--color-border-muted, var(--color-border-default));
  background: transparent;
  color: inherit;
}

.ui-modal__header--brand {
  padding: 20px 25px;
  background: linear-gradient(135deg, var(--color-action-primary) 0%, var(--color-action-primary-hover) 100%);
  color: var(--color-text-inverse);
}

.ui-modal__header--divider-none {
  border-bottom: 0;
}

.ui-modal__header--divider-soft {
  border-bottom-color: var(--color-border-muted, var(--color-border-soft));
}

.ui-modal__title {
  margin: 0;
  font-size: 1.2em;
  font-weight: 600;
  color: var(--color-text-strong, var(--color-text-heading));
}

.ui-modal__header--brand .ui-modal__title {
  color: var(--color-text-inverse);
  font-size: 1.4em;
}

.ui-modal__close {
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.ui-modal__close:hover {
  background-color: var(--base-modal-close-hover-background);
  color: var(--color-text-strong, var(--color-text-heading));
}

.ui-modal__header--brand .ui-modal__close {
  color: var(--color-text-inverse);
}

.ui-modal__header--brand .ui-modal__close:hover {
  background-color: var(--color-overlay-inverse-soft);
  color: var(--color-text-inverse);
}

.ui-modal__body {
  display: var(--ui-dialog-body-display, block);
  flex-direction: var(--ui-dialog-body-direction, row);
  min-height: var(--ui-dialog-body-min-height, auto);
  padding: var(--ui-dialog-body-padding, 20px);
  overflow-y: auto;
  flex: 1;
  background: transparent;
  text-align: var(--ui-dialog-body-text-align, start);
}

.ui-modal__body--padding-none {
  padding: 0;
}

.ui-modal__body--padding-compact {
  padding: 12px;
}

.ui-modal__body--padding-spacious {
  padding: 24px;
}

.ui-modal__body--scroll-contained {
  overflow: hidden;
}

.ui-modal__body--scroll-none {
  overflow: visible;
}

.ui-modal__footer {
  display: flex;
  justify-content: var(--ui-dialog-actions-justify, flex-end);
  flex-wrap: var(--ui-dialog-actions-wrap, nowrap);
  gap: var(--ui-dialog-actions-gap, 10px);
  padding: var(--ui-dialog-actions-padding, 16px 20px);
  border-top: 1px solid var(--color-border-muted, var(--color-border-default));
  background: transparent;
}

.ui-modal__footer--divider-none {
  border-top: 0;
}

.ui-modal__footer--divider-soft {
  border-top-color: var(--color-border-muted, var(--color-border-soft));
}

.ui-modal__footer--tone-muted {
  background: var(--color-surface-muted);
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

.ui-modal__container--chrome-inverse {
  background: var(--color-overlay-backdrop-solid);
  border: 1px solid var(--color-overlay-inverse-soft);
  color: var(--color-text-inverse);
  box-shadow: 0 18px 48px var(--shadow-medium);
  backdrop-filter: blur(10px);
}

.ui-modal__container--chrome-inverse .ui-modal__title {
  color: var(--color-text-inverse);
  font-size: 15px;
  font-weight: 600;
}

@media (--breakpoint-md-down) {
  .ui-modal__container--mobile-fullscreen {
    --ui-dialog-mobile-width: 100%;
    --ui-dialog-mobile-max-width: 100%;
    --ui-dialog-mobile-min-height: 0;
    --ui-dialog-mobile-max-height: 100dvh;

    margin: 0;
    border-radius: 0;
  }
}

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
