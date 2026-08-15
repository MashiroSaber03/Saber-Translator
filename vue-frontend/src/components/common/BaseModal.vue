<template>
  <Teleport to="body">
    <Transition name="modal-fade">
      <div
        v-if="modelValue"
        ref="overlayRef"
        class="ui-modal__overlay"
        :class="[uiBackdropClass, uiBackdropEffectClass]"
        data-testid="base-dialog-overlay"
        @mousedown.self="handleOverlayMouseDown"
      >
        <div
          ref="dialogRef"
          class="ui-modal__container"
          :class="[uiSizeClass, uiFrameClass, uiMobilePresentationClass, customClass]"
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
            :class="[uiBodyPaddingClass, uiBodyScrollClass]"
            data-testid="base-dialog-body"
          >
            <slot></slot>
          </div>

          <div
            v-if="$slots.footer"
            class="ui-modal__footer"
            :class="[uiFooterDividerClass, uiFooterToneClass]"
            data-testid="base-dialog-footer"
          >
            <slot name="footer"></slot>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import { computed, ref, toRef, watch, useId } from 'vue'
import { useBodyScrollLock } from '@/composables/useBodyScrollLock'
import { useOverlayDismiss } from '@/composables/useOverlayDismiss'
import { useDialogLifecycle } from '@/composables/useDialogLifecycle'

interface Props {
  modelValue?: boolean
  title?: string
  showHeader?: boolean
  showCloseButton?: boolean
  closeOnOverlay?: boolean
  closeOnEsc?: boolean
  size?: 'small' | 'medium' | 'large' | 'full'
  backdrop?: 'default' | 'strong'
  backdropEffect?: 'none' | 'blur-sm'
  mobilePresentation?: 'default' | 'fullscreen'
  headerVariant?: 'default' | 'brand'
  frameVariant?: 'default' | 'soft' | 'floating' | 'outlined' | 'warning'
  dividerVariant?: 'default' | 'none' | 'soft'
  footerTone?: 'default' | 'muted'
  customClass?: string
  bodyPadding?: 'default' | 'none' | 'compact' | 'spacious'
  scrollMode?: 'auto' | 'contained' | 'none'
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
  footerPadding?: string
}

const props = withDefaults(defineProps<Props>(), {
  modelValue: true,
  title: '',
  showHeader: true,
  showCloseButton: true,
  closeOnOverlay: true,
  closeOnEsc: true,
  size: 'medium',
  backdrop: 'default',
  backdropEffect: 'none',
  mobilePresentation: 'default',
  headerVariant: 'default',
  frameVariant: 'default',
  dividerVariant: 'default',
  footerTone: 'default',
  customClass: '',
  bodyPadding: 'default',
  scrollMode: 'auto',
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
  footerPadding: '',
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

const uiFrameClass = computed(() => {
  return `ui-modal__container--frame-${props.frameVariant}`
})

const uiBackdropClass = computed(() => {
  return `ui-modal__overlay--backdrop-${props.backdrop}`
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
  const responsiveValue = (name: string, value: string) => (value ? `var(${name}, ${value})` : '')
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
    ['--ui-dialog-actions-padding', props.footerPadding],
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

useBodyScrollLock(toRef(props, 'modelValue'))

watch(
  () => props.modelValue,
  newValue => {
    if (newValue) {
      emit('open')
    } else {
      resetOverlayDismissState()
    }
  }
)
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
  justify-content: center;
  align-items: center;
  z-index: var(--z-overlay);
}

.ui-modal__overlay--backdrop-strong {
  background: var(--color-overlay-backdrop-strong);
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

.ui-modal__container:focus {
  outline: none;
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
  background: linear-gradient(
    135deg,
    var(--color-action-primary) 0%,
    var(--color-action-primary-hover) 100%
  );
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

.ui-modal__close:focus:not(:focus-visible) {
  outline: none;
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
  overflow-x: hidden;
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
