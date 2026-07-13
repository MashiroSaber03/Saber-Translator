<script setup lang="ts">
import { computed } from 'vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import type { RouteLocationRaw } from 'vue-router'
import type { UiIconName } from '@/components/ui/iconRegistry'

const props = withDefaults(defineProps<{
  as?: 'button' | 'a' | 'router-link' | 'span'
  to?: RouteLocationRaw
  href?: string
  target?: string
  rel?: string
  type?: 'button' | 'submit' | 'reset'
  label?: string
  title?: string
  ariaLabel?: string
  iconName?: UiIconName
  iconSize?: number
  iconOnly?: boolean
  collapseLabelOnMobile?: boolean
  active?: boolean
  pressed?: boolean
  variant?: 'ghost' | 'solid' | 'plain'
  tone?: 'neutral' | 'primary' | 'danger'
  disabled?: boolean
}>(), {
  as: 'button',
  to: '/',
  href: '#',
  target: undefined,
  rel: undefined,
  type: 'button',
  label: undefined,
  title: undefined,
  ariaLabel: undefined,
  iconName: undefined,
  iconSize: 18,
  iconOnly: false,
  collapseLabelOnMobile: false,
  active: false,
  pressed: undefined,
  variant: 'ghost',
  tone: 'neutral',
  disabled: false,
})

const emit = defineEmits<{
  click: [event: MouseEvent]
}>()

const componentTag = computed(() => {
  if (props.as === 'router-link') return 'RouterLink'
  if (props.as === 'a') return 'a'
  if (props.as === 'span') return 'span'
  return 'button'
})

const actionAttrs = computed(() => {
  if (props.as === 'router-link') {
    return {
      to: props.to,
      'aria-disabled': props.disabled ? 'true' : undefined,
    }
  }

  if (props.as === 'a') {
    return {
      href: props.disabled ? undefined : props.href,
      target: props.target,
      rel: props.rel,
      'aria-disabled': props.disabled ? 'true' : undefined,
    }
  }

  if (props.as === 'span') {
    return {}
  }

  return {
    type: props.type,
    disabled: props.disabled,
  }
})

const accessibleLabel = computed(() => props.ariaLabel || ((props.iconOnly || props.collapseLabelOnMobile) ? props.label : undefined))
const titleText = computed(() => props.title || props.label)

function handleClick(event: MouseEvent) {
  if (props.disabled || props.as === 'span') {
    event.preventDefault()
    event.stopPropagation()
    return
  }
  emit('click', event)
}
</script>

<template>
  <component
    :is="componentTag"
    class="product-header-action"
    :class="[
      `product-header-action--${variant}`,
      `product-header-action--tone-${tone}`,
      {
        'product-header-action--active': active,
        'product-header-action--icon-only': iconOnly,
        'product-header-action--collapse-label-md': collapseLabelOnMobile,
        'product-header-action--static': as === 'span',
        'product-header-action--disabled': disabled,
      },
    ]"
    v-bind="actionAttrs"
    :aria-label="accessibleLabel"
    :aria-pressed="pressed === undefined ? undefined : String(pressed)"
    :title="titleText"
    @click="handleClick"
  >
    <UiIcon v-if="iconName" :name="iconName" :size="iconSize" />
    <span v-if="label && !iconOnly" class="product-header-action__label">{{ label }}</span>
    <slot />
  </component>
</template>

<style scoped>
.product-header-action {
  --internal-product-header-action-surface: var(--product-header-action-context-surface, var(--color-surface-muted));
  --internal-product-header-action-border-color: var(--product-header-action-context-border, transparent);
  --internal-product-header-action-text-color: var(--product-header-action-context-text, var(--color-text-heading));
  --internal-product-header-action-hover-surface: var(--product-header-action-context-hover-surface, var(--color-surface-interactive-hover));
  --internal-product-header-action-hover-border-color: var(--product-header-action-context-hover-border, var(--product-header-action-border-color, var(--internal-product-header-action-border-color)));
  --internal-product-header-action-hover-text-color: var(--product-header-action-context-hover-text, var(--product-header-action-text-color, var(--internal-product-header-action-text-color)));
  --internal-product-header-action-solid-surface: var(--product-header-action-context-solid-surface, linear-gradient(135deg, var(--color-action-brand) 0%, var(--color-action-brand-strong) 100%));
  --internal-product-header-action-solid-hover-surface: var(--product-header-action-context-solid-hover-surface, var(--product-header-action-solid-surface, var(--internal-product-header-action-solid-surface)));
  --internal-product-header-action-solid-text-color: var(--product-header-action-context-solid-text, var(--color-text-inverse));
  --internal-product-header-action-solid-shadow-color: var(--product-header-action-context-solid-shadow, var(--shadow-action-brand));
  --internal-product-header-action-plain-text-color: var(--product-header-action-context-plain-text, var(--product-header-action-text-color, var(--internal-product-header-action-text-color)));
  --internal-product-header-action-active-surface: var(--product-header-action-context-active-surface, var(--color-focus-brand-soft));
  --internal-product-header-action-active-text-color: var(--product-header-action-context-active-text, var(--color-action-primary));

  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 38px;
  padding: 8px 12px;
  border: 1px solid var(--product-header-action-border-color, var(--internal-product-header-action-border-color));
  border-radius: 999px;
  background: var(--product-header-action-surface, var(--internal-product-header-action-surface));
  color: var(--product-header-action-text-color, var(--internal-product-header-action-text-color));
  cursor: pointer;
  font: inherit;
  font-size: 0.9rem;
  font-weight: 600;
  line-height: 1;
  text-decoration: none;
  transition: background-color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease, color 0.2s ease, transform 0.2s ease;
  user-select: none;
  white-space: nowrap;
}

.product-header-action:hover:not(.product-header-action--disabled, .product-header-action--static) {
  background: var(--product-header-action-hover-surface, var(--internal-product-header-action-hover-surface));
  border-color: var(--product-header-action-hover-border-color, var(--internal-product-header-action-hover-border-color));
  color: var(--product-header-action-hover-text-color, var(--internal-product-header-action-hover-text-color));
  transform: translateY(-1px);
}

.product-header-action--solid {
  background: var(--product-header-action-solid-surface, var(--internal-product-header-action-solid-surface));
  border-color: transparent;
  color: var(--product-header-action-solid-text-color, var(--internal-product-header-action-solid-text-color));
  box-shadow: 0 4px 12px var(--product-header-action-solid-shadow-color, var(--internal-product-header-action-solid-shadow-color));
}

.product-header-action--solid:hover:not(.product-header-action--disabled, .product-header-action--static) {
  background: var(--product-header-action-solid-hover-surface, var(--internal-product-header-action-solid-hover-surface));
  border-color: transparent;
  color: var(--product-header-action-solid-text-color, var(--internal-product-header-action-solid-text-color));
  box-shadow: 0 6px 18px var(--product-header-action-solid-shadow-color, var(--internal-product-header-action-solid-shadow-color));
}

.product-header-action--plain {
  min-height: auto;
  padding: 0;
  border-color: transparent;
  background: transparent;
  color: var(--product-header-action-plain-text-color, var(--internal-product-header-action-plain-text-color));
}

.product-header-action--plain:hover:not(.product-header-action--disabled, .product-header-action--static) {
  background: transparent;
  border-color: transparent;
  color: var(--product-header-action-hover-text-color, var(--internal-product-header-action-hover-text-color));
}

.product-header-action--tone-danger {
  color: var(--product-header-action-danger-text-color, var(--color-status-error));
}

.product-header-action--tone-danger:hover:not(.product-header-action--disabled, .product-header-action--static) {
  color: var(--product-header-action-danger-text-color, var(--color-status-error));
}

.product-header-action--active {
  background: var(--product-header-action-active-surface, var(--internal-product-header-action-active-surface));
  color: var(--product-header-action-active-text-color, var(--internal-product-header-action-active-text-color));
}

.product-header-action--icon-only {
  width: 38px;
  padding: 0;
}

.product-header-action--disabled {
  cursor: not-allowed;
  opacity: 0.58;
}

.product-header-action--static {
  cursor: default;
  pointer-events: none;
}

.product-header-action--static:hover {
  transform: none;
}

.product-header-action__label {
  min-width: 0;
}

@media (--breakpoint-md-down) {
  .product-header-action {
    min-height: 34px;
    padding: 6px 10px;
    font-size: 0.82rem;
  }

  .product-header-action--icon-only {
    width: 34px;
  }

  .product-header-action--collapse-label-md {
    width: 34px;
    padding: 0;
  }

  .product-header-action--collapse-label-md .product-header-action__label {
    display: none;
  }
}
</style>
