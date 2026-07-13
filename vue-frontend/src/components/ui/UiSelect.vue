<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, useAttrs, useId } from 'vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import type { UiSelectOption, UiSelectValue } from '@/components/ui/selectTypes'

defineOptions({ inheritAttrs: false })

const props = withDefaults(defineProps<{
  modelValue?: UiSelectValue
  options?: UiSelectOption[]
  disabled?: boolean
  error?: boolean | string
  size?: 'lg' | 'md' | 'sm' | 'xs'
  variant?: 'default' | 'studio'
  placeholder?: string
}>(), {
  options: () => [],
  disabled: false,
  error: false,
  size: 'md',
  variant: 'default',
  placeholder: '请选择',
})

const emit = defineEmits<{
  'update:modelValue': [value: UiSelectValue]
  change: [value: UiSelectValue]
}>()

const attrs = useAttrs()
const selectRef = ref<HTMLElement | null>(null)
const dropdownRef = ref<HTMLElement | null>(null)
const isOpen = ref(false)
const dropdownStyle = ref<Record<string, string>>({})
const dropdownId = useId()

const VIEWPORT_PADDING = 12
const DROPDOWN_GAP = 6
const MAX_DROPDOWN_HEIGHT = 320

const selectedOption = computed(() =>
  props.options.find(option => option.value === props.modelValue)
)

const displayValue = computed(() => selectedOption.value?.label ?? props.placeholder)

function getOptionCount(): number {
  return Math.max(props.options.length, 1)
}

function updatePosition(): void {
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
  const width = Math.min(Math.max(rect.width, 160), viewportWidth - VIEWPORT_PADDING * 2)
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
    maxHeight: `${Math.round(maxHeight)}px`,
  }
}

function openDropdown(): void {
  if (props.disabled || isOpen.value) return
  isOpen.value = true
  void nextTick(() => {
    updatePosition()
    requestAnimationFrame(() => updatePosition())
  })
}

function closeDropdown(): void {
  isOpen.value = false
}

function toggleDropdown(): void {
  if (isOpen.value) {
    closeDropdown()
    return
  }
  openDropdown()
}

function selectOption(option: UiSelectOption): void {
  if (option.disabled) return
  emit('update:modelValue', option.value)
  emit('change', option.value)
  closeDropdown()
}

function handleKeydown(event: KeyboardEvent): void {
  if (props.disabled) return

  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault()
    toggleDropdown()
    return
  }

  if (event.key === 'ArrowDown') {
    event.preventDefault()
    openDropdown()
    return
  }

  if (event.key === 'Escape' && isOpen.value) {
    event.preventDefault()
    closeDropdown()
  }
}

function handleClickOutside(event: MouseEvent): void {
  const target = event.target as Node
  if (selectRef.value?.contains(target)) return
  if (dropdownRef.value?.contains(target)) return
  closeDropdown()
}

function handleScrollOrResize(): void {
  if (isOpen.value) {
    updatePosition()
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
  window.addEventListener('scroll', handleScrollOrResize, true)
  window.addEventListener('resize', handleScrollOrResize)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
  window.removeEventListener('scroll', handleScrollOrResize, true)
  window.removeEventListener('resize', handleScrollOrResize)
})
</script>

<template>
  <div
    ref="selectRef"
    v-bind="attrs"
    class="ui-select"
    :class="[
      `ui-select--${size}`,
      `ui-select--${variant}`,
      {
        'ui-select--open': isOpen,
        'ui-select--error': Boolean(error),
        'ui-select--disabled': disabled,
        'ui-select--placeholder': !selectedOption,
      },
    ]"
    role="combobox"
    :tabindex="disabled ? -1 : 0"
    :aria-expanded="isOpen ? 'true' : 'false'"
    aria-haspopup="listbox"
    :aria-controls="isOpen ? dropdownId : undefined"
    :aria-disabled="disabled ? 'true' : undefined"
    :aria-invalid="Boolean(error) ? 'true' : undefined"
    @click="toggleDropdown"
    @keydown="handleKeydown"
  >
    <span class="ui-select__value">{{ displayValue }}</span>
    <span class="ui-select__arrow" aria-hidden="true">
      <UiIcon name="chevron-down" size="14" />
    </span>

    <Teleport to="body">
      <div
        v-if="isOpen"
        :id="dropdownId"
        ref="dropdownRef"
        class="ui-select-dropdown"
        role="listbox"
        :style="dropdownStyle"
      >
        <div
          v-for="option in options"
          :key="String(option.value)"
          class="ui-select-option"
          :class="{ 'ui-select-option--selected': option.value === modelValue }"
          role="option"
          :tabindex="option.disabled ? -1 : 0"
          :aria-selected="option.value === modelValue ? 'true' : 'false'"
          :aria-disabled="option.disabled ? 'true' : undefined"
          :data-ui-select-value="String(option.value)"
          @click="selectOption(option)"
          @keydown.enter.prevent="selectOption(option)"
          @keydown.space.prevent="selectOption(option)"
        >
          {{ option.label }}
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
:where(.ui-select) {
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  min-height: var(--ui-select-min-height, var(--ui-selector-control-min-height));
  padding: var(--ui-select-padding, var(--ui-selector-control-padding));
  border: var(--ui-select-border, 1px solid var(--ui-selector-control-border, var(--color-border-input, ButtonBorder)));
  border-radius: var(--ui-select-radius, var(--ui-selector-control-radius));
  background: var(--ui-select-background, var(--ui-selector-control-background, var(--color-surface-base, Canvas)));
  color: var(--ui-select-color, var(--ui-selector-control-text, var(--color-text-default, CanvasText)));
  font-family: inherit;
  font-size: var(--ui-select-font-size, var(--ui-selector-control-font-size));
  line-height: var(--ui-select-line-height, var(--ui-selector-control-line-height));
  cursor: pointer;
  transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}

:where(.ui-select:hover) {
  border-color: var(--ui-select-hover-border, var(--ui-selector-control-hover-border));
}

:where(.ui-select:focus),
:where(.ui-select--open) {
  outline: none;
  border-color: var(--ui-selector-control-focus-border);
  box-shadow: 0 0 0 2px var(--ui-select-focus-shadow);
}

:where(.ui-select--lg) {
  min-height: 44px;
  padding: 11px 14px;
  font-size: 1rem;
}

:where(.ui-select--sm) {
  min-height: 32px;
  padding: 6px 10px;
  font-size: 0.85rem;
}

:where(.ui-select--xs) {
  min-height: 28px;
  padding: 4px 8px;
  font-size: 0.78rem;
}

:where(.ui-select--studio) {
  min-height: 38px;
  padding: var(--ui-select-padding, 10px 12px);
  border: var(--ui-select-border, 1px solid var(--ui-selector-control-border, var(--color-border-input, ButtonBorder)));
  border-radius: var(--ui-select-radius, 14px);
  background: var(--ui-select-background, var(--ui-selector-control-background, var(--color-surface-base, Canvas)));
  color: var(--ui-select-color, var(--ui-selector-control-text, var(--color-text-default, CanvasText)));
  font-size: var(--ui-select-font-size, var(--ui-selector-control-font-size));
}

:where(.ui-select--studio.ui-select--lg) {
  min-height: 44px;
  padding: var(--ui-select-lg-padding, 12px 14px);
  border-radius: var(--ui-select-lg-radius, 16px);
}

:where(.ui-select--studio:focus),
:where(.ui-select--studio.ui-select--open) {
  border-color: var(--ui-selector-control-focus-border);
  box-shadow: 0 0 0 3px var(--ui-select-focus-shadow);
}

:where(.ui-select--error) {
  border-color: var(--color-status-error, var(--ui-select-error-border));
}

:where(.ui-select--disabled) {
  opacity: 0.65;
  cursor: not-allowed;
}

.ui-select__value {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  color: inherit;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ui-select--placeholder .ui-select__value {
  color: var(--color-text-muted);
}

.ui-select__arrow {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  margin-left: 8px;
  color: var(--ui-selector-arrow-text);
  transition: transform 0.2s ease;
}

.ui-select--open .ui-select__arrow {
  transform: rotate(180deg);
}

.ui-select-dropdown {
  position: fixed;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  overscroll-behavior: contain;
  padding: 6px 0;
  border: 1px solid var(--ui-selector-dropdown-border, var(--color-border-default, ButtonBorder));
  border-radius: var(--ui-selector-dropdown-radius);
  background: var(--ui-selector-dropdown-background, var(--color-surface-base, Canvas));
  box-shadow: 0 12px 26px var(--ui-selector-dropdown-shadow-color);
  color: var(--ui-selector-control-text, var(--color-text-default, CanvasText));
  z-index: var(--z-popover);
}

.ui-select-option {
  display: block;
  width: 100%;
  min-height: 38px;
  padding: 9px 12px;
  border: 0;
  background: var(--ui-selector-dropdown-background, var(--color-surface-base, Canvas));
  color: var(--ui-selector-control-text, var(--color-text-default, CanvasText));
  font: inherit;
  line-height: 1.4;
  text-align: left;
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease;
}

.ui-select-option:hover,
.ui-select-option:focus {
  outline: none;
  background: var(--ui-selector-option-hover-background);
}

.ui-select-option--selected {
  background: var(--ui-selector-option-selected-background);
  color: var(--ui-selector-option-selected-text);
  font-weight: 500;
}

.ui-select-option[aria-disabled="true"] {
  opacity: 0.55;
  cursor: not-allowed;
}

</style>
