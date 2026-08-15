<template>
  <div
    ref="selectRef"
    class="ui-combobox"
    :class="{
      'ui-combobox--open': isOpen,
      'ui-combobox--disabled': disabled,
      'ui-combobox--fit': fit,
    }"
  >
    <button
      ref="triggerRef"
      :id="inputId"
      class="ui-combobox-trigger"
      type="button"
      role="combobox"
      :disabled="disabled"
      :aria-label="ariaLabel || undefined"
      :aria-expanded="isOpen ? 'true' : 'false'"
      aria-haspopup="listbox"
      :aria-controls="isOpen ? dropdownId : undefined"
      :aria-activedescendant="activeDescendant"
      :aria-disabled="disabled ? 'true' : undefined"
      @click="toggleDropdown"
      @keydown="handleTriggerKeydown"
      :title="title"
    >
      <span class="ui-combobox-value">{{ displayValue }}</span>
      <span class="ui-combobox-arrow">
        <UiIcon name="chevron-down" size="12" />
      </span>
    </button>

    <Teleport to="body">
      <div
        v-if="isOpen"
        :id="dropdownId"
        ref="dropdownRef"
        class="ui-combobox-dropdown"
        role="listbox"
        :style="dropdownStyle"
      >
        <div class="ui-combobox-options">
          <template v-if="hasGroups">
            <div
              v-for="group in groupedOptions"
              :key="group.label"
              class="ui-combobox-group"
              role="group"
              :aria-label="group.label"
            >
              <div class="ui-combobox-group-label" aria-hidden="true">{{ group.label }}</div>
              <div
                v-for="entry in group.options"
                :id="optionId(entry.index)"
                :key="`${typeof entry.option.value}:${String(entry.option.value)}`"
                class="ui-combobox-option"
                :class="{
                  'ui-combobox-option--selected': entry.option.value === modelValue,
                  'ui-combobox-option--active': entry.index === activeIndex,
                }"
                role="option"
                :aria-selected="entry.option.value === modelValue ? 'true' : 'false'"
                :aria-disabled="entry.option.disabled ? 'true' : undefined"
                @mouseenter="setActiveOption(entry)"
                @mousedown.prevent
                @click="selectOption(entry.option)"
              >
                {{ entry.option.label }}
              </div>
            </div>
          </template>
          <template v-else>
            <div
              v-for="entry in flatOptions"
              :id="optionId(entry.index)"
              :key="`${typeof entry.option.value}:${String(entry.option.value)}`"
              class="ui-combobox-option"
              :class="{
                'ui-combobox-option--selected': entry.option.value === modelValue,
                'ui-combobox-option--active': entry.index === activeIndex,
              }"
              role="option"
              :aria-selected="entry.option.value === modelValue ? 'true' : 'false'"
              :aria-disabled="entry.option.disabled ? 'true' : undefined"
              @mouseenter="setActiveOption(entry)"
              @mousedown.prevent
              @click="selectOption(entry.option)"
            >
              {{ entry.option.label }}
            </div>
          </template>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick, useId, watch } from 'vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import type { UiSelectGroup, UiSelectOption, UiSelectValue } from '@/components/ui/selectTypes'

const props = withDefaults(defineProps<{
  modelValue: UiSelectValue
  inputId?: string
  ariaLabel?: string
  options?: UiSelectOption[]
  groups?: UiSelectGroup[]
  placeholder?: string
  disabled?: boolean
  title?: string
  fit?: boolean
}>(), {
  inputId: undefined,
  ariaLabel: '',
  options: () => [],
  groups: () => [],
  placeholder: '请选择',
  disabled: false,
  title: '',
  fit: false
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: UiSelectValue): void
  (e: 'change', value: UiSelectValue): void
}>()

interface IndexedOption {
  option: UiSelectOption
  index: number
}

const isOpen = ref(false)
const activeIndex = ref(-1)
const selectRef = ref<HTMLElement | null>(null)
const triggerRef = ref<HTMLButtonElement | null>(null)
const dropdownRef = ref<HTMLElement | null>(null)
const dropdownStyle = ref<Record<string, string>>({})
const dropdownId = useId()

const VIEWPORT_PADDING = 12
const DROPDOWN_GAP = 6
const MAX_DROPDOWN_HEIGHT = 360

const hasGroups = computed(() => props.groups && props.groups.length > 0)
const allOptions = computed(() => {
  if (hasGroups.value) {
    return props.groups.flatMap(g => g.options)
  }
  return props.options
})
const flatOptions = computed<IndexedOption[]>(() => allOptions.value.map((option, index) => ({
  option,
  index,
})))
const groupedOptions = computed(() => {
  let index = 0
  return props.groups.map(group => ({
    label: group.label,
    options: group.options.map(option => ({ option, index: index++ })),
  }))
})
const enabledOptionIndexes = computed(() => flatOptions.value
  .filter(entry => !entry.option.disabled)
  .map(entry => entry.index))
const activeDescendant = computed(() => (
  isOpen.value && activeIndex.value >= 0
    ? optionId(activeIndex.value)
    : undefined
))

const displayValue = computed(() => {
  const option = allOptions.value.find(o => o.value === props.modelValue)
  return option ? option.label : props.placeholder
})

function optionId(index: number): string {
  return `${dropdownId}-option-${index}`
}

function resetActiveIndex(preferred: 'selected' | 'first' | 'last' = 'selected'): void {
  const enabledIndexes = enabledOptionIndexes.value
  if (enabledIndexes.length === 0) {
    activeIndex.value = -1
    return
  }

  if (preferred === 'first') {
    activeIndex.value = enabledIndexes[0] ?? -1
    return
  }
  if (preferred === 'last') {
    activeIndex.value = enabledIndexes.at(-1) ?? -1
    return
  }

  const selectedIndex = allOptions.value.findIndex(option => (
    !option.disabled && option.value === props.modelValue
  ))
  activeIndex.value = selectedIndex >= 0 ? selectedIndex : (enabledIndexes[0] ?? -1)
}

function moveActiveIndex(direction: -1 | 1): void {
  const enabledIndexes = enabledOptionIndexes.value
  if (enabledIndexes.length === 0) return

  const currentPosition = enabledIndexes.indexOf(activeIndex.value)
  if (currentPosition < 0) {
    activeIndex.value = direction > 0 ? (enabledIndexes[0] ?? -1) : (enabledIndexes.at(-1) ?? -1)
    return
  }

  const nextPosition = (currentPosition + direction + enabledIndexes.length) % enabledIndexes.length
  activeIndex.value = enabledIndexes[nextPosition] ?? -1
}

function setActiveOption(entry: IndexedOption): void {
  if (!entry.option.disabled) activeIndex.value = entry.index
}

function toggleDropdown(): void {
  if (props.disabled) return

  if (!isOpen.value) {
    openDropdown()
  } else {
    closeDropdown()
  }
}

function openDropdown(preferred: 'selected' | 'first' | 'last' = 'selected'): void {
  if (props.disabled || isOpen.value) return
  resetActiveIndex(preferred)
  isOpen.value = true
  void nextTick(() => {
    updatePosition()
    requestAnimationFrame(() => updatePosition())
  })
}

function closeDropdown(): void {
  isOpen.value = false
}

function handleTriggerKeydown(event: KeyboardEvent): void {
  if (props.disabled) return

  if (event.key === 'Escape' && isOpen.value) {
    event.preventDefault()
    closeDropdown()
    return
  }

  if (!isOpen.value) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      openDropdown()
      return
    }
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      openDropdown(event.key === 'ArrowDown' ? 'first' : 'last')
    }
    return
  }

  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault()
    moveActiveIndex(event.key === 'ArrowDown' ? 1 : -1)
    return
  }

  if (event.key === 'Home' || event.key === 'End') {
    event.preventDefault()
    resetActiveIndex(event.key === 'Home' ? 'first' : 'last')
    return
  }

  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault()
    const option = allOptions.value[activeIndex.value]
    if (option) selectOption(option)
    return
  }

  if (event.key === 'Tab') {
    closeDropdown()
  }
}

function getOptionCount(): number {
  if (hasGroups.value) {
    return props.groups.reduce((count, group) => count + group.options.length + 1, 0)
  }
  return props.options.length
}

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

function selectOption(option: UiSelectOption): void {
  if (option.disabled) return
  emit('update:modelValue', option.value)
  emit('change', option.value)
  closeDropdown()
  triggerRef.value?.focus()
}

function handleClickOutside(event: MouseEvent): void {
  if (selectRef.value && selectRef.value.contains(event.target as Node)) {
    return
  }

  if (dropdownRef.value && dropdownRef.value.contains(event.target as Node)) {
    return
  }

  closeDropdown()
}

function handleScrollOrResize() {
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

watch(() => props.disabled, (disabled) => {
  if (disabled) closeDropdown()
})

watch([() => props.options, () => props.groups, () => props.modelValue], () => {
  if (isOpen.value) resetActiveIndex()
})

watch(activeIndex, (index) => {
  if (!isOpen.value || index < 0) return
  void nextTick(() => document.getElementById(optionId(index))?.scrollIntoView?.({ block: 'nearest' }))
})
</script>

<style scoped>
.ui-combobox {
  position: relative;
  min-width: 160px;
  font-size: 14px;
  color: var(--ui-selector-control-text, var(--color-text-default, CanvasText));
}

.ui-combobox--fit {
  width: 100%;
  min-width: 0;
}

.ui-combobox-trigger {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  height: var(--ui-selector-control-min-height);
  padding: var(--ui-selector-control-padding);
  border: 1px solid var(--ui-selector-control-border, var(--color-border-input, ButtonBorder));
  border-radius: var(--ui-selector-control-radius);
  background: var(--ui-selector-control-background, var(--color-surface-base, Canvas));
  color: var(--ui-selector-control-text, var(--color-text-default, CanvasText));
  appearance: none;
  font: inherit;
  text-align: left;
  cursor: pointer;
  transition: border-color 0.15s, box-shadow 0.15s;
}

.ui-combobox-trigger:hover:not(:disabled) {
  border-color: var(--ui-selector-control-hover-border);
}

.ui-combobox--open .ui-combobox-trigger {
  border-color: var(--ui-selector-control-focus-border);
  box-shadow: 0 0 0 2px var(--ui-selector-control-focus-shadow);
}

.ui-combobox--disabled .ui-combobox-trigger {
  opacity: 0.6;
  cursor: not-allowed;
}

.ui-combobox-value {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--ui-selector-control-text, var(--color-text-default, CanvasText));
}

.ui-combobox-arrow {
  margin-left: 8px;
  color: var(--ui-selector-arrow-text);
  transition: transform 0.2s;
}

.ui-combobox--open .ui-combobox-arrow {
  transform: rotate(180deg);
}

.ui-combobox-dropdown {
  position: fixed;
  max-height: 360px;
  margin-top: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
  border: 1px solid var(--ui-selector-dropdown-border, var(--color-border-default, ButtonBorder));
  border-radius: var(--ui-selector-dropdown-radius);
  background: var(--ui-selector-dropdown-background, var(--color-surface-base, Canvas));
  box-shadow: 0 12px 26px var(--ui-selector-dropdown-shadow-color);
  color: var(--ui-selector-control-text, var(--color-text-default, CanvasText));
  z-index: var(--z-popover);
}

.ui-combobox-options {
  padding: 6px 0;
  background: var(--ui-selector-dropdown-background, var(--color-surface-base, Canvas));
  color: var(--ui-selector-control-text, var(--color-text-default, CanvasText));
}

.ui-combobox-group {
  margin-bottom: 4px;
  background: var(--ui-selector-dropdown-background, var(--color-surface-base, Canvas));
}

.ui-combobox-group:last-child {
  margin-bottom: 0;
}

.ui-combobox-group-label {
  padding: 8px 12px 4px;
  background: var(--ui-selector-group-label-background);
  color: var(--ui-selector-group-label-text);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0;
  text-transform: uppercase;
}

.ui-combobox-option {
  padding: 9px 12px;
  background: var(--ui-selector-dropdown-background, var(--color-surface-base, Canvas));
  color: var(--ui-selector-control-text, var(--color-text-default, CanvasText));
  font-size: 14px;
  line-height: 1.4;
  cursor: pointer;
  transition: background 0.15s;
}

.ui-combobox-option:hover,
.ui-combobox-option--active {
  background: var(--ui-selector-option-hover-background);
  color: var(--ui-selector-control-text, var(--color-text-default, CanvasText));
}

.ui-combobox-option--selected {
  background: var(--ui-selector-option-selected-background);
  color: var(--ui-selector-option-selected-text);
  font-weight: 500;
}

.ui-combobox-option[aria-disabled="true"] {
  opacity: 0.55;
  cursor: not-allowed;
}

</style>
