<template>
  <div
    ref="selectRef"
    class="ui-combobox"
    :class="[
      `ui-combobox--${variant}`,
      {
        'ui-combobox--open': isOpen,
        'ui-combobox--disabled': disabled,
        'ui-combobox--fit': fit,
      }
    ]"
  >
    <div
      :id="inputId"
      class="ui-combobox-trigger"
      role="combobox"
      :tabindex="disabled ? -1 : 0"
      :aria-label="ariaLabel || undefined"
      :aria-expanded="isOpen ? 'true' : 'false'"
      aria-haspopup="listbox"
      :aria-controls="isOpen ? dropdownId : undefined"
      :aria-disabled="disabled ? 'true' : undefined"
      @click="toggleDropdown"
      @keydown="handleTriggerKeydown"
      :title="title"
    >
      <span class="ui-combobox-value">{{ displayValue }}</span>
      <span class="ui-combobox-arrow">
        <UiIcon name="chevron-down" size="12" />
      </span>
    </div>

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
            >
              <div class="ui-combobox-group-label">{{ group.label }}</div>
              <div
                v-for="option in group.options"
                :key="option.value"
                class="ui-combobox-option"
                :class="{ 'ui-combobox-option--selected': option.value === modelValue }"
                role="option"
                :aria-selected="option.value === modelValue ? 'true' : 'false'"
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
              class="ui-combobox-option"
              :class="{ 'ui-combobox-option--selected': option.value === modelValue }"
              role="option"
              :aria-selected="option.value === modelValue ? 'true' : 'false'"
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
import { ref, computed, onMounted, onUnmounted, nextTick, useId } from 'vue'
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
  variant?: 'default' | 'compact' | 'workflow'
  fit?: boolean
}>(), {
  inputId: undefined,
  ariaLabel: '',
  options: () => [],
  groups: () => [],
  placeholder: '请选择',
  disabled: false,
  title: '',
  variant: 'default',
  fit: false
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: UiSelectValue): void
  (e: 'change', value: UiSelectValue): void
}>()

const isOpen = ref(false)
const selectRef = ref<HTMLElement | null>(null)
const dropdownRef = ref<HTMLElement | null>(null)
const dropdownStyle = ref<Record<string, string>>({})
const dropdownId = useId()

const VIEWPORT_PADDING = 12
const DROPDOWN_GAP = 6
const MAX_DROPDOWN_HEIGHT = 360

const hasGroups = computed(() => props.groups && props.groups.length > 0)
const groupedOptions = computed(() => props.groups)
const flatOptions = computed(() => props.options)

const allOptions = computed(() => {
  if (hasGroups.value) {
    return props.groups.flatMap(g => g.options)
  }
  return props.options
})

const displayValue = computed(() => {
  const option = allOptions.value.find(o => o.value === props.modelValue)
  return option ? option.label : props.placeholder
})

function toggleDropdown(): void {
  if (props.disabled) return

  if (!isOpen.value) {
    openDropdown()
  } else {
    closeDropdown()
  }
}

function openDropdown(): void {
  if (props.disabled || isOpen.value) return
  isOpen.value = true
  nextTick(() => {
    updatePosition()
    requestAnimationFrame(() => updatePosition())
  })
}

function closeDropdown(): void {
  isOpen.value = false
}

function handleTriggerKeydown(event: KeyboardEvent): void {
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

function selectOption(value: UiSelectValue): void {
  emit('update:modelValue', value)
  emit('change', value)
  closeDropdown()
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
</script>

<style scoped>
.ui-combobox {
  position: relative;
  min-width: 160px;
  font-size: 14px;
  color: var(--ui-selector-control-text);
}

.ui-combobox--fit {
  width: 100%;
  min-width: 0;
}

.ui-combobox-trigger {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: var(--ui-selector-control-min-height);
  padding: var(--ui-selector-control-padding);
  border: 1px solid var(--ui-selector-control-border);
  border-radius: var(--ui-selector-control-radius);
  background: var(--ui-selector-control-background);
  color: var(--ui-selector-control-text);
  cursor: pointer;
  transition: border-color 0.15s, box-shadow 0.15s;
}

.ui-combobox--compact {
  min-width: 70px;
  flex: 0 0 auto;
}

.ui-combobox--compact .ui-combobox-trigger {
  height: 38px;
  padding: 0 10px;
}

.ui-combobox--workflow .ui-combobox-trigger {
  min-height: 42px;
  border-color: var(--ui-selector-workflow-border);
  border-radius: 10px;
}

.ui-combobox-trigger:hover {
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
  color: var(--ui-selector-control-text);
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
  /* Teleport 下拉层使用视口坐标定位，top/left/width 由 JS 计算。 */
  position: fixed;
  margin-top: 0;
  background: var(--ui-selector-dropdown-background);
  border: 1px solid var(--ui-selector-dropdown-border);
  border-radius: var(--ui-selector-dropdown-radius);
  box-shadow: 0 12px 26px var(--ui-selector-dropdown-shadow-color);
  z-index: var(--z-popover);
  max-height: 360px;
  overflow-y: auto;
  overscroll-behavior: contain;
  color: var(--ui-selector-control-text);
}

.ui-combobox-options {
  padding: 6px 0;
  background: var(--ui-selector-dropdown-background);
  color: var(--ui-selector-control-text);
}

.ui-combobox-group {
  margin-bottom: 4px;
  background: var(--ui-selector-dropdown-background);
}

.ui-combobox-group:last-child {
  margin-bottom: 0;
}

.ui-combobox-group-label {
  padding: 8px 12px 4px;
  font-size: 11px;
  font-weight: 600;
  color: var(--ui-selector-group-label-text);
  background: var(--ui-selector-group-label-background);
  text-transform: uppercase;
  letter-spacing: 0;
}

.ui-combobox-option {
  padding: 9px 12px;
  cursor: pointer;
  color: var(--ui-selector-control-text);
  background: var(--ui-selector-dropdown-background);
  font-size: 14px;
  line-height: 1.4;
  transition: background 0.15s;
}

.ui-combobox-option:hover {
  background: var(--ui-selector-option-hover-background);
  color: var(--ui-selector-control-text);
}

.ui-combobox-option--selected {
  background: var(--ui-selector-option-selected-background);
  color: var(--ui-selector-option-selected-text);
  font-weight: 500;
}

</style>
