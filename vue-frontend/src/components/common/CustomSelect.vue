<template>
  <div
    class="custom-select"
    :class="{ open: isOpen, disabled: disabled }"
    ref="selectRef"
  >
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

    <Teleport to="body">
      <div
        v-if="isOpen"
        ref="dropdownRef"
        class="custom-select-dropdown"
        :style="dropdownStyle"
      >
        <div
          v-if="searchable"
          class="custom-select-search"
          @click.stop
        >
          <input
            ref="searchInputRef"
            v-model="searchQuery"
            type="text"
            class="custom-select-search-input"
            :placeholder="searchPlaceholder"
            @click.stop
          />
        </div>

        <div class="custom-select-options">
          <template v-if="visibleOptionCount > 0 && hasGroups">
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

          <template v-else-if="visibleOptionCount > 0">
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

          <div v-else class="custom-select-empty">
            {{ noResultsText }}
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'

type SelectValue = string | number

interface SelectOption {
  label: string
  value: SelectValue
}

interface SelectGroup {
  label: string
  options: SelectOption[]
}

const props = withDefaults(defineProps<{
  modelValue: SelectValue
  options?: SelectOption[]
  groups?: SelectGroup[]
  placeholder?: string
  disabled?: boolean
  title?: string
  searchable?: boolean
  searchPlaceholder?: string
  noResultsText?: string
}>(), {
  options: () => [],
  groups: () => [],
  placeholder: '请选择',
  disabled: false,
  title: '',
  searchable: false,
  searchPlaceholder: '搜索...',
  noResultsText: '无匹配项'
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: SelectValue): void
  (e: 'change', value: SelectValue): void
}>()

const isOpen = ref(false)
const selectRef = ref<HTMLElement | null>(null)
const dropdownRef = ref<HTMLElement | null>(null)
const searchInputRef = ref<HTMLInputElement | null>(null)
const dropdownStyle = ref<Record<string, string>>({})
const searchQuery = ref('')

const VIEWPORT_PADDING = 12
const DROPDOWN_GAP = 6
const MAX_DROPDOWN_HEIGHT = 360

const hasGroups = computed(() => props.groups.length > 0)
const normalizedSearchQuery = computed(() => searchQuery.value.trim().toLocaleLowerCase())

function optionMatches(option: SelectOption): boolean {
  if (!props.searchable || !normalizedSearchQuery.value) {
    return true
  }

  const keyword = normalizedSearchQuery.value
  return `${option.label} ${String(option.value)}`.toLocaleLowerCase().includes(keyword)
}

const groupedOptions = computed(() => {
  if (!hasGroups.value) return []

  return props.groups
    .map((group) => ({
      ...group,
      options: group.options.filter(optionMatches)
    }))
    .filter((group) => group.options.length > 0)
})

const flatOptions = computed(() => props.options.filter(optionMatches))

const allOptions = computed(() => {
  if (hasGroups.value) {
    return props.groups.flatMap(group => group.options)
  }
  return props.options
})

const visibleOptionCount = computed(() => {
  if (hasGroups.value) {
    return groupedOptions.value.reduce((count, group) => count + group.options.length, 0)
  }
  return flatOptions.value.length
})

const displayValue = computed(() => {
  const option = allOptions.value.find(option => option.value === props.modelValue)
  return option ? option.label : props.placeholder
})

function toggleDropdown(): void {
  if (props.disabled) return

  if (!isOpen.value) {
    searchQuery.value = ''
    isOpen.value = true
    nextTick(() => {
      updatePosition()
      requestAnimationFrame(() => updatePosition())
      if (props.searchable) {
        searchInputRef.value?.focus()
        searchInputRef.value?.select()
      }
    })
    return
  }

  isOpen.value = false
  searchQuery.value = ''
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

function selectOption(value: SelectValue): void {
  emit('update:modelValue', value)
  emit('change', value)
  isOpen.value = false
  searchQuery.value = ''
}

function handleClickOutside(event: MouseEvent): void {
  if (selectRef.value && selectRef.value.contains(event.target as Node)) {
    return
  }

  if (dropdownRef.value && dropdownRef.value.contains(event.target as Node)) {
    return
  }

  isOpen.value = false
  searchQuery.value = ''
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

watch(searchQuery, () => {
  if (isOpen.value) {
    nextTick(() => updatePosition())
  }
})
</script>

<style>
.custom-select {
  position: relative;
  min-width: 160px;
  font-size: 14px;
  color: #1f2430;
}

.custom-select-trigger {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 40px;
  padding: 0 12px;
  border: 1px solid #cfd6e4;
  border-radius: 8px;
  background: #ffffff;
  color: #1f2430;
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
  color: #1f2430;
}

.custom-select-arrow {
  margin-left: 8px;
  color: #666666;
  transition: transform 0.2s;
}

.custom-select.open .custom-select-arrow {
  transform: rotate(180deg);
}

.custom-select-dropdown {
  position: fixed;
  margin-top: 0;
  background: #ffffff;
  border: 1px solid #e0e0e0;
  border-radius: 10px;
  box-shadow: 0 12px 26px rgba(19, 36, 70, 0.18);
  z-index: 2000;
  max-height: 360px;
  overflow-y: auto;
  overscroll-behavior: contain;
  color: #1f2430;
}

.custom-select-search {
  position: sticky;
  top: 0;
  z-index: 1;
  padding: 8px;
  background: #ffffff;
  border-bottom: 1px solid #edf0f6;
}

.custom-select-search-input {
  width: 100%;
  height: 36px;
  padding: 0 10px;
  border: 1px solid #cfd6e4;
  border-radius: 8px;
  background: #ffffff;
  color: #1f2430;
  outline: none;
}

.custom-select-search-input:focus {
  border-color: #5b73f2;
  box-shadow: 0 0 0 2px rgba(88, 125, 255, 0.16);
}

.custom-select-options {
  padding: 6px 0;
  background: #ffffff;
  color: #1f2430;
}

.custom-select-group {
  margin-bottom: 4px;
  background: #ffffff;
}

.custom-select-group:last-child {
  margin-bottom: 0;
}

.custom-select-group-label {
  padding: 8px 12px 4px;
  font-size: 11px;
  font-weight: 600;
  color: #666666;
  background: #f5f5f5;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.custom-select-option {
  padding: 9px 12px;
  cursor: pointer;
  color: #1f2430;
  background: #ffffff;
  font-size: 14px;
  line-height: 1.4;
  transition: background 0.15s;
}

.custom-select-option:hover {
  background: #e3f2fd;
  color: #1f2430;
}

.custom-select-option.selected {
  background: #e8edff;
  color: #3040c2;
  font-weight: 500;
}

.custom-select-empty {
  padding: 18px 12px;
  text-align: center;
  color: #7a8194;
  font-size: 13px;
}
</style>
