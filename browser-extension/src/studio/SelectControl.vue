<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, useId } from 'vue'
export type Choice = { value: string | number; label: string }
const props = defineProps<{
  modelValue: string | number
  options: Choice[]
  label: string
  disabled?: boolean
  searchable?: boolean
}>()
const emit = defineEmits<{ 'update:modelValue': [value: string | number] }>()
const id = useId()
const open = ref(false)
const query = ref('')
const index = ref(0)
const trigger = ref<HTMLButtonElement>()
const popup = ref<HTMLElement>()
const position = ref({})
let anchor: DOMRect | undefined
const choices = computed(() =>
  props.options.filter(option => option.label.toLowerCase().includes(query.value.toLowerCase()))
)
const selected = computed(
  () => props.options.find(option => option.value === props.modelValue)?.label || '请选择'
)
function close() {
  open.value = false
  query.value = ''
}
async function toggle() {
  if (open.value) {
    close()
    return
  }
  const rect = trigger.value!.getBoundingClientRect()
  anchor = rect
  const below = innerHeight - rect.bottom - 12
  const above = rect.top - 12
  position.value = {
    left: `${rect.left}px`,
    width: `${rect.width}px`,
    maxHeight: `${Math.min(280, Math.max(below, above))}px`,
    ...(below >= Math.min(240, above)
      ? { top: `${rect.bottom + 6}px` }
      : { bottom: `${innerHeight - rect.top + 6}px` }),
  }
  index.value = Math.max(
    0,
    props.options.findIndex(option => option.value === props.modelValue)
  )
  open.value = true
  await nextTick()
  if (props.searchable) popup.value?.querySelector('input')?.focus({ preventScroll: true })
  else popup.value?.querySelector<HTMLElement>('[role="listbox"]')?.focus({ preventScroll: true })
}
function choose(value: string | number) {
  emit('update:modelValue', value)
  close()
  trigger.value?.focus()
}
function key(event: KeyboardEvent) {
  if (event.key === 'Escape') {
    event.preventDefault()
    event.stopPropagation()
    close()
    trigger.value?.focus()
  }
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault()
    if (!open.value) {
      void toggle()
      return
    }
    index.value = Math.max(
      0,
      Math.min(choices.value.length - 1, index.value + (event.key === 'ArrowDown' ? 1 : -1))
    )
    void nextTick(() =>
      popup.value?.querySelector('[data-active="true"]')?.scrollIntoView({ block: 'nearest' })
    )
  }
  if (event.key === 'Enter' && open.value) {
    event.preventDefault()
    const option = choices.value[index.value]
    if (option) choose(option.value)
  }
  if (event.key === 'Tab' && open.value) {
    close()
    trigger.value?.focus()
  }
}
function outside(event: PointerEvent) {
  if (
    !trigger.value?.contains(event.target as Node) &&
    !popup.value?.contains(event.target as Node)
  )
    close()
}
function scroll(event: Event) {
  if (!open.value || popup.value?.contains(event.target as Node)) return
  const rect = trigger.value!.getBoundingClientRect()
  // A scroll queued before opening need not dismiss a correctly positioned menu.
  if (rect.top !== anchor!.top || rect.left !== anchor!.left) close()
}
window.addEventListener('pointerdown', outside)
window.addEventListener('scroll', scroll, true)
window.addEventListener('resize', close)
onBeforeUnmount(() => {
  window.removeEventListener('pointerdown', outside)
  window.removeEventListener('scroll', scroll, true)
  window.removeEventListener('resize', close)
})
</script>
<template>
  <button
    ref="trigger"
    class="select-control"
    type="button"
    role="combobox"
    :aria-label="label"
    :disabled="disabled"
    :aria-expanded="open"
    aria-haspopup="listbox"
    :aria-controls="open ? id : undefined"
    @click="toggle"
    @keydown="key"
  >
    <span>{{ selected }}</span
    ><span class="chevron" aria-hidden="true" />
  </button>
  <Teleport to="body">
    <div v-if="open" ref="popup" class="select-menu" :style="position" tabindex="-1" @keydown="key">
      <input
        v-if="searchable"
        v-model="query"
        class="select-search"
        role="combobox"
        aria-expanded="true"
        :aria-controls="id"
        :aria-activedescendant="choices[index] ? `${id}-${index}` : undefined"
        aria-label="搜索选项"
        placeholder="搜索…"
        @input="index = 0"
      />
      <div
        :id="id"
        role="listbox"
        :aria-label="label"
        tabindex="-1"
        :aria-activedescendant="choices[index] ? `${id}-${index}` : undefined"
      >
        <button
          v-for="(option, i) in choices"
          :key="option.value"
          :id="`${id}-${i}`"
          type="button"
          role="option"
          :aria-selected="option.value === modelValue"
          :data-active="i === index"
          :data-value="option.value"
          tabindex="-1"
          @click="choose(option.value)"
        >
          <span>{{ option.label }}</span
          ><span v-if="option.value === modelValue" aria-hidden="true">✓</span>
        </button>
        <p v-if="!choices.length" class="muted">没有匹配的选项</p>
      </div>
    </div>
  </Teleport>
</template>
