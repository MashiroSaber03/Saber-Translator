<script setup lang="ts">
import { ref } from 'vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'

const props = withDefaults(defineProps<{
  inputId: string
  label: string
  accept?: string
  multiple?: boolean
  disabled?: boolean
}>(), {
  accept: '',
  multiple: false,
  disabled: false,
})

const emit = defineEmits<{
  select: [files: File[]]
}>()

const isDragging = ref(false)
const dropzoneInputRef = ref<InstanceType<typeof UiFileInput> | null>(null)

function emitFiles(fileList: FileList | File[] | null | undefined): void {
  if (props.disabled || !fileList?.length) return
  emit('select', Array.from(fileList))
}

function handleFilesChange(files: File[]): void {
  emitFiles(files)
  dropzoneInputRef.value?.clear()
}

function handleDrop(event: DragEvent): void {
  event.preventDefault()
  isDragging.value = false
  emitFiles(event.dataTransfer?.files)
}

function handleDragEnter(event: DragEvent): void {
  event.preventDefault()
  if (!props.disabled) {
    isDragging.value = true
  }
}

function handleDragOver(event: DragEvent): void {
  event.preventDefault()
  if (!props.disabled) {
    isDragging.value = true
  }
}

function handleDragLeave(event: DragEvent): void {
  event.preventDefault()
  isDragging.value = false
}
</script>

<template>
  <label
    class="product-file-dropzone"
    :class="{
      'product-file-dropzone--disabled': disabled,
      'product-file-dropzone--dragging': isDragging,
    }"
    :for="inputId"
    :aria-label="label"
    @dragenter="handleDragEnter"
    @dragover="handleDragOver"
    @dragleave="handleDragLeave"
    @drop="handleDrop"
  >
    <UiFileInput
      ref="dropzoneInputRef"
      :id="inputId"
      class="product-file-dropzone__input"
      :accept="accept"
      :multiple="multiple"
      :disabled="disabled"
      :aria-label="label"
      @files-change="handleFilesChange"
    />
    <span class="product-file-dropzone__content">
      <slot :is-dragging="isDragging" />
    </span>
  </label>
</template>

<style scoped>
.product-file-dropzone {
  display: block;
  position: relative;
  padding: var(--product-file-dropzone-padding, 16px);
  border: 2px dashed var(--product-file-dropzone-border, var(--color-border-muted));
  border-radius: var(--product-file-dropzone-radius, 8px);
  background: var(--product-file-dropzone-background, var(--color-surface-card));
  color: var(--product-file-dropzone-color, var(--color-text-supporting));
  cursor: pointer;
  text-align: center;
  transition: border-color 0.2s ease, background-color 0.2s ease, box-shadow 0.2s ease;
}

.product-file-dropzone:hover,
.product-file-dropzone:focus-within,
.product-file-dropzone--dragging {
  border-color: var(--product-file-dropzone-border-hover, var(--color-action-primary));
  background: var(--product-file-dropzone-background-hover, var(--color-surface-interactive-hover));
}

.product-file-dropzone:focus-within {
  box-shadow: 0 0 0 3px var(--color-focus-brand-subtle);
}

.product-file-dropzone--disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.product-file-dropzone__input {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  cursor: inherit;
}

.product-file-dropzone__content {
  display: block;
  pointer-events: none;
}
</style>
