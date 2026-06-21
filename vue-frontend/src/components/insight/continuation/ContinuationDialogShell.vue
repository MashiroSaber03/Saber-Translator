<script setup lang="ts">
import './ContinuationDialogShell.global.styles.css'
import BaseModal from '@/components/common/BaseModal.vue'

withDefaults(defineProps<{
  title: string
  customClass?: string
}>(), {
  customClass: '',
})

const emit = defineEmits<{
  close: []
}>()

function handleUpdate(value: boolean): void {
  if (!value) {
    emit('close')
  }
}
</script>

<template>
  <BaseModal
    :model-value="true"
    :title="title"
    size="medium"
    :custom-class="['continuation-dialog-modal', customClass].filter(Boolean).join(' ')"
    @update:model-value="handleUpdate"
    @close="emit('close')"
  >
    <slot />

    <template #footer>
      <slot name="footer" />
    </template>
  </BaseModal>
</template>


