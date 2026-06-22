<script setup lang="ts">
import BaseModal from '@/components/common/BaseModal.vue'
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  title: string
  customClass?: string
}>(), {
  customClass: '',
})

const emit = defineEmits<{
  close: []
}>()

const modalStyle = computed(() => ({
  width: '90%',
  maxWidth: props.customClass.includes('continuation-dialog-modal--wide') ? '600px' : '520px',
  maxHeight: '90vh',
  borderRadius: '12px',
  '--ui-dialog-actions-padding': '16px 24px',
}))

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
    body-padding="spacious"
    :custom-style="modalStyle"
    @update:model-value="handleUpdate"
    @close="emit('close')"
  >
    <slot />

    <template #footer>
      <slot name="footer" />
    </template>
  </BaseModal>
</template>
