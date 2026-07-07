<script setup lang="ts">
import BaseModal from '@/components/common/BaseModal.vue'
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  title: string
  customClass?: string
  widthVariant?: 'default' | 'wide'
}>(), {
  customClass: '',
  widthVariant: 'default',
})

const emit = defineEmits<{
  close: []
}>()

const modalMaxWidth = computed(() => (
  props.widthVariant === 'wide' ? '600px' : '520px'
))

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
    width="90%"
    :max-width="modalMaxWidth"
    max-height="90vh"
    footer-padding="16px 24px"
    @update:model-value="handleUpdate"
    @close="emit('close')"
  >
    <slot />

    <template #footer>
      <slot name="footer" />
    </template>
  </BaseModal>
</template>
