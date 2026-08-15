<script setup lang="ts">
import BaseModal from '@/components/common/BaseModal.vue'
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  title: string
  customClass?: string
  widthVariant?: 'default' | 'wide'
  dismissible?: boolean
}>(), {
  customClass: '',
  widthVariant: 'default',
  dismissible: true,
})

const emit = defineEmits<{
  close: []
}>()

const modalMaxWidth = computed(() => (
  props.widthVariant === 'wide' ? '600px' : '520px'
))

function handleUpdate(value: boolean): void {
  if (!value && props.dismissible) {
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
    :show-close-button="dismissible"
    :close-on-overlay="dismissible"
    :close-on-esc="dismissible"
    footer-padding="16px 24px"
    @update:model-value="handleUpdate"
  >
    <slot />

    <template #footer>
      <slot name="footer" />
    </template>
  </BaseModal>
</template>
