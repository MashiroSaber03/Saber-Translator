<script setup lang="ts">
import BaseModal from './BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'

interface Props {
  message: string
  title?: string
  confirmText?: string
  cancelText?: string
  confirmType?: 'primary' | 'danger'
}

withDefaults(defineProps<Props>(), {
  title: '确认操作',
  confirmText: '确定',
  cancelText: '取消',
  confirmType: 'primary'
})

const emit = defineEmits<{
  confirm: []
  cancel: []
}>()

function handleConfirm(): void {
  emit('confirm')
}

function handleCancel(): void {
  emit('cancel')
}
</script>

<template>
  <BaseModal
    :title="title"
    size="small"
    custom-class="confirm-modal"
    body-text-align="center"
    :close-on-overlay="true"
    :close-on-esc="true"
    @close="handleCancel"
  >
    <div class="confirm-modal-body">
      <p class="confirm-message">{{ message }}</p>
    </div>

    <template #footer>
      <UiButton
        variant="secondary"
        @click="handleCancel"
      >
        {{ cancelText }}
      </UiButton>
      <UiButton
        :variant="confirmType"
        @click="handleConfirm"
      >
        {{ confirmText }}
      </UiButton>
    </template>
  </BaseModal>
</template>

<style scoped>
.confirm-message {
  margin: 0;
  font-size: 14px;
  line-height: 1.6;
  color: var(--color-text-strong);
}
</style>
