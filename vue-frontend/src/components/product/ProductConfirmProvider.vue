<script setup lang="ts">
import ConfirmModal from '@/components/common/ConfirmModal.vue'
import { useProductConfirmState } from '@/composables/useProductConfirm'

const {
  activeConfirm,
  confirmActive,
  cancelActive,
} = useProductConfirmState()
</script>

<template>
  <ConfirmModal
    v-if="activeConfirm"
    :title="activeConfirm.title"
    :message="activeConfirm.message"
    :confirm-text="activeConfirm.confirmText"
    :cancel-text="activeConfirm.cancelText"
    :confirm-type="activeConfirm.tone"
    @confirm="confirmActive"
    @cancel="cancelActive"
  />
</template>
