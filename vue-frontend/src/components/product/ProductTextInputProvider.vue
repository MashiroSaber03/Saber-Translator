<script setup lang="ts">
import { ref, watch } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiInput from '@/components/ui/UiInput.vue'
import { useProductTextInputState } from '@/composables/useProductTextInput'

const {
  activeInput,
  submitActive,
  cancelActive,
} = useProductTextInputState()

const inputValue = ref('')

watch(activeInput, (input) => {
  inputValue.value = input?.initialValue ?? ''
})

function handleSubmit() {
  submitActive(inputValue.value)
}
</script>

<template>
  <BaseModal
    v-if="activeInput"
    :title="activeInput.title"
    size="small"
    custom-class="product-text-input-modal"
    :close-on-overlay="true"
    :close-on-esc="true"
    @close="cancelActive"
  >
    <form class="product-text-input" @submit.prevent="handleSubmit">
      <p class="product-text-input__message">{{ activeInput.message }}</p>
      <UiInput
        v-model="inputValue"
        type="text"
        autofocus
        :placeholder="activeInput.placeholder"
        :aria-label="activeInput.placeholder || activeInput.message"
      />
    </form>

    <template #footer>
      <ProductActionRow
        aria-label="文本输入操作"
        variant="dialog"
      >
        <UiButton variant="secondary" @click="cancelActive">
          {{ activeInput.cancelText }}
        </UiButton>
        <UiButton variant="primary" @click="handleSubmit">
          {{ activeInput.confirmText }}
        </UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<style scoped>
.product-text-input {
  display: grid;
  gap: 12px;
}

.product-text-input__message {
  margin: 0;
  color: var(--color-text-strong);
  font-size: 14px;
  line-height: 1.6;
}
</style>
