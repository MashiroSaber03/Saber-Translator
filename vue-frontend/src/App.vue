<script setup lang="ts">
import { onMounted } from 'vue'
import { RouterView } from 'vue-router'
import { useSettingsStore } from '@/stores/settings'
import ToastNotification from '@/components/common/ToastNotification.vue'
import ProductConfirmProvider from '@/components/product/ProductConfirmProvider.vue'
import ProductTextInputProvider from '@/components/product/ProductTextInputProvider.vue'

const settingsStore = useSettingsStore()

onMounted(() => {
  settingsStore.initSettings()
})
</script>

<template>
  <RouterView />
  <ProductConfirmProvider />
  <ProductTextInputProvider />
  <ToastNotification />
</template>
