<script setup lang="ts">
import { computed } from 'vue'
import ProductHeaderAction from './ProductHeaderAction.vue'
import { useSettingsStore } from '@/stores/settings'

const props = withDefaults(defineProps<{
  iconSize?: 'md' | 'lg'
}>(), {
  iconSize: 'md',
})

const settingsStore = useSettingsStore()

const label = computed(() => (
  settingsStore.theme === 'light'
    ? '切换深色模式'
    : settingsStore.theme === 'dark'
      ? '切换跟随系统'
      : '切换浅色模式'
))

const iconName = computed(() => (
  settingsStore.theme === 'light'
    ? 'sun'
    : settingsStore.theme === 'dark'
      ? 'moon'
      : 'monitor'
))

const iconPixelSize = computed(() => (
  props.iconSize === 'lg' ? 20 : 18
))
</script>

<template>
  <ProductHeaderAction
    type="button"
    class="product-theme-toggle"
    :class="`product-theme-toggle--icon-${iconSize}`"
    :title="label"
    :aria-label="label"
    :icon-name="iconName"
    :icon-size="iconPixelSize"
    icon-only
    @click="settingsStore.toggleTheme()"
  />
</template>

<style scoped>
.product-theme-toggle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

</style>
