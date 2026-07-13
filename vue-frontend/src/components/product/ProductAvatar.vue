<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  imageSrc?: string
  label: string
  fallbackText?: string
  size?: 'sm' | 'md' | 'lg' | 'hero'
  shape?: 'circle' | 'rounded' | 'portrait'
}>(), {
  imageSrc: '',
  fallbackText: '',
  size: 'md',
  shape: 'circle',
})

const fallbackInitial = computed(() => props.fallbackText.trim().slice(0, 1) || '角')
</script>

<template>
  <div
    class="product-avatar"
    :class="[`product-avatar--${size}`, `product-avatar--${shape}`]"
    :role="imageSrc ? undefined : 'img'"
    :aria-label="imageSrc ? undefined : label"
  >
    <img
      v-if="imageSrc"
      class="product-avatar__image"
      :src="imageSrc"
      :alt="label"
    >
    <span v-else class="product-avatar__fallback" aria-hidden="true">
      {{ fallbackInitial }}
    </span>
  </div>
</template>

<style scoped>
.product-avatar {
  --internal-product-avatar-width: 56px;
  --internal-product-avatar-height: var(--product-avatar-width, var(--internal-product-avatar-width));
  --internal-product-avatar-radius: 999px;
  --internal-product-avatar-font-size: 20px;
  --internal-product-avatar-font-weight: 600;

  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  justify-content: center;
  width: var(--product-avatar-width, var(--internal-product-avatar-width));
  height: var(--product-avatar-height, var(--internal-product-avatar-height));
  overflow: hidden;
  border-radius: var(--product-avatar-radius, var(--internal-product-avatar-radius));
  background: var(--product-avatar-background, linear-gradient(135deg, var(--color-action-brand), var(--color-action-brand-strong)));
  color: var(--product-avatar-color, var(--color-text-inverse));
}

.product-avatar--sm {
  --internal-product-avatar-width: 40px;
  --internal-product-avatar-font-size: 16px;
}

.product-avatar--md {
  --internal-product-avatar-width: 56px;
  --internal-product-avatar-font-size: 20px;
}

.product-avatar--lg {
  --internal-product-avatar-width: 64px;
  --internal-product-avatar-font-size: 24px;
}

.product-avatar--hero {
  --internal-product-avatar-width: 116px;
  --internal-product-avatar-height: 164px;
  --internal-product-avatar-font-size: 32px;
  --internal-product-avatar-font-weight: 700;
}

.product-avatar--rounded {
  --internal-product-avatar-radius: 12px;
}

.product-avatar--portrait {
  --internal-product-avatar-radius: 24px;
}

.product-avatar__image {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.product-avatar__fallback {
  font-size: var(--product-avatar-font-size, var(--internal-product-avatar-font-size));
  font-weight: var(--product-avatar-font-weight, var(--internal-product-avatar-font-weight));
  line-height: 1;
}
</style>
