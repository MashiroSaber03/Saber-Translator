<script setup lang="ts">
import UiIcon from '@/components/ui/UiIcon.vue'
import type { UiIconName } from '@/components/ui/iconRegistry'

type ProductSectionHeaderSize = 'md' | 'sm'

const props = withDefaults(defineProps<{
  title: string
  description?: string
  headingLevel?: 2 | 3 | 4 | 5
  iconName?: UiIconName
  size?: ProductSectionHeaderSize
}>(), {
  description: '',
  headingLevel: 4,
  iconName: undefined,
  size: 'md',
})
</script>

<template>
  <header
    class="product-section-header"
    :class="`product-section-header--${props.size}`"
  >
    <div class="product-section-header__copy">
      <component :is="`h${headingLevel}`" class="product-section-header__title">
        <span v-if="$slots.icon" class="product-section-header__icon-text" aria-hidden="true"><slot name="icon" /></span>
        <UiIcon v-else-if="iconName" :name="iconName" size="16" />
        <span class="product-section-header__title-text">{{ title }}</span>
      </component>
      <p v-if="description" class="product-section-header__description">
        {{ description }}
      </p>
    </div>
    <div v-if="$slots.actions" class="product-section-header__actions">
      <slot name="actions" />
    </div>
  </header>
</template>

<style scoped>
.product-section-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  min-width: 0;
  margin-bottom: 16px;
}

.product-section-header__copy {
  flex: 1 1 auto;
  min-width: 0;
}

.product-section-header__title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  margin: 0;
  color: var(--product-section-header-title-text, var(--color-text-default));
  font-size: 16px;
  font-weight: 600;
  letter-spacing: 0;
  line-height: 1.3;
  overflow-wrap: anywhere;
}

.product-section-header__title-text {
  min-width: 0;
  overflow-wrap: anywhere;
}

.product-section-header__icon-text {
  flex: 0 0 auto;
  font-size: var(--product-section-header-icon-font-size, 1em);
  line-height: 1;
}

.product-section-header--sm {
  align-items: center;
  margin-bottom: 12px;
}

.product-section-header--sm .product-section-header__title {
  font-size: 14px;
}

.product-section-header__description {
  margin: 4px 0 0;
  color: var(--color-text-supporting, var(--color-text-secondary));
  font-size: 12px;
  line-height: 1.45;
}

.product-section-header__actions {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}

@media (--breakpoint-sm-down) {
  .product-section-header {
    flex-direction: column;
    align-items: stretch;
  }

  .product-section-header__actions {
    justify-content: flex-start;
  }
}
</style>
