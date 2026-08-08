<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'

defineProps<{
  canUseBookConstraints: boolean
}>()

defineEmits<{
  (event: 'openGlossary'): void
  (event: 'openNonTranslate'): void
}>()
</script>

<template>
  <ProductFormSection class="book-constraint-section">
    <template #title>书籍约束</template>
    <p class="book-constraint-section__hint">
      术语表和禁翻表按单本漫画保存，不与其他书共享。
    </p>
    <ProductActionRow
      class="book-constraint-section__actions"
      justify="between"
      aria-label="书籍约束操作"
    >
      <UiButton
        variant="secondary"
        type="button"
        class="book-constraint-section__action"
        block
        :disabled="!canUseBookConstraints"
        @click="$emit('openGlossary')"
      >
        术语表
      </UiButton>
      <UiButton
        variant="secondary"
        type="button"
        class="book-constraint-section__action"
        block
        :disabled="!canUseBookConstraints"
        @click="$emit('openNonTranslate')"
      >
        禁翻表
      </UiButton>
    </ProductActionRow>
    <ProductStatusBanner
      v-if="!canUseBookConstraints"
      class="book-constraint-section__status"
      tone="neutral"
      role="note"
    >
      仅书架模式可用
    </ProductStatusBanner>
  </ProductFormSection>
</template>

<style scoped>
.book-constraint-section {
  --product-form-section-background: var(--color-surface-quiet);
  --product-form-section-border: var(--color-border-muted);
  --product-form-section-divider: transparent;
  --product-form-section-title-text: var(--color-text-heading);
  --product-form-section-margin-bottom: 0;
  --product-form-section-padding: 12px;
  --product-form-section-title-margin-bottom: 0;
  --product-form-section-title-padding-bottom: 0;
  --product-form-section-title-border-bottom: 0;
  --product-form-section-title-font-size: 15px;
  --product-form-section-title-font-weight: 700;
  --book-constraint-section-hint-text: var(--color-text-supporting);

  margin-top: 14px;
  border-radius: 12px;
}

.book-constraint-section__hint {
  margin: 0;
  margin-top: 6px;
  color: var(--book-constraint-section-hint-text);
  font-size: 12px;
  line-height: 1.4;
}

.book-constraint-section__actions {
  --ui-button-padding: 0;
  --ui-button-radius: 8px;
  --ui-button-font-size: 13px;
  --ui-button-secondary-background: var(--color-surface-base);
  --ui-button-secondary-color: var(--color-text-default);
  --ui-button-secondary-border: 1px solid var(--color-border-input);
  --ui-button-secondary-hover-background: var(--color-surface-interactive-hover);

  gap: 10px;
  margin-top: 12px;
}

.book-constraint-section__action {
  flex: 1;
  min-height: 38px;
  font-weight: 600;
}

.book-constraint-section__status {
  margin-top: 8px;
}
</style>
