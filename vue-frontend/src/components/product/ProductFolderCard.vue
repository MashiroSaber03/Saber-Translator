<script setup lang="ts">
import { computed } from 'vue'

import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import UiIcon from '@/components/ui/UiIcon.vue'

const props = withDefaults(defineProps<{
  ariaLabel?: string
  count: number
  countAriaLabel?: string
  countId?: string | number
  countSuffix?: string
  folderName: string
}>(), {
  ariaLabel: undefined,
  countAriaLabel: undefined,
  countId: undefined,
  countSuffix: '张',
})

const emit = defineEmits<{
  select: []
}>()

const resolvedAriaLabel = computed(() => props.ariaLabel ?? `打开文件夹 ${props.folderName}`)
const resolvedCountAriaLabel = computed(() => props.countAriaLabel ?? `${props.folderName} 文件夹图片数量`)
const countItems = computed<ProductChipItem[]>(() => [
  {
    id: props.countId ?? `${props.folderName}-count`,
    label: `${props.count} ${props.countSuffix}`,
    tone: 'neutral',
  },
])
</script>

<template>
  <ProductRecordCard
    as="button"
    class="product-folder-card"
    :aria-label="resolvedAriaLabel"
    @click="emit('select')"
  >
    <template #icon>
      <UiIcon class="product-folder-card__icon" name="folder-open" size="14" />
    </template>

    <div class="product-folder-card__info">
      <span class="product-folder-card__name" :title="folderName">{{ folderName }}</span>
      <ProductChipList
        :aria-label="resolvedCountAriaLabel"
        :items="countItems"
      />
    </div>
  </ProductRecordCard>
</template>

<style scoped>
.product-folder-card {
  flex-shrink: 0;
}

.product-folder-card__icon {
  color: var(--product-folder-card-icon-text, var(--color-text-supporting));
}

.product-folder-card__info {
  display: flex;
  flex-direction: column;
  gap: 6px;
  width: 100%;
  min-width: 0;
}

.product-folder-card__name {
  color: var(--product-folder-card-name-text, var(--color-text-heading));
  font-size: var(--product-folder-card-name-font-size, 13px);
  font-weight: 500;
  line-height: 1.4;
  overflow-wrap: anywhere;
}
</style>
