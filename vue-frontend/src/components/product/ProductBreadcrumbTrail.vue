<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'

export interface ProductBreadcrumbTrailItem {
  path: string
  name: string
}

const props = withDefaults(defineProps<{
  items: ProductBreadcrumbTrailItem[]
  ariaLabel?: string
}>(), {
  ariaLabel: '当前位置',
})

const emit = defineEmits<{
  select: [path: string]
}>()

function isCurrentItem(index: number): boolean {
  return index === props.items.length - 1
}

function selectItem(path: string): void {
  emit('select', path)
}
</script>

<template>
  <nav class="product-breadcrumb-trail" :aria-label="ariaLabel">
    <template v-for="(item, index) in items" :key="item.path || 'root'">
      <span
        v-if="isCurrentItem(index)"
        class="product-breadcrumb-trail__item product-breadcrumb-trail__item--current"
        aria-current="page"
      >
        <UiIcon v-if="index === 0" name="folder-open" size="13" />
        <span>{{ item.name }}</span>
      </span>

      <UiButton
        v-else
        variant="toolbar"
        size="sm"
        type="button"
        class="product-breadcrumb-trail__item"
        :aria-label="`打开${item.name}`"
        @click="selectItem(item.path)"
      >
        <UiIcon v-if="index === 0" name="folder-open" size="13" />
        <span>{{ item.name }}</span>
      </UiButton>

      <span
        v-if="index < items.length - 1"
        class="product-breadcrumb-trail__separator"
        aria-hidden="true"
      >
        /
      </span>
    </template>
  </nav>
</template>

<style scoped>
.product-breadcrumb-trail {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 2px;
  padding: 8px 10px;
  border-radius: 6px;
  background: var(--product-breadcrumb-trail-background, var(--color-surface-quiet));
  color: var(--product-breadcrumb-trail-text, var(--color-text-default));
  font-size: 12px;
  line-height: 1.4;
}

.product-breadcrumb-trail__item {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: var(--product-breadcrumb-trail-link, var(--color-text-link));
  font: inherit;
  word-break: break-word;
}

.product-breadcrumb-trail__item:not(.product-breadcrumb-trail__item--current):hover {
  text-decoration: underline;
}

.product-breadcrumb-trail__item--current {
  color: var(--product-breadcrumb-trail-current, var(--color-text-heading));
  font-weight: 600;
}

.product-breadcrumb-trail__separator {
  margin: 0 2px;
  color: var(--product-breadcrumb-trail-separator, var(--color-text-muted));
}
</style>
