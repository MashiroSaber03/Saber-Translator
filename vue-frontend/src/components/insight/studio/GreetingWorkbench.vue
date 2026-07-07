<template>
  <div class="greeting-workbench">
    <div class="greeting-workbench__hero-block">
      <div class="greeting-workbench__section-head">
        <div class="greeting-workbench__section-copy">
          <h3 class="greeting-workbench__section-title">主问候</h3>
          <p class="greeting-workbench__section-description">角色进入对话时最先展示的开场白。它决定了语气、场景和第一印象。</p>
        </div>
        <ProductActionRow aria-label="主问候操作">
          <UiButton variant="secondary" :disabled="generating" @click="$emit('generate')">
            {{ generating ? '生成中...' : '批量生成' }}
          </UiButton>
        </ProductActionRow>
      </div>
      <UiTextarea
        class="greeting-workbench__textarea"
        :model-value="firstMessage"
        variant="studio"
        size="lg"
        rows="6"
        @update:model-value="$emit('update:firstMessage', $event)"
      />
    </div>

    <div class="greeting-workbench__list-block">
      <div class="greeting-workbench__section-head">
        <div class="greeting-workbench__section-copy">
          <h3 class="greeting-workbench__section-title">备用问候</h3>
          <p class="greeting-workbench__section-description">维护多种开场方式，可随时采用为主问候或继续打磨。</p>
        </div>
        <ProductActionRow aria-label="备用问候操作">
          <UiButton variant="primary" @click="$emit('add')">添加备用问候</UiButton>
        </ProductActionRow>
      </div>

      <ProductEmptyState
        v-if="alternates.length === 0"
        description="建议生成 3-5 条不同场景的开场白。"
        icon-name="message"
        role="note"
        size="compact"
        title="还没有备用问候"
      />

      <div v-else class="greeting-workbench__alternate-list">
        <ProductRecordCard v-for="(item, index) in alternates" :key="`alt-${index}`" class="greeting-workbench__alternate-card">
          <div class="greeting-workbench__alternate-head">
            <div class="greeting-workbench__alternate-title">
              <span class="greeting-workbench__index-chip">#{{ index + 1 }}</span>
              <strong class="greeting-workbench__alternate-name">备用问候</strong>
            </div>
            <ProductActionRow aria-label="备用问候条目操作">
              <UiButton variant="secondary" @click="$emit('promote', item)" size="sm">设为主问候</UiButton>
              <UiButton variant="secondary" :disabled="index === 0" @click="$emit('move', index, -1)" size="sm">上移</UiButton>
              <UiButton variant="secondary" :disabled="index === alternates.length - 1" @click="$emit('move', index, 1)" size="sm">下移</UiButton>
              <UiButton variant="secondary" tone="danger" @click="$emit('remove', index)" size="sm">删除</UiButton>
            </ProductActionRow>
          </div>
          <UiTextarea
            class="greeting-workbench__textarea"
            :model-value="item"
            variant="studio"
            size="lg"
            rows="4"
            @update:model-value="$emit('update:item', index, $event)"
          />
        </ProductRecordCard>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiButton from '@/components/ui/UiButton.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
defineProps<{
  firstMessage: string
  alternates: string[]
  generating: boolean
}>()

defineEmits<{
  (e: 'update:firstMessage', value: string): void
  (e: 'update:item', index: number, value: string): void
  (e: 'add'): void
  (e: 'remove', index: number): void
  (e: 'move', index: number, direction: -1 | 1): void
  (e: 'promote', value: string): void
  (e: 'generate'): void
}>()
</script>

<style scoped>
.greeting-workbench {
  --greeting-workbench-block-border: var(--studio-border-default);
  --greeting-workbench-block-background: color-mix(in srgb, var(--color-surface-card) 82%, transparent);
  --greeting-workbench-alternate-card-background: var(--studio-surface-soft);

  display: flex;
  flex-direction: column;
  gap: 18px;
}

.greeting-workbench__hero-block,
.greeting-workbench__list-block {
  border-radius: 20px;
  padding: 18px;
  background: var(--greeting-workbench-block-background);
  border: 1px solid var(--greeting-workbench-block-border);
}

.greeting-workbench__section-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.greeting-workbench__alternate-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}

.greeting-workbench__section-copy,
.greeting-workbench__alternate-title {
  min-width: 0;
}

.greeting-workbench__section-title {
  margin: 0;
}

.greeting-workbench__section-description {
  margin: 6px 0 0;
  color: var(--studio-text-muted);
  font-size: 13px;
  line-height: 1.6;
}

.greeting-workbench__textarea {
  margin-top: 14px;
}

.greeting-workbench__alternate-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-top: 14px;
}

.greeting-workbench__alternate-card {
  --product-record-card-background: var(--greeting-workbench-alternate-card-background);
  --product-record-card-border: var(--studio-border-default);
  --product-record-card-radius: 18px;
  --product-record-card-padding: 14px;
  --product-record-card-gap: 14px;
}

.greeting-workbench__alternate-title {
  display: flex;
  gap: 8px;
  align-items: center;
}

.greeting-workbench__index-chip {
  border-radius: 999px;
  padding: 3px 8px;
  background: var(--studio-surface-tint);
  color: var(--color-text-link-strong);
  font-size: 11px;
}

@media (--breakpoint-lg-down) {
  .greeting-workbench__section-head,
  .greeting-workbench__alternate-head {
    flex-direction: column;
  }
}
</style>
