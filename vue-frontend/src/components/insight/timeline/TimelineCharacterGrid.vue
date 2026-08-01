<script setup lang="ts">
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import type { TimelineCharacter } from '@/types/insight'

defineProps<{
  characters: TimelineCharacter[]
}>()

defineEmits<{
  (event: 'showPage', pageNum: number): void
}>()

function firstAppearanceItems(character: TimelineCharacter): ProductChipItem[] {
  return [
    {
      id: `${character.name}-first-appearance`,
      label: `首次出现：第 ${character.first_appearance} 页`,
      tone: 'neutral',
    },
  ]
}
</script>

<template>
  <div class="timeline-character-grid__section">
    <h4 class="timeline-character-grid__title">
      <UiIcon name="users" size="15" />
      <span>主要角色</span>
    </h4>
    <div class="timeline-character-grid__grid">
      <ProductRecordCard
        v-for="character in characters"
        :key="character.name"
        as="button"
        class="timeline-character-grid__card"
        :aria-label="`查看角色${character.name}首次出现的第 ${character.first_appearance} 页`"
        @click="$emit('showPage', character.first_appearance)"
      >
        <span class="timeline-character-grid__name">{{ character.name }}</span>
        <span class="timeline-character-grid__description">{{ character.description }}</span>
        <template #footer>
          <ProductChipList
            aria-label="角色出现信息"
            :items="firstAppearanceItems(character)"
          />
        </template>
      </ProductRecordCard>
    </div>
  </div>
</template>

<style scoped>
.timeline-character-grid__section {
  margin-bottom: 20px;
}

.timeline-character-grid__title {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  margin: 0 0 12px;
}

.timeline-character-grid__grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 280px), 1fr));
  gap: 16px;
}

.timeline-character-grid__card {
  --product-record-card-background: var(--insight-surface-secondary);
  --product-record-card-border: transparent;
  --product-record-card-radius: 12px;
  --product-record-card-padding: 16px;
  --product-record-card-shadow: 0 2px 8px var(--timeline-panel-character-shadow);
}

.timeline-character-grid__card:hover {
  transform: translateY(-2px);
}

.timeline-character-grid__name,
.timeline-character-grid__description {
  display: block;
}

.timeline-character-grid__name {
  font-weight: 600;
  font-size: 15px;
  color: var(--insight-text-primary);
  margin-bottom: 8px;
}

.timeline-character-grid__description {
  font-size: 13px;
  color: var(--insight-text-secondary);
  line-height: 1.5;
  margin: 0 0 8px;
}
</style>
