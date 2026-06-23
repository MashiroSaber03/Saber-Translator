<script setup lang="ts">
import type { TimelineCharacter } from './timelineTypes'

defineProps<{
  characters: TimelineCharacter[]
}>()

defineEmits<{
  (event: 'showPage', pageNum: number): void
}>()
</script>

<template>
  <div class="characters-section">
    <h4>👥 主要角色</h4>
    <div class="characters-grid">
      <div
        v-for="character in characters"
        :key="character.name"
        class="character-card"
        role="button"
        tabindex="0"
        @click="$emit('showPage', character.first_appearance)"
        @keydown.enter="$emit('showPage', character.first_appearance)"
        @keydown.space.prevent="$emit('showPage', character.first_appearance)"
      >
        <span class="character-name">{{ character.name }}</span>
        <span class="character-desc">{{ character.description }}</span>
        <span class="first-appear">首次出现：第 {{ character.first_appearance }} 页</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.characters-section {
  margin-bottom: 20px;
}

.characters-section h4 {
  font-size: 14px;
  margin: 0 0 12px;
}

.characters-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}

.character-card {
  display: block;
  text-align: left;
  background: var(--insight-surface-secondary);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 2px 8px var(--timeline-panel-character-shadow);
  cursor: pointer;
  transition: transform 0.2s;
}

.character-card:hover {
  transform: translateY(-2px);
}

.character-name,
.character-desc,
.first-appear {
  display: block;
}

.character-name {
  font-weight: 600;
  font-size: 15px;
  color: var(--insight-text-primary);
  margin-bottom: 8px;
}

.character-desc {
  font-size: 13px;
  color: var(--insight-text-secondary);
  line-height: 1.5;
  margin: 0 0 8px;
}

.first-appear {
  width: fit-content;
  font-size: 12px;
  color: var(--insight-text-muted);
  background: var(--insight-surface-page);
  padding: 3px 8px;
  border-radius: 10px;
}
</style>
