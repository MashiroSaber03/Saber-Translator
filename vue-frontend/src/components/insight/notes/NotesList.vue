<script setup lang="ts">
import type { NoteData } from '@/stores/insightStore'
import NoteCard from './NoteCard.vue'

defineProps<{
  notes: NoteData[]
}>()

defineEmits<{
  (event: 'delete', noteId: string): void
  (event: 'edit', note: NoteData): void
  (event: 'showPage', pageNum: number): void
}>()
</script>

<template>
  <div class="notes-list">
    <div v-if="notes.length === 0" class="placeholder-text">
      暂无笔记
    </div>

    <NoteCard
      v-for="note in notes"
      :key="note.id"
      :note="note"
      @delete="$emit('delete', $event)"
      @edit="$emit('edit', $event)"
      @show-page="$emit('showPage', $event)"
    />
  </div>
</template>

<style scoped>
.notes-list {
  max-height: 300px;
  margin-bottom: 12px;
  overflow-y: auto;
}

.placeholder-text {
  padding: 20px;
  color: var(--insight-text-muted);
  font-size: 14px;
  text-align: center;
}
</style>
