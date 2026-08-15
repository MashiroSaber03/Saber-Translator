<script setup lang="ts">
import type { NoteData } from '@/types/insight'
import ProductScrollStack from '@/components/product/ProductScrollStack.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import NoteCard from './NoteCard.vue'

withDefaults(defineProps<{
  busyNoteIds?: string[]
  editingNoteId?: string | null
  notes: NoteData[]
}>(), {
  busyNoteIds: () => [],
  editingNoteId: null,
})

defineEmits<{
  (event: 'delete', noteId: string): void
  (event: 'edit', note: NoteData): void
  (event: 'showPage', pageNum: number): void
}>()
</script>

<template>
  <ProductScrollStack
    class="notes-list"
    role="list"
    aria-label="笔记列表"
    padding="none"
    :empty="notes.length === 0"
  >
    <template #empty>
      <ProductStatusBanner
        class="notes-list__empty-status"
        tone="neutral"
        role="note"
        icon-name="file-text"
      >
        暂无笔记
      </ProductStatusBanner>
    </template>

    <NoteCard
      v-for="note in notes"
      :key="note.id"
      :note="note"
      :busy="busyNoteIds.includes(note.id)"
      :editing="editingNoteId === note.id"
      @delete="$emit('delete', $event)"
      @edit="$emit('edit', $event)"
      @show-page="$emit('showPage', $event)"
    />
  </ProductScrollStack>
</template>

<style scoped>
.notes-list {
  flex: 0 0 auto;
  margin-bottom: 12px;
  overflow-y: visible;
}

.notes-list__empty-status {
  --product-status-banner-border: 0;
  --product-status-banner-background: transparent;
  --product-status-banner-padding: 20px;
  --product-status-banner-icon-display: none;
  --product-status-banner-body-color: var(--insight-text-muted);
  --product-status-banner-body-font-size: 14px;
  --product-status-banner-text-align: center;

  margin: 8px;
}
</style>
