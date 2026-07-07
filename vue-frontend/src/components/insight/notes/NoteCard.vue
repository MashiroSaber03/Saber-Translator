<script setup lang="ts">
import { computed } from 'vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import type { UiIconName } from '@/components/ui/iconRegistry'
import type { NoteData, NoteType } from '@/stores/insightStore'

const props = defineProps<{
  note: NoteData
}>()

const emit = defineEmits<{
  (event: 'delete', noteId: string): void
  (event: 'edit', note: NoteData): void
  (event: 'showPage', pageNum: number): void
}>()

const noteTitleText = computed(() => {
  return props.note.title || props.note.question || props.note.content || '未命名笔记'
})

const cardLabel = computed(() => `笔记：${noteTitleText.value}`)
const editLabel = computed(() => `编辑笔记：${noteTitleText.value}`)
const deleteLabel = computed(() => `删除笔记：${noteTitleText.value}`)

const tagChips = computed<ProductChipItem[]>(() => {
  return props.note.tags?.map(tag => ({
    id: tag,
    label: tag,
    tone: 'neutral',
  })) ?? []
})

const citationChips = computed<ProductChipItem[]>(() => {
  const citations = props.note.citations ?? []
  const visibleCitations = citations.slice(0, 3).map(citation => ({
    id: citation.page,
    label: `第${citation.page}页`,
    ariaLabel: `查看第 ${citation.page} 页`,
    interactive: true,
    tone: 'primary' as const,
  }))
  if (citations.length > 3) {
    visibleCitations.push({
      id: 'more-citations',
      label: `+${citations.length - 3}`,
      tone: 'neutral',
    })
  }
  return visibleCitations
})

const pageChips = computed<ProductChipItem[]>(() => {
  if (!props.note.pageNum) return []
  return [{
    id: props.note.pageNum,
    label: `第 ${props.note.pageNum} 页`,
    ariaLabel: `查看第 ${props.note.pageNum} 页`,
    iconName: 'file-text',
    interactive: true,
    tone: 'neutral',
  }]
})

function formatDate(dateStr: string): string {
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function getNoteTypeIcon(type: NoteType): UiIconName {
  return type === 'qa' ? 'message' : 'file-text'
}

function showPage(id: string | number): void {
  if (typeof id !== 'number') return
  emit('showPage', id)
}
</script>

<template>
  <ProductRecordCard
    class="note-card"
    role="listitem"
    :accent="note.type === 'qa'"
    :aria-label="cardLabel"
  >
    <template #icon>
      <UiIcon class="note-card__type-icon" :name="getNoteTypeIcon(note.type)" />
    </template>

    <template #meta>
      <span class="note-card__date">{{ formatDate(note.createdAt) }}</span>
    </template>

    <template #actions>
      <UiIconButton
        :label="editLabel"
        size="sm"
        @click.stop="emit('edit', note)"
      >
        <UiIcon name="pencil" />
      </UiIconButton>
      <UiIconButton
        :label="deleteLabel"
        variant="danger"
        size="sm"
        @click.stop="emit('delete', note.id)"
      >
        <UiIcon name="trash" />
      </UiIconButton>
    </template>

    <UiButton
      variant="toolbar"
      class="note-card__open-button"
      :aria-label="editLabel"
      @click="emit('edit', note)"
    >
      <span v-if="note.title" class="note-card__title">{{ note.title }}</span>
      <span v-if="note.type === 'qa'" class="note-card__content">
        <span class="note-card__qa-preview">Q: {{ note.question?.substring(0, 60) }}...</span>
      </span>
      <span v-else class="note-card__content">{{ note.content }}</span>

      <ProductChipList
        v-if="tagChips.length > 0"
        class="note-card__tags"
        aria-label="笔记标签"
        :items="tagChips"
      />
    </UiButton>

    <template v-if="citationChips.length > 0 || pageChips.length > 0" #footer>
      <ProductChipList
        v-if="note.type === 'qa' && citationChips.length > 0"
        aria-label="引用页码"
        :items="citationChips"
        @select="showPage"
      />

      <ProductChipList
        v-if="pageChips.length > 0"
        aria-label="关联页码"
        :items="pageChips"
        @select="showPage"
      />
    </template>
  </ProductRecordCard>
</template>

<style scoped>
.note-card__type-icon {
  color: var(--insight-text-secondary);
}

.note-card__open-button {
  display: block;
  width: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
  font: inherit;
  text-align: left;
}

.note-card__title {
  display: block;
  margin-bottom: 6px;
  overflow: hidden;
  color: var(--insight-text-primary);
  font-weight: 600;
  font-size: 14px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.note-card__content {
  display: block;
  color: var(--insight-text-secondary);
  font-size: 14px;
  line-height: 1.5;
  white-space: pre-wrap;
}

.note-card__qa-preview {
  color: var(--insight-text-secondary);
  font-size: 13px;
  font-style: italic;
}

.note-card__tags {
  margin-top: 8px;
}
</style>
