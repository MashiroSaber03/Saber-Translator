<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import type { BookData } from '@/types/api'

const props = withDefaults(defineProps<{
  book: BookData
  chapterCount: number
  formatDate: (date?: string) => string
  getTagColor: (tagName: string) => string
  characterStudioAllowed?: boolean
  insightAllowed?: boolean
}>(), {
  characterStudioAllowed: true,
  insightAllowed: true,
})

const emit = defineEmits<{
  (event: 'addTag'): void
  (event: 'edit'): void
  (event: 'delete'): void
  (event: 'characterStudio'): void
  (event: 'insight'): void
  (event: 'removeTag', tagName: string): void
}>()

const coverFailed = ref(false)
const hasVisibleCover = computed(() => {
  return Boolean(props.book.cover && props.book.cover.length > 0 && !coverFailed.value)
})
const tagItems = computed<ProductChipItem[]>(() => props.book.tags?.map(tag => {
  const tagColor = props.getTagColor(tag)

  return {
    id: tag,
    label: tag,
    ariaLabel: `移除标签 ${tag}`,
    iconName: 'x',
    interactive: true,
    tone: 'custom',
    backgroundColor: tagColor,
    borderColor: tagColor,
    textColor: 'var(--color-text-inverse)',
  }
}) ?? [])

watch(() => props.book.cover, () => {
  coverFailed.value = false
})

function removeTag(tagId: string | number): void {
  emit('removeTag', String(tagId))
}

function handleCoverError(): void {
  coverFailed.value = true
}
</script>

<template>
  <div class="book-detail-summary">
    <div class="book-detail-summary__cover">
      <img
        v-if="hasVisibleCover"
        class="book-detail-summary__cover-image"
        :src="book.cover"
        :alt="`${book.title} 封面`"
        @error="handleCoverError"
      >
      <div v-else class="book-detail-summary__cover-placeholder" aria-label="无封面">📖</div>
    </div>
    <div class="book-detail-summary__meta">
      <h3 class="book-detail-summary__title">{{ book.title }}</h3>
      <div class="book-detail-summary__meta-item">
        <span class="book-detail-summary__meta-label">标签：</span>
        <ProductChipList
          v-if="tagItems.length > 0"
          class="book-detail-summary__tags"
          aria-label="书籍详情标签"
          :items="tagItems"
          @select="removeTag"
        />
        <span v-else class="book-detail-summary__no-tags-hint">暂无标签</span>
        <UiIconButton
          class="book-detail-summary__add-tag"
          label="添加标签"
          variant="soft"
          size="sm"
          @click="emit('addTag')"
        >
          <span aria-hidden="true">+</span>
        </UiIconButton>
      </div>
      <p class="book-detail-summary__meta-item"><span class="book-detail-summary__meta-label">章节数：</span><span>{{ chapterCount }}</span></p>
      <p class="book-detail-summary__meta-item"><span class="book-detail-summary__meta-label">创建时间：</span><span>{{ formatDate(book.createdAt) }}</span></p>
      <p class="book-detail-summary__meta-item"><span class="book-detail-summary__meta-label">最后更新：</span><span>{{ formatDate(book.updatedAt) }}</span></p>
      <div class="book-detail-summary__actions">
        <UiButton v-if="insightAllowed !== false" size="sm" variant="primary" @click="emit('insight')">
          <span aria-hidden="true">●</span>
          漫画分析
        </UiButton>
        <UiButton
          v-if="characterStudioAllowed !== false"
          size="sm"
          variant="secondary"
          @click="emit('characterStudio')"
        >
          角色工坊
        </UiButton>
        <UiButton size="sm" variant="secondary" @click="emit('edit')">编辑书籍</UiButton>
        <UiButton size="sm" variant="danger" @click="emit('delete')">删除书籍</UiButton>
      </div>
    </div>
  </div>
</template>

<style scoped>
.book-detail-summary {
  --book-detail-summary-cover-shadow: var(--shadow-medium);

  display: flex;
  align-items: flex-start;
  gap: 24px;
}

.book-detail-summary__cover {
  flex-shrink: 0;
  width: 140px;
  overflow: hidden;
  border-radius: 12px;
  aspect-ratio: 3 / 4;
  background: linear-gradient(135deg, var(--color-action-brand) 0%, var(--color-action-brand-strong) 100%);
  box-shadow: 0 8px 24px var(--book-detail-summary-cover-shadow);
}

.book-detail-summary__cover-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.book-detail-summary__cover-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  color: var(--color-text-inverse);
  font-size: 3rem;
  line-height: 1;
}

.book-detail-summary__meta {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-width: 0;
}

.book-detail-summary__title {
  margin: 0 0 16px;
  color: var(--color-text-default);
  font-weight: 600;
  font-size: 1.3rem;
  line-height: 1.3;
  word-break: break-word;
}

.book-detail-summary__meta-item {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 6px 0;
  color: var(--color-text-supporting);
  font-size: 0.9rem;
}

.book-detail-summary__meta-label {
  flex-shrink: 0;
  min-width: 70px;
  color: var(--color-text-default);
  font-weight: 500;
}

.book-detail-summary__tags {
  --product-chip-list-text: var(--color-text-supporting);
}

.book-detail-summary__no-tags-hint {
  color: var(--color-text-supporting);
  font-style: italic;
}

.book-detail-summary__add-tag {
  flex: 0 0 auto;
  width: 22px;
  height: 22px;
  margin-left: 6px;
  border: 1px dashed var(--color-border-muted);
  border-radius: 50%;
  background: transparent;
  color: var(--color-text-supporting);
  font-size: 0.9rem;
}

.book-detail-summary__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 16px;
}

@media (--breakpoint-sm-down) {
  .book-detail-summary {
    flex-direction: column;
    gap: 16px;
  }

  .book-detail-summary__cover {
    align-self: center;
  }
}
</style>
