<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { UiSelectOption, UiSelectValue } from '@/components/ui/selectTypes'
import { getBookDetail } from '@/api/bookshelf'
import {
  createInsightAnalysisJob,
  type V2InsightAnalysisJobAccepted,
} from '@/api/v2/insight'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import type { ChapterData } from '@/types/bookshelf'

const props = defineProps<{
  modelValue: boolean
}>()

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  created: [result: V2InsightAnalysisJobAccepted]
}>()

const bookshelfStore = useBookshelfStore()
const bookId = ref('')
const scope = ref<'chapter' | 'full' | 'incremental'>('full')
const chapters = ref<ChapterData[]>([])
const selectedChapterIds = ref(new Set<string>())
const loadingChapters = ref(false)
const submitting = ref(false)
const errorMessage = ref('')
let bookRequestVersion = 0

const bookOptions = computed<UiSelectOption[]>(() => (
  bookshelfStore.books.map(book => ({ label: book.title, value: book.id }))
))
const scopeOptions: UiSelectOption[] = [
  { label: '全书分析', value: 'full' },
  { label: '只分析有变化的页面', value: 'incremental' },
  { label: '指定章节', value: 'chapter' },
]
const selectableChapters = computed(() => (
  chapters.value.filter(chapter => (chapter.imageCount || 0) > 0)
))
const canSubmit = computed(() => (
  Boolean(bookId.value)
  && (scope.value !== 'chapter' || selectedChapterIds.value.size > 0)
  && !submitting.value
))

watch(
  () => props.modelValue,
  async visible => {
    if (!visible) return
    errorMessage.value = ''
    if (!bookshelfStore.books.length) {
      try {
        await bookshelfStore.loadBooks()
      } catch (error) {
        if (props.modelValue) {
          errorMessage.value = error instanceof Error ? error.message : '读取书籍列表失败'
        }
      }
    }
  },
  { immediate: true },
)

function close() {
  emit('update:modelValue', false)
}

async function selectBook(value: UiSelectValue) {
  const selectedBookId = String(value)
  const requestVersion = ++bookRequestVersion
  bookId.value = selectedBookId
  chapters.value = []
  selectedChapterIds.value = new Set()
  errorMessage.value = ''
  if (!selectedBookId) return
  loadingChapters.value = true
  try {
    const book = await getBookDetail(selectedBookId)
    if (
      requestVersion === bookRequestVersion
      && bookId.value === selectedBookId
    ) {
      chapters.value = book.chapters || []
    }
  } catch (error) {
    if (requestVersion === bookRequestVersion) {
      errorMessage.value = error instanceof Error ? error.message : '读取书籍章节失败'
    }
  } finally {
    if (requestVersion === bookRequestVersion) {
      loadingChapters.value = false
    }
  }
}

function selectScope(value: UiSelectValue) {
  scope.value = String(value) as typeof scope.value
  errorMessage.value = ''
}

function toggleChapter(chapterId: string, selected: boolean) {
  const next = new Set(selectedChapterIds.value)
  if (selected) next.add(chapterId)
  else next.delete(chapterId)
  selectedChapterIds.value = next
}

function toggleAllChapters() {
  selectedChapterIds.value = (
    selectedChapterIds.value.size === selectableChapters.value.length
      ? new Set()
      : new Set(selectableChapters.value.map(chapter => chapter.id))
  )
}

async function submit() {
  if (!canSubmit.value) return
  submitting.value = true
  errorMessage.value = ''
  try {
    const result = await createInsightAnalysisJob({
      bookId: bookId.value,
      scope: scope.value,
      ...(scope.value === 'chapter'
        ? { chapterIds: [...selectedChapterIds.value] }
        : {}),
    })
    emit('created', result)
    close()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '创建批量分析任务失败'
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <BaseModal
    :model-value="modelValue"
    title="新建批量分析"
    size="small"
    :close-on-overlay="true"
    :close-on-esc="true"
    @close="close"
  >
    <div class="task-analysis">
      <UiField
        label="书籍"
        variant="settings"
        control-id="task-analysis-book"
      >
        <UiSelect
          id="task-analysis-book"
          :model-value="bookId"
          :options="bookOptions"
          placeholder="选择要分析的书籍"
          @change="selectBook"
        />
      </UiField>

      <UiField
        label="分析范围"
        variant="settings"
        control-id="task-analysis-scope"
      >
        <UiSelect
          id="task-analysis-scope"
          :model-value="scope"
          :options="scopeOptions"
          @change="selectScope"
        />
      </UiField>

      <section v-if="scope === 'chapter'" class="task-analysis__chapters">
        <div class="task-analysis__chapter-header">
          <strong>选择章节</strong>
          <UiButton
            size="xs"
            variant="ghost"
            :disabled="!selectableChapters.length"
            @click="toggleAllChapters"
          >
            {{ selectedChapterIds.size === selectableChapters.length ? '清空' : '全选' }}
          </UiButton>
        </div>
        <p v-if="loadingChapters">正在读取章节…</p>
        <p v-else-if="bookId && !selectableChapters.length">这本书还没有可分析的页面。</p>
        <div v-else class="task-analysis__chapter-list">
          <UiCheckbox
            v-for="chapter in selectableChapters"
            :key="chapter.id"
            :model-value="selectedChapterIds.has(chapter.id)"
            :label="chapter.title"
            :description="`${chapter.imageCount || 0} 页`"
            @change="selected => toggleChapter(chapter.id, selected)"
          />
        </div>
      </section>

      <ProductStatusBanner v-if="errorMessage" tone="danger">
        {{ errorMessage }}
      </ProductStatusBanner>
    </div>

    <template #footer>
      <ProductActionRow aria-label="批量分析操作" variant="dialog">
        <UiButton variant="secondary" @click="close">取消</UiButton>
        <UiButton variant="primary" :disabled="!canSubmit" @click="submit">
          {{ submitting ? '创建中…' : '加入任务队列' }}
        </UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<style scoped>
.task-analysis {
  display: grid;
  gap: 16px;
}

.task-analysis__chapters {
  display: grid;
  gap: 10px;
  max-height: 280px;
  padding: 12px;
  overflow: auto;
  background: var(--color-surface-muted);
  border-radius: 8px;
}

.task-analysis__chapters p {
  margin: 0;
  color: var(--color-text-muted);
  font-size: 13px;
}

.task-analysis__chapter-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.task-analysis__chapter-list {
  display: grid;
  gap: 10px;
}
</style>
