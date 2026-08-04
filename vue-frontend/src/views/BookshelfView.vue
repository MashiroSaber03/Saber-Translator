<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { getV2ServerInfo } from '@/api/v2/system'
import { getBookDetail } from '@/api/bookshelf'
import BookCard from '@/components/bookshelf/BookCard.vue'
import BookSearch from '@/components/bookshelf/BookSearch.vue'
import BookModal from '@/components/bookshelf/BookModal.vue'
import BookDetailModal from '@/components/bookshelf/BookDetailModal.vue'
import TagManageModal from '@/components/bookshelf/TagManageModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductCardGrid from '@/components/product/ProductCardGrid.vue'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductHeaderAction from '@/components/product/ProductHeaderAction.vue'
import ProductHeaderMetaPill from '@/components/product/ProductHeaderMetaPill.vue'
import ProductPageHeader from '@/components/product/ProductPageHeader.vue'
import ProductThemeToggle from '@/components/product/ProductThemeToggle.vue'
import AppShell from '@/components/ui/AppShell.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { UiSelectValue } from '@/components/ui/selectTypes'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import BaseModal from '@/components/common/BaseModal.vue'
import { copyTextToClipboard } from '@/utils/clipboard'
import { showToast } from '@/utils/toast'
import { createTranslationBatch } from '@/api/v2/translation'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import { confirmProductAction } from '@/composables/useProductConfirm'

const router = useRouter()
const bookshelfStore = useBookshelfStore()
const taskCenterStore = useTaskCenterStore()

const lanUrl = ref<string>('获取中...')

const showBookModal = ref(false)
const showDetailModal = ref(false)
const showTagManageModal = ref(false)
const editingBookId = ref<string | null>(null)
const showBatchTagsModal = ref(false)
const batchTagAction = ref<'add' | 'remove'>('add')
const selectedBatchTagNames = ref(new Set<string>())
const batchBusy = ref(false)

const filteredBooks = computed(() => bookshelfStore.books)
const allTags = computed(() => bookshelfStore.tags)
const isEmpty = computed(() => (
  filteredBooks.value.length === 0
  && !bookshelfStore.searchQuery
  && bookshelfStore.selectedTagNames.length === 0
))
const selectedBookCount = computed(() => bookshelfStore.selectedBookIds.size)
const sortValue = computed(() => `${bookshelfStore.sortBy}:${bookshelfStore.sortOrder}`)
const sortOptions = [
  { label: '更新时间（新→旧）', value: 'updatedAt:desc' },
  { label: '更新时间（旧→新）', value: 'updatedAt:asc' },
  { label: '创建时间（新→旧）', value: 'createdAt:desc' },
  { label: '创建时间（旧→新）', value: 'createdAt:asc' },
  { label: '标题（升序）', value: 'title:asc' },
  { label: '标题（降序）', value: 'title:desc' },
]

function handlePageShow(event: PageTransitionEvent) {
  if (event.persisted) {
    bookshelfStore.loadBooks()
    bookshelfStore.loadTags()
    if (showDetailModal.value && bookshelfStore.currentBook) {
      openBookDetail(bookshelfStore.currentBook.id)
    }
  }
}

async function loadServerInfo(): Promise<void> {
  try {
    const response = await getV2ServerInfo()
    if (response.lanUrl) {
      lanUrl.value = response.lanUrl
    }
  } catch {
    lanUrl.value = '获取失败'
  }
}

onMounted(async () => {
  window.addEventListener('pageshow', handlePageShow)

  await Promise.all([
    bookshelfStore.loadBooks(),
    bookshelfStore.loadTags(),
    loadServerInfo(),
  ])
})

onUnmounted(() => {
  window.removeEventListener('pageshow', handlePageShow)
})

async function copyLanUrl() {
  const copied = await copyTextToClipboard(lanUrl.value)
  showToast(copied ? '局域网地址已复制！' : '复制局域网地址失败', copied ? 'success' : 'error')
}

function openCreateBookModal() {
  editingBookId.value = null
  showBookModal.value = true
}

function openEditBookModal(bookId: string) {
  editingBookId.value = bookId
  showBookModal.value = true
}

async function openBookDetail(bookId: string) {
  try {
    const book = await getBookDetail(bookId)
    bookshelfStore.updateBook(bookId, book)

    bookshelfStore.setCurrentBook(bookId)
    showDetailModal.value = true

  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : '未知错误'
    showToast(`加载书籍详情失败: ${errorMsg}`, 'error')
  }
}

function openTagManageModal() {
  showTagManageModal.value = true
}

function goToTranslate() {
  router.push('/translate')
}

function setSort(value: UiSelectValue) {
  const [by, order] = String(value).split(':') as [
    'title' | 'createdAt' | 'updatedAt',
    'asc' | 'desc',
  ]
  bookshelfStore.setSort(by, order)
}

async function translateSelectedBooks() {
  const bookIds = [...bookshelfStore.selectedBookIds]
  if (!bookIds.length || batchBusy.value) return
  batchBusy.value = true
  try {
    const result = await createTranslationBatch({ bookIds }, { mode: 'standard' })
    await Promise.all([taskCenterStore.refresh(), bookshelfStore.loadBooks()])
    showToast(
      result.skipped.length
        ? `已创建 ${result.jobIds.length} 个任务，跳过 ${result.skipped.length} 个章节`
        : `已创建 ${result.jobIds.length} 个后端翻译任务`,
      result.skipped.length ? 'warning' : 'success',
    )
    taskCenterStore.open({ batchId: result.batchId })
  } catch (error) {
    showToast(error instanceof Error ? error.message : '创建批量翻译失败', 'error')
  } finally {
    batchBusy.value = false
  }
}

async function deleteSelectedBooks() {
  const bookIds = [...bookshelfStore.selectedBookIds]
  if (!bookIds.length || batchBusy.value) return
  const confirmed = await confirmProductAction({
    title: '批量删除书籍',
    message: `确定永久删除选中的 ${bookIds.length} 本书籍吗？存在活动任务的书籍会被保留。`,
    confirmText: '批量删除',
    tone: 'danger',
  })
  if (!confirmed) return
  batchBusy.value = true
  try {
    const result = await bookshelfStore.batchDeleteBooksApi(bookIds)
    showToast(
      result.rejected.length
        ? `已删除 ${result.deleted.length} 本，${result.rejected.length} 本因任务或导入被保留`
        : `已删除 ${result.deleted.length} 本书籍`,
      result.rejected.length ? 'warning' : 'success',
    )
    if (result.rejected.length) {
      taskCenterStore.open({ bookId: result.rejected[0]?.bookId })
    }
  } catch (error) {
    showToast(error instanceof Error ? error.message : '批量删除失败', 'error')
  } finally {
    batchBusy.value = false
  }
}

function openBatchTags(action: 'add' | 'remove') {
  batchTagAction.value = action
  selectedBatchTagNames.value = new Set()
  showBatchTagsModal.value = true
}

function toggleBatchTag(tagName: string, selected: boolean) {
  const next = new Set(selectedBatchTagNames.value)
  if (selected) next.add(tagName)
  else next.delete(tagName)
  selectedBatchTagNames.value = next
}

async function applyBatchTags() {
  const bookIds = [...bookshelfStore.selectedBookIds]
  const tagNames = [...selectedBatchTagNames.value]
  if (!bookIds.length || !tagNames.length || batchBusy.value) return
  batchBusy.value = true
  try {
    const updated = await bookshelfStore.batchUpdateTagsApi(
      bookIds,
      tagNames,
      batchTagAction.value,
    )
    showToast(`已更新 ${updated} 本书籍的标签`, 'success')
    showBatchTagsModal.value = false
  } catch (error) {
    showToast(error instanceof Error ? error.message : '批量更新标签失败', 'error')
  } finally {
    batchBusy.value = false
  }
}
</script>

<template>
  <AppShell class="bookshelf-page">
    <ProductPageHeader
      variant="brand"
      logo-title="书架首页"
      nav-label="书架外部链接"
      actions-label="书架偏好操作"
    >
      <template #meta>
        <ProductHeaderMetaPill
          label="局域网访问"
          :value="lanUrl"
          title="其他设备可通过此地址访问"
        >
          <template #actions>
            <ProductHeaderAction
              variant="plain"
              title="复制地址"
              aria-label="复制局域网地址"
              label="复制"
              @click="copyLanUrl"
            />
          </template>
        </ProductHeaderMetaPill>
      </template>

      <template #nav>
        <ProductHeaderAction
          as="a"
          href="http://www.mashirosaber.top"
          target="_blank"
          rel="noopener noreferrer"
          class="bookshelf-header__tutorial-link"
          label="使用教程"
        />
        <ProductHeaderAction
          as="a"
          href="https://github.com/MashiroSaber03/Saber-Translator"
          target="_blank"
          rel="noopener noreferrer"
          class="bookshelf-header__github-link"
          aria-label="打开 GitHub 仓库"
          icon-name="github"
          icon-only
        />
      </template>

      <template #actions>
        <ProductThemeToggle
          class="bookshelf-header__theme-toggle"
        />
      </template>
    </ProductPageHeader>

    <main class="bookshelf-main">
      <div class="bookshelf-toolbar">
        <h1 class="bookshelf-toolbar__title">我的书架</h1>
        <ProductActionRow
          class="bookshelf-toolbar__actions"
          aria-label="书架主要操作"
          justify="end"
        >
          <UiButton variant="primary" @click="openCreateBookModal">
            <UiIcon name="plus" size="16" />
            <span>新建书籍</span>
          </UiButton>
          <UiButton variant="secondary" @click="openTagManageModal">
            <UiIcon name="tags" size="16" />
            <span>管理标签</span>
          </UiButton>
          <UiButton variant="secondary" @click="goToTranslate">
            <UiIcon name="languages" size="16" />
            <span>快速翻译</span>
          </UiButton>
          <UiButton
            :variant="bookshelfStore.batchMode ? 'primary' : 'secondary'"
            @click="bookshelfStore.batchMode ? bookshelfStore.exitBatchMode() : bookshelfStore.enterBatchMode()"
          >
            <UiIcon name="check" size="16" />
            <span>{{ bookshelfStore.batchMode ? '退出批量管理' : '批量管理' }}</span>
          </UiButton>
        </ProductActionRow>
      </div>

      <div class="bookshelf-query-bar">
        <BookSearch
          :tags="allTags"
          :selected-tag-names="bookshelfStore.selectedTagNames"
          @search="bookshelfStore.setSearchQuery"
          @filter-tag="bookshelfStore.toggleTagFilter"
        />
        <UiSelect
          class="bookshelf-query-bar__sort"
          aria-label="书架排序"
          :model-value="sortValue"
          :options="sortOptions"
          @change="setSort"
        />
      </div>

      <ProductActionRow
        v-if="bookshelfStore.batchMode"
        class="bookshelf-batch-bar"
        aria-label="书架批量操作"
      >
        <UiButton variant="secondary" @click="bookshelfStore.toggleSelectAll">
          {{ bookshelfStore.isAllSelected ? '取消全选' : '全选当前书籍' }}
        </UiButton>
        <span class="bookshelf-batch-bar__count">已选择 {{ selectedBookCount }} 本</span>
        <UiButton
          variant="primary"
          :disabled="selectedBookCount === 0 || batchBusy"
          @click="translateSelectedBooks"
        >
          翻译全部章节
        </UiButton>
        <UiButton
          variant="secondary"
          :disabled="selectedBookCount === 0 || batchBusy"
          @click="openBatchTags('add')"
        >
          添加标签
        </UiButton>
        <UiButton
          variant="secondary"
          :disabled="selectedBookCount === 0 || batchBusy"
          @click="openBatchTags('remove')"
        >
          移除标签
        </UiButton>
        <UiButton
          variant="danger"
          :disabled="selectedBookCount === 0 || batchBusy"
          @click="deleteSelectedBooks"
        >
          删除选中书籍
        </UiButton>
      </ProductActionRow>

      <div class="bookshelf-main__books">
        <ProductCardGrid
          v-if="filteredBooks.length > 0"
          aria-label="书籍列表"
          gap="24px"
          min-item-width="160px"
        >
          <BookCard
            v-for="book in filteredBooks"
            :key="book.id"
            :book="book"
            :tags="allTags"
            :selectable="bookshelfStore.batchMode"
            :selected="bookshelfStore.selectedBookIds.has(book.id)"
            @click="openBookDetail(book.id)"
            @select="bookshelfStore.toggleBookSelection(book.id)"
          />
        </ProductCardGrid>

        <ProductEmptyState
          v-else-if="isEmpty"
          icon-name="book-open"
          title="书架空空如也"
          description="点击&quot;新建书籍&quot;开始你的翻译之旅"
        >
          <template #actions>
            <UiButton variant="primary" @click="openCreateBookModal">
              <UiIcon name="plus" size="16" />
              <span>新建第一本书</span>
            </UiButton>
          </template>
        </ProductEmptyState>

        <ProductEmptyState
          v-else
          icon-name="search"
          title="未找到匹配的书籍"
          description="尝试调整搜索条件或标签筛选"
        />
      </div>
    </main>

    <BookModal
      v-if="showBookModal"
      :book-id="editingBookId"
      @close="showBookModal = false"
      @saved="showBookModal = false"
    />

    <BookDetailModal
      v-if="showDetailModal"
      @close="showDetailModal = false"
      @edit="openEditBookModal"
    />

    <TagManageModal
      v-if="showTagManageModal"
      @close="showTagManageModal = false"
    />

    <BaseModal
      v-model="showBatchTagsModal"
      :title="batchTagAction === 'add' ? '批量添加标签' : '批量移除标签'"
      size="small"
      :close-on-overlay="true"
      :close-on-esc="true"
    >
      <div class="bookshelf-batch-tags">
        <UiCheckbox
          v-for="tag in allTags"
          :key="tag.id || tag.name"
          :model-value="selectedBatchTagNames.has(tag.name)"
          :label="tag.name"
          @change="toggleBatchTag(tag.name, $event)"
        />
        <p v-if="!allTags.length">还没有可用标签，请先在“管理标签”中创建。</p>
      </div>
      <template #footer>
        <ProductActionRow aria-label="批量标签操作" variant="dialog">
          <UiButton variant="secondary" @click="showBatchTagsModal = false">取消</UiButton>
          <UiButton
            variant="primary"
            :disabled="selectedBatchTagNames.size === 0 || batchBusy"
            @click="applyBatchTags"
          >
            应用
          </UiButton>
        </ProductActionRow>
      </template>
    </BaseModal>
  </AppShell>
</template>

<style scoped>
.bookshelf-main {
  max-width: 1400px;
  min-height: 0;
  margin: 0 auto;
  padding: 24px;
}

.bookshelf-page {
  padding-inline: 20px;
}

.bookshelf-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 32px;
}

.bookshelf-toolbar__title {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 0;
  color: var(--color-text-default);
  font-weight: 700;
  font-size: 1.8rem;
}

.bookshelf-toolbar__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}

.bookshelf-main__books {
  min-height: 400px;
}

.bookshelf-query-bar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(210px, 280px);
  gap: 16px;
  align-items: start;
}

.bookshelf-query-bar__sort {
  width: 100%;
}

.bookshelf-batch-bar {
  position: sticky;
  top: 12px;
  z-index: var(--z-sticky);
  flex-wrap: wrap;
  margin: 16px 0;
  padding: 12px;
  background: var(--color-surface-card);
  border: 1px solid var(--color-border-default);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-soft);
}

.bookshelf-batch-bar__count {
  color: var(--color-text-supporting);
}

.bookshelf-batch-tags {
  display: grid;
  gap: 12px;
}

@media (--breakpoint-md-down) {
  .bookshelf-query-bar {
    grid-template-columns: 1fr;
  }
}
</style>
