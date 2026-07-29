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
import { copyTextToClipboard } from '@/utils/clipboard'
import { showToast } from '@/utils/toast'

const router = useRouter()
const bookshelfStore = useBookshelfStore()

const lanUrl = ref<string>('获取中...')

const showBookModal = ref(false)
const showDetailModal = ref(false)
const showTagManageModal = ref(false)
const editingBookId = ref<string | null>(null)

const filteredBooks = computed(() => bookshelfStore.filteredBooks)
const allTags = computed(() => bookshelfStore.tags)
const isEmpty = computed(() => filteredBooks.value.length === 0 && !bookshelfStore.searchQuery)

function handlePageShow(event: PageTransitionEvent) {
  if (event.persisted) {
    bookshelfStore.loadBooks()
    bookshelfStore.loadTags()
    if (showDetailModal.value && bookshelfStore.currentBook) {
      openBookDetail(bookshelfStore.currentBook.id)
    }
  }
}

onMounted(async () => {
  window.addEventListener('pageshow', handlePageShow)

  await Promise.all([
    bookshelfStore.loadBooks(),
    bookshelfStore.loadTags(),
  ])

  try {
    const response = await getV2ServerInfo()
    if (response.lanUrl) {
      lanUrl.value = response.lanUrl
    }
  } catch {
    lanUrl.value = '获取失败'
  }

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
    const response = await getBookDetail(bookId)

    if (!response.success) {
      throw new Error(response.error || '加载失败')
    }

    if (response.book) {
      bookshelfStore.updateBook(bookId, response.book)
    }

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
        </ProductActionRow>
      </div>

      <BookSearch
        :tags="allTags"
        :selected-tag-names="bookshelfStore.selectedTagNames"
        @search="bookshelfStore.setSearchQuery"
        @filter-tag="bookshelfStore.toggleTagFilter"
      />

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
            @click="openBookDetail(book.id)"
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
</style>
