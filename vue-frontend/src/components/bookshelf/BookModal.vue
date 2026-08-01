<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductFileDropzone from '@/components/product/ProductFileDropzone.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { showToast } from '@/utils/toast'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'

interface Props {
  bookId?: string | null
}

const props = withDefaults(defineProps<Props>(), {
  bookId: null,
})

const emit = defineEmits<{
  close: []
  saved: []
}>()

const bookshelfStore = useBookshelfStore()

const title = ref('')
const coverFile = ref<File | undefined>()
const coverPreview = ref<string | null>(null)
let ownedCoverPreview: string | null = null
const selectedTags = ref<string[]>([])
const tagInput = ref('')
const showTagSuggestions = ref(false)

const isEditing = computed(() => !!props.bookId)
const modalTitle = computed(() => isEditing.value ? '编辑书籍' : '新建书籍')
const availableTags = computed(() => bookshelfStore.tags)
const selectedTagItems = computed<ProductChipItem[]>(() => selectedTags.value.map(tagName => ({
  id: tagName,
  label: tagName,
  ariaLabel: `移除标签 ${tagName}`,
  iconName: 'x',
  interactive: true,
  tone: 'primary',
})))
const filteredTagSuggestions = computed(() => {
  if (!tagInput.value) return availableTags.value
  const query = tagInput.value.toLowerCase()
  return availableTags.value.filter(tag =>
    tag.name.toLowerCase().includes(query) && !selectedTags.value.includes(tag.name)
  )
})

onMounted(() => {
  if (props.bookId) {
    const book = bookshelfStore.books.find(b => b.id === props.bookId)
    if (book) {
      title.value = book.title
      coverPreview.value = book.cover || null
      if (book.tags && book.tags.length > 0) {
        selectedTags.value = [...book.tags]
      }
    }
  }
})

async function handleCoverSelect(files: File[]) {
  const file = files[0]
  if (!file) return

  if (!file.type.startsWith('image/')) {
    showToast('请选择图片文件', 'error')
    return
  }

  if (ownedCoverPreview) URL.revokeObjectURL(ownedCoverPreview)
  coverFile.value = file
  ownedCoverPreview = URL.createObjectURL(file)
  coverPreview.value = ownedCoverPreview
}

onUnmounted(() => {
  if (ownedCoverPreview) URL.revokeObjectURL(ownedCoverPreview)
})

function addTag(tagName: string) {
  if (!selectedTags.value.includes(tagName)) {
    selectedTags.value.push(tagName)
  }
  tagInput.value = ''
  showTagSuggestions.value = false
}

async function createAndAddTag() {
  const name = tagInput.value.trim()
  if (!name) return

  const existing = availableTags.value.find(t => t.name === name)
  if (existing) {
    addTag(existing.name)
    return
  }

  try {
    const newTag = await bookshelfStore.createTag(name)
    if (newTag) {
      addTag(newTag.name)
    }
  } catch (error) {
    showToast('创建标签失败', 'error')
  }
}

function removeTag(tagName: string) {
  selectedTags.value = selectedTags.value.filter(name => name !== tagName)
}

async function saveBook() {
  if (!title.value.trim()) {
    showToast('请输入书籍名称', 'warning')
    return
  }

  const tagNames = selectedTags.value

  try {
    if (isEditing.value && props.bookId) {
      const success = await bookshelfStore.updateBookApi(props.bookId, {
        title: title.value.trim(),
        cover: coverFile.value,
        tags: tagNames,
      })
      if (success) {
        showToast('书籍更新成功', 'success')
        emit('saved')
      } else {
        showToast('更新失败', 'error')
      }
    } else {
      const book = await bookshelfStore.createBook(
        title.value.trim(),
        coverFile.value,
        tagNames.length > 0 ? tagNames : undefined
      )
      if (book) {
        showToast('书籍创建成功', 'success')
        emit('saved')
      } else {
        showToast('创建失败', 'error')
      }
    }
  } catch (error) {
    showToast(isEditing.value ? '更新失败' : '创建失败', 'error')
  }
}
</script>

<template>
  <BaseModal :title="modalTitle" @close="emit('close')">
    <form class="book-modal__form" @submit.prevent="saveBook">
      <UiField
        label="书籍名称"
        variant="dialog"
        control-id="bookTitle"
        required
      >
        <UiInput
          id="bookTitle"
          v-model="title"
          type="text"
          placeholder="请输入书籍名称"
          required
        />
      </UiField>

      <UiField
        label="封面图片"
        variant="dialog"
        control-id="bookCoverInput"
        hint="支持 JPG、PNG、WebP 格式，建议比例 3:4"
      >
        <ProductFileDropzone
          input-id="bookCoverInput"
          label="上传书籍封面"
          accept="image/*"
          @select="handleCoverSelect"
        >
          <div class="book-modal__cover-preview">
            <img
              v-if="coverPreview"
              class="book-modal__cover-image"
              :src="coverPreview"
              alt="封面预览"
            >
            <div v-else class="book-modal__cover-placeholder">
              <UiIcon name="camera" class="book-modal__upload-icon" size="32" />
              <span>点击或拖拽上传封面</span>
            </div>
          </div>
        </ProductFileDropzone>
      </UiField>

      <UiField
        label="标签"
        variant="dialog"
        control-id="bookTagInput"
        hint="输入后按回车添加新标签，或从已有标签中选择"
      >
        <div class="book-modal__tag-input-container">
          <ProductChipList
            v-if="selectedTagItems.length > 0"
            class="book-modal__selected-tags"
            aria-label="已选标签"
            :items="selectedTagItems"
            @select="removeTag(String($event))"
          />
          <div class="book-modal__tag-dropdown">
            <UiInput
              id="bookTagInput"
              v-model="tagInput"
              variant="embedded"
              type="text"
              placeholder="输入标签名称..."
              autocomplete="off"
              @focus="showTagSuggestions = true"
              @keydown.enter.prevent="createAndAddTag"
            />
            <div
              v-if="showTagSuggestions && filteredTagSuggestions.length > 0"
              class="book-modal__tag-suggestions"
            >
              <ProductRecordCard
                v-for="tag in filteredTagSuggestions"
                :key="tag.name"
                as="button"
                class="book-modal__tag-suggestion"
                :aria-label="`添加标签 ${tag.name}`"
                @click="addTag(tag.name)"
              >
                {{ tag.name }}
              </ProductRecordCard>
            </div>
          </div>
        </div>
      </UiField>
    </form>

    <template #footer>
      <ProductActionRow
        aria-label="书籍表单操作"
        variant="dialog"
      >
        <UiButton type="button" variant="secondary" @click="emit('close')">取消</UiButton>
        <UiButton type="button" variant="primary" @click="saveBook">保存</UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<style scoped>
.book-modal__cover-preview {
  width: 150px;
  height: 200px;
  margin: 0 auto;
  background: var(--color-surface-subtle);
  border-radius: 4px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.book-modal__cover-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.book-modal__cover-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: var(--color-text-supporting);
}

.book-modal__upload-icon {
  display: inline-flex;
}

.book-modal__tag-input-container {
  border: 1px solid var(--color-border-muted);
  border-radius: 6px;
  padding: 8px;
}

.book-modal__selected-tags {
  margin-bottom: 8px;
}

.book-modal__tag-dropdown {
  position: relative;
}

.book-modal__tag-suggestions {
  --book-modal-tag-suggestions-shadow: var(--shadow-medium);

  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: var(--color-surface-card);
  border: 1px solid var(--color-border-muted);
  border-radius: 6px;
  box-shadow: 0 4px 12px var(--book-modal-tag-suggestions-shadow);
  max-height: 200px;
  overflow-y: auto;
  z-index: var(--z-local-toolbar);
}

.book-modal__tag-suggestion {
  --product-record-card-background: transparent;
  --product-record-card-border: transparent;
  --product-record-card-padding: 10px 12px;
  --product-record-card-radius: 0;
  --product-record-card-shadow-hover: none;

  color: var(--color-text-default);
}

.book-modal__tag-suggestion:hover {
  --product-record-card-background: var(--color-surface-subtle);
}
</style>
