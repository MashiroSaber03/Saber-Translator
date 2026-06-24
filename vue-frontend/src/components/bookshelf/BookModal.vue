<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
/**
 * 书籍新建/编辑模态框组件
 */

import { ref, computed, onMounted } from 'vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { showToast } from '@/utils/toast'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'

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

// 表单数据
const title = ref('')
const coverData = ref<string | null>(null)
const selectedTags = ref<string[]>([])
const tagInput = ref('')
const showTagSuggestions = ref(false)

// 计算属性
const isEditing = computed(() => !!props.bookId)
const modalTitle = computed(() => isEditing.value ? '编辑书籍' : '新建书籍')
const availableTags = computed(() => bookshelfStore.tags)
const filteredTagSuggestions = computed(() => {
  if (!tagInput.value) return availableTags.value
  const query = tagInput.value.toLowerCase()
  // 使用 tag.name 作为唯一标识
  return availableTags.value.filter(tag => 
    tag.name.toLowerCase().includes(query) && !selectedTags.value.includes(tag.name)
  )
})

// 初始化表单数据
onMounted(() => {
  if (props.bookId) {
    const book = bookshelfStore.books.find(b => b.id === props.bookId)
    if (book) {
      title.value = book.title
      coverData.value = book.cover || null
      if (book.tags && book.tags.length > 0) {
        selectedTags.value = [...book.tags]
      }
    }
  }
})

// 处理封面上传
function handleCoverUpload(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return

  // 验证文件类型
  if (!file.type.startsWith('image/')) {
    showToast('请选择图片文件', 'error')
    return
  }

  // 读取文件为 Base64
  const reader = new FileReader()
  reader.onload = (e) => {
    coverData.value = e.target?.result as string
  }
  reader.readAsDataURL(file)
}

// 处理封面拖拽
function handleCoverDrop(event: DragEvent) {
  event.preventDefault()
  const file = event.dataTransfer?.files[0]
  if (!file || !file.type.startsWith('image/')) return

  const reader = new FileReader()
  reader.onload = (e) => {
    coverData.value = e.target?.result as string
  }
  reader.readAsDataURL(file)
}

function addTag(tagName: string) {
  if (!selectedTags.value.includes(tagName)) {
    selectedTags.value.push(tagName)
  }
  tagInput.value = ''
  showTagSuggestions.value = false
}

// 创建并添加新标签
async function createAndAddTag() {
  const name = tagInput.value.trim()
  if (!name) return

  // 检查是否已存在
  const existing = availableTags.value.find(t => t.name === name)
  if (existing) {
      addTag(existing.name)
    return
  }

  // 创建新标签
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

// 保存书籍
async function saveBook() {
  if (!title.value.trim()) {
    showToast('请输入书籍名称', 'warning')
    return
  }

  const tagNames = selectedTags.value

  try {
    if (isEditing.value && props.bookId) {
      // 更新书籍时一次性传递所有数据，包括 tags。
      const success = await bookshelfStore.updateBookApi(props.bookId, {
        title: title.value.trim(),
        cover: coverData.value || undefined,
        tags: tagNames
      })
      if (success) {
        showToast('书籍更新成功', 'success')
        emit('saved')
      } else {
        showToast('更新失败', 'error')
      }
    } else {
      // 创建书籍
      const book = await bookshelfStore.createBook(
        title.value.trim(),
        undefined,
        coverData.value || undefined,
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
    <form @submit.prevent="saveBook">
      <!-- 书籍名称 -->
      <div class="book-modal__field">
        <label for="bookTitle">书籍名称 <span class="required">*</span></label>
        <UiInput
          id="bookTitle"
          v-model="title"
          class="book-modal__title-input"
          type="text"
          placeholder="请输入书籍名称"
          required
        />
      </div>

      <!-- 封面图片 -->
      <div class="book-modal__field">
        <label>封面图片</label>
        <label
          for="bookCoverInput"
          class="cover-upload-area"
          @dragover.prevent
          @drop="handleCoverDrop"
        >
          <UiFileInput
            id="bookCoverInput"
            class="book-modal__cover-input"
            accept="image/*"
            @change="handleCoverUpload"
          />
          <div class="cover-preview">
            <img
              v-if="coverData"
              :src="coverData"
              alt="封面预览"
            >
            <div v-else class="cover-placeholder">
              <span class="upload-icon">📷</span>
              <span>点击或拖拽上传封面</span>
            </div>
          </div>
        </label>
        <p class="form-hint">支持 JPG、PNG、WebP 格式，建议比例 3:4</p>
      </div>

      <!-- 标签 -->
      <div class="book-modal__field">
        <label>标签</label>
        <div class="tag-input-container">
          <!-- 已选标签 -->
          <div class="selected-tags">
            <span
              v-for="tagName in selectedTags"
              :key="tagName"
              class="selected-tag"
            >
              {{ tagName }}
              <UiButton
                variant="toolbar"
                type="button"
                class="remove-tag"
                :aria-label="`移除标签 ${tagName}`"
                @click="removeTag(tagName)"
              >
                ×
              </UiButton>
            </span>
          </div>
          <!-- 标签输入 -->
          <div class="tag-dropdown">
            <UiInput
              v-model="tagInput"
              class="book-modal__tag-input"
              type="text"
              placeholder="输入标签名称..."
              autocomplete="off"
              @focus="showTagSuggestions = true"
              @keydown.enter.prevent="createAndAddTag"
            />
            <div
              v-if="showTagSuggestions && filteredTagSuggestions.length > 0"
              class="tag-suggestions"
            >
              <UiButton
                variant="toolbar"
                v-for="tag in filteredTagSuggestions"
                :key="tag.name"
                type="button"
                class="tag-suggestion"
                @click="addTag(tag.name)"
              >
                {{ tag.name }}
              </UiButton>
            </div>
          </div>
        </div>
        <p class="form-hint">输入后按回车添加新标签，或从已有标签中选择</p>
      </div>
    </form>

    <template #footer>
      <UiButton type="button" variant="secondary" @click="emit('close')">取消</UiButton>
      <UiButton type="button" variant="primary" @click="saveBook">保存</UiButton>
    </template>
  </BaseModal>
</template>

<style scoped>
.tag-suggestions {
  --book-modal-shadow-default: rgba(0, 0, 0, .1);
}

.book-modal__field {
  margin-bottom: 20px;
}

.book-modal__field label {
  display: block;
  margin-bottom: 8px;
  font-weight: 500;
  color: var(--color-text-default);
}

.required {
  color: var(--color-text-danger-strong);
}

.book-modal__title-input {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 6px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.book-modal__title-input:focus {
  border-color: var(--color-action-primary, var(--color-border-brand-gradient));
}

.cover-upload-area {
  position: relative;
  display: block;
  cursor: pointer;
  border: 2px dashed var(--color-border-muted, var(--color-border-subtle));
  border-radius: 8px;
  padding: 16px;
  text-align: center;
  transition: border-color 0.2s;
}

.book-modal__cover-input {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  cursor: pointer;
}

.cover-upload-area:hover {
  border-color: var(--color-action-primary, var(--color-border-brand-gradient));
}

.book-modal__cover-input:focus-visible + .cover-preview {
  outline: 2px solid var(--color-action-primary);
  outline-offset: 4px;
}

.cover-preview {
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

.cover-preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.cover-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: var(--color-text-supporting, var(--color-text-muted));
}

.upload-icon {
  font-size: 32px;
}

.form-hint {
  margin-top: 8px;
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-muted));
}

.tag-input-container {
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 6px;
  padding: 8px;
}

.selected-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 8px;
}

.selected-tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  background: var(--color-action-primary, var(--color-action-brand));
  color: var(--color-text-inverse);
  font-size: 12px;
  border-radius: 4px;
}

.remove-tag {
  background: none;
  border: none;
  color: inherit;
  cursor: pointer;
  padding: 0;
  font-size: 14px;
  line-height: 1;
}

.tag-dropdown {
  position: relative;
}

.book-modal__tag-input {
  width: 100%;
  padding: 8px;
  border: none;
  outline: none;
  font-size: 14px;
  background: transparent;
}

.tag-suggestions {
  position: absolute;
  top: 100%;
  left: 0;
  right: 0;
  background: var(--color-surface-card, var(--color-surface-base));
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 6px;
  box-shadow: 0 4px 12px var(--book-modal-shadow-default);
  max-height: 200px;
  overflow-y: auto;
  z-index: var(--z-local-toolbar);
}

.tag-suggestion {
  display: block;
  width: 100%;
  padding: 10px 12px;
  text-align: left;
  background: none;
  border: none;
  cursor: pointer;
  font-size: 14px;
  color: var(--color-text-default);
}

.tag-suggestion:hover {
  background: var(--color-surface-subtle);
}
</style>
