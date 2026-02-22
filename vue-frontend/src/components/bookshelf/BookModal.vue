<script setup lang="ts">
/**
 * 书籍新建/编辑模态框组件
 */

import { ref, computed, onMounted } from 'vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { showToast } from '@/utils/toast'
import BaseModal from '@/components/common/BaseModal.vue'

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
  // 【复刻原版】使用 tag.name 作为唯一标识
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
      // 【复刻原版】book.tags 存储的是标签名称,直接使用即可
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

// 添加标签 - 【复刻原版】使用 name 作为标识
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
    addTag(existing.name)  // 【复刻原版】使用 name
    return
  }

  // 创建新标签
  try {
    const newTag = await bookshelfStore.createTag(name)
    if (newTag) {
      addTag(newTag.name)  // 【复刻原版】使用 name
    }
  } catch (error) {
    showToast('创建标签失败', 'error')
  }
}

// 移除标签 - 【复刻原版】使用 name 作为标识
function removeTag(tagName: string) {
  selectedTags.value = selectedTags.value.filter(name => name !== tagName)
}

// 【已删除】getTagName 函数不再需要,直接使用 name

// 保存书籍
async function saveBook() {
  if (!title.value.trim()) {
    showToast('请输入书籍名称', 'warning')
    return
  }

  // 【复刻原版】selectedTags 已经是标签名称数组,直接使用
  const tagNames = selectedTags.value

  try {
    if (isEditing.value && props.bookId) {
      // 【复刻原版 bookshelf.js saveBook】更新书籍时一次性传递所有数据包括 tags
      const success = await bookshelfStore.updateBookApi(props.bookId, {
        title: title.value.trim(),
        cover: coverData.value || undefined,
        tags: tagNames  // 【复刻原版】一同传递 tags 数组
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
      <div class="form-group">
        <label for="bookTitle">书籍名称 <span class="required">*</span></label>
        <input
          id="bookTitle"
          v-model="title"
          type="text"
          placeholder="请输入书籍名称"
          required
        >
      </div>

      <!-- 封面图片 -->
      <div class="form-group">
        <label>封面图片</label>
        <div
          class="cover-upload-area"
          @dragover.prevent
          @drop="handleCoverDrop"
          @click="($refs.coverInput as HTMLInputElement).click()"
        >
          <input
            ref="coverInput"
            type="file"
            accept="image/*"
            hidden
            @change="handleCoverUpload"
          >
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
        </div>
        <p class="form-hint">支持 JPG、PNG、WebP 格式，建议比例 3:4</p>
      </div>

      <!-- 标签 -->
      <div class="form-group">
        <label>标签</label>
        <div class="tag-input-container">
          <!-- 已选标签 -->
          <div class="selected-tags">
            <!-- 【复刻原版】selectedTags 已经存储标签名称,直接使用 -->
            <span
              v-for="tagName in selectedTags"
              :key="tagName"
              class="selected-tag"
            >
              {{ tagName }}
              <button type="button" class="remove-tag" @click="removeTag(tagName)">×</button>
            </span>
          </div>
          <!-- 标签输入 -->
          <div class="tag-dropdown">
            <input
              v-model="tagInput"
              type="text"
              placeholder="输入标签名称..."
              autocomplete="off"
              @focus="showTagSuggestions = true"
              @keypress.enter.prevent="createAndAddTag"
            >
            <div
              v-if="showTagSuggestions && filteredTagSuggestions.length > 0"
              class="tag-suggestions"
            >
              <!-- 【复刻原版】使用 tag.name 作为 key 和参数 -->
              <button
                v-for="tag in filteredTagSuggestions"
                :key="tag.name"
                type="button"
                class="tag-suggestion"
                @click="addTag(tag.name)"
              >
                {{ tag.name }}
              </button>
            </div>
          </div>
        </div>
        <p class="form-hint">输入后按回车添加新标签，或从已有标签中选择</p>
      </div>
    </form>

    <template #footer>
      <button type="button" class="btn btn-secondary" @click="emit('close')">取消</button>
      <button type="button" class="btn btn-primary" @click="saveBook">保存</button>
    </template>
  </BaseModal>
</template>

<style scoped>
.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-weight: 500;
  color: var(--text-primary, #333);
}

.required {
  color: #e74c3c;
}

.form-group input[type="text"] {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.form-group input[type="text"]:focus {
  border-color: var(--primary-color, #667eea);
}

.cover-upload-area {
  cursor: pointer;
  border: 2px dashed var(--border-color, #ddd);
  border-radius: 8px;
  padding: 16px;
  text-align: center;
  transition: border-color 0.2s;
}

.cover-upload-area:hover {
  border-color: var(--primary-color, #667eea);
}

.cover-preview {
  width: 150px;
  height: 200px;
  margin: 0 auto;
  background: var(--bg-secondary, #f5f5f5);
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
  color: var(--text-secondary, #999);
}

.upload-icon {
  font-size: 32px;
}

.form-hint {
  margin-top: 8px;
  font-size: 12px;
  color: var(--text-secondary, #999);
}

.tag-input-container {
  border: 1px solid var(--border-color, #ddd);
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
  background: var(--primary-color, #667eea);
  color: #fff;
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

.tag-dropdown input {
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
  background: var(--card-bg, #fff);
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  box-shadow: 0 4px 12px rgb(0, 0, 0, 0.1);
  max-height: 200px;
  overflow-y: auto;
  z-index: 10;
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
  color: var(--text-primary, #333);
}

.tag-suggestion:hover {
  background: var(--bg-secondary, #f5f5f5);
}
</style>
