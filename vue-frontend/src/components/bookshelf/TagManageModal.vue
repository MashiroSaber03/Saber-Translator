<script setup lang="ts">
/**
 * 标签管理模态框组件
 * 【复刻原版 bookshelf.js renderTagManageList + editTag】
 * 功能：创建、编辑、删除标签
 */

import { ref, computed } from 'vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { showToast } from '@/utils/toast'
import BaseModal from '@/components/common/BaseModal.vue'

const emit = defineEmits<{
  close: []
}>()

const bookshelfStore = useBookshelfStore()

// 新标签表单
const newTagName = ref('')
const newTagColor = ref('#667eea')

// 编辑状态 - 使用 tag.name 作为标识符（后端 API 不返回 id）
const editingTagName_id = ref<string | null>(null)  // 正在编辑的标签名称
const editTagName = ref('')
const editTagColor = ref('')

// 计算属性
const tags = computed(() => bookshelfStore.tags)

// 创建新标签
async function createTag() {
  const name = newTagName.value.trim()
  if (!name) {
    showToast('请输入标签名称', 'warning')
    return
  }

  // 检查是否已存在
  if (tags.value.some(t => t.name === name)) {
    showToast('标签已存在', 'warning')
    return
  }

  try {
    await bookshelfStore.createTag(name, newTagColor.value)
    showToast('标签创建成功', 'success')
    newTagName.value = ''
    newTagColor.value = '#667eea'
  } catch (error) {
    showToast('创建失败', 'error')
  }
}

/**
 * 开始编辑标签
 * 【复刻原版 bookshelf.js editTag 第一步】
 * 使用 tag.name 作为唯一标识符（后端 API 不返回 id）
 */
function startEditTag(tag: { name: string; color?: string }) {
  editingTagName_id.value = tag.name  // 使用 name 作为标识
  editTagName.value = tag.name
  editTagColor.value = tag.color || '#667eea'
}

/**
 * 取消编辑
 */
function cancelEdit() {
  editingTagName_id.value = null
  editTagName.value = ''
  editTagColor.value = ''
}

/**
 * 保存编辑
 * 【复刻原版 bookshelf.js editTag API 调用】
 */
async function saveEditTag() {
  if (!editingTagName_id.value) return
  
  const name = editTagName.value.trim()
  if (!name) {
    showToast('标签名称不能为空', 'warning')
    return
  }
  
  // editingTagName_id 就是原标签名称（API 使用名称作为路径参数）
  const originalTagName = editingTagName_id.value
  
  // 检查新名称是否与其他标签重复（排除自己）
  if (name !== originalTagName && tags.value.some(t => t.name === name)) {
    showToast('标签名称已存在', 'warning')
    return
  }
  
  try {
    const success = await bookshelfStore.updateTagApi(
      originalTagName,
      name,
      editTagColor.value
    )
    
    if (success) {
      showToast('标签更新成功', 'success')
      cancelEdit()
    } else {
      showToast('更新失败', 'error')
    }
  } catch (error) {
    showToast('更新失败', 'error')
    console.error('更新标签失败:', error)
  }
}

// 删除标签
async function deleteTag(tagName: string) {
  try {
    const success = await bookshelfStore.deleteTagApi(tagName)
    if (success) {
      showToast('标签已删除', 'success')
    } else {
      showToast('删除失败', 'error')
    }
  } catch (error) {
    showToast('删除失败', 'error')
  }
}
</script>

<template>
  <BaseModal title="标签管理" @close="emit('close')">
    <!-- 新建标签表单 -->
    <div class="tag-manage-form">
      <div class="form-row">
        <input
          v-model="newTagName"
          type="text"
          placeholder="输入新标签名称..."
          @keypress.enter="createTag"
        >
        <input
          v-model="newTagColor"
          type="color"
          title="选择颜色"
        >
        <button class="btn btn-primary btn-sm" @click="createTag">添加</button>
      </div>
    </div>

    <!-- 标签列表 -->
    <div class="tag-list">
      <div v-if="tags.length === 0" class="empty-hint">
        暂无标签，请在上方添加
      </div>
      
      <!-- 【复刻原版 renderTagManageList】标签项样式 -->
      <div
        v-for="tag in tags"
        :key="tag.name"
        class="tag-manage-item"
      >
        <!-- 非编辑状态：显示标签信息和操作按钮 -->
        <div v-if="editingTagName_id !== tag.name" class="tag-view-mode">
          <span
            class="tag-color-dot"
            :style="{ backgroundColor: tag.color || '#667eea' }"
          ></span>
          <span class="tag-name">{{ tag.name }}</span>
          <span class="tag-book-count">{{ tag.book_count || 0 }} 本</span>
          <!-- 【复刻原版】编辑和删除按钮 -->
          <button
            class="tag-edit-btn"
            @click="startEditTag(tag)"
          >
            编辑
          </button>
          <button
            class="tag-delete-btn"
            @click="deleteTag(tag.name)"
          >
            删除
          </button>
        </div>
        
        <!-- 编辑状态：内联编辑表单 -->
        <div v-if="editingTagName_id === tag.name" class="tag-edit-mode">
          <input
            v-model="editTagColor"
            type="color"
            class="edit-color-input"
            title="选择颜色"
          >
          <input
            v-model="editTagName"
            type="text"
            class="edit-name-input"
            placeholder="标签名称"
            @keypress.enter="saveEditTag"
          >
          <button
            class="tag-save-btn"
            @click="saveEditTag"
          >
            保存
          </button>
          <button
            class="tag-cancel-btn"
            @click="cancelEdit"
          >
            取消
          </button>
        </div>
      </div>
    </div>

    <template #footer>
      <button class="btn btn-secondary" @click="emit('close')">关闭</button>
    </template>
  </BaseModal>
</template>

<style scoped>
.tag-manage-form {
  margin-bottom: 20px;
}

.form-row {
  display: flex;
  gap: 8px;
}

.form-row input[type="text"] {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  font-size: 14px;
  outline: none;
  background: var(--input-bg, #fff);
  color: var(--text-primary, #333);
}

.form-row input[type="text"]:focus {
  border-color: var(--primary-color, #667eea);
}

.form-row input[type="color"] {
  width: 40px;
  height: 40px;
  padding: 2px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 6px;
  cursor: pointer;
}

.tag-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 300px;
  overflow-y: auto;
}

.empty-hint {
  text-align: center;
  padding: 32px;
  color: var(--text-secondary, #999);
}

/* 【复刻原版 bookshelf.css .tag-manage-item】 */
.tag-manage-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  background: var(--bg-secondary, #f8f9fa);
  border-radius: 6px;
}

/* 标签查看模式和编辑模式容器 */
.tag-view-mode,
.tag-edit-mode {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
}

/* 【复刻原版】颜色圆点 */
.tag-color-dot {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  flex-shrink: 0;
}

.tag-name {
  flex: 1;
  font-size: 14px;
  color: var(--text-primary, #333);
}

/* 【复刻原版】书籍数量 */
.tag-book-count {
  font-size: 12px;
  color: var(--text-secondary, #999);
  margin-right: 8px;
}

/* 【复刻原版 .tag-edit-btn】编辑按钮样式 */
.tag-edit-btn {
  padding: 4px 12px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.tag-edit-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
}

/* 【复刻原版 .tag-delete-btn】删除按钮样式 */
.tag-delete-btn {
  padding: 4px 12px;
  background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.tag-delete-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(220, 53, 69, 0.4);
}

/* 编辑状态样式 */
.edit-color-input {
  width: 32px;
  height: 32px;
  padding: 2px;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  cursor: pointer;
  flex-shrink: 0;
}

.edit-name-input {
  flex: 1;
  padding: 6px 10px;
  border: 1px solid var(--primary-color, #667eea);
  border-radius: 4px;
  font-size: 14px;
  outline: none;
  background: var(--input-bg, #fff);
  color: var(--text-primary, #333);
}

.edit-name-input:focus {
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
}

.tag-save-btn {
  padding: 4px 12px;
  background: linear-gradient(135deg, #28a745 0%, #218838 100%);
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.tag-save-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(40, 167, 69, 0.4);
}

.tag-cancel-btn {
  padding: 4px 12px;
  background: var(--bg-tertiary, #e9ecef);
  color: var(--text-primary, #333);
  border: none;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.tag-cancel-btn:hover {
  background: var(--hover-bg, #dee2e6);
}
</style>

