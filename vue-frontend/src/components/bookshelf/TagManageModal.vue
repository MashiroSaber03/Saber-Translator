<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'
/**
 * 标签管理模态框组件
 * 功能：创建、编辑、删除标签
 */

import { ref, computed } from 'vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { showToast } from '@/utils/toast'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'

const emit = defineEmits<{
  close: []
}>()

const bookshelfStore = useBookshelfStore()

// 新标签表单
const newTagName = ref('')
const newTagColor = ref('#667eea')

const editingTagName = ref<string | null>(null)
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
    const tag = await bookshelfStore.createTag(name, newTagColor.value)
    if (tag) {
      showToast('标签创建成功', 'success')
      newTagName.value = ''
      newTagColor.value = '#667eea'
    } else {
      showToast('创建失败', 'error')
    }
  } catch (error) {
    showToast('创建失败', 'error')
  }
}

function startEditTag(tag: { name: string; color?: string }) {
  editingTagName.value = tag.name
  editTagName.value = tag.name
  editTagColor.value = tag.color || '#667eea'
}

function cancelEdit() {
  editingTagName.value = null
  editTagName.value = ''
  editTagColor.value = ''
}

async function saveEditTag() {
  if (!editingTagName.value) return
  
  const name = editTagName.value.trim()
  if (!name) {
    showToast('标签名称不能为空', 'warning')
    return
  }
  
  const originalTagName = editingTagName.value
  
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
  } catch {
    showToast('更新失败', 'error')
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
        <UiInput
          v-model="newTagName"
          class="tag-manage-modal__new-name-input"
          type="text"
          placeholder="输入新标签名称..."
          @keydown.enter="createTag"
        />
        <UiInput
          v-model="newTagColor"
          class="tag-manage-modal__new-color-input"
          type="color"
          title="选择颜色"
        />
        <UiButton variant="primary" size="sm" @click="createTag">添加</UiButton>
      </div>
    </div>

    <!-- 标签列表 -->
    <div class="tag-list">
      <div v-if="tags.length === 0" class="empty-hint">
        暂无标签，请在上方添加
      </div>
      
      <div
        v-for="tag in tags"
        :key="tag.name"
        class="tag-manage-item"
      >
        <!-- 非编辑状态：显示标签信息和操作按钮 -->
        <div v-if="editingTagName !== tag.name" class="tag-view-mode">
          <span
            class="tag-color-dot"
            :style="{ backgroundColor: tag.color || '#667eea' }"
          ></span>
          <span class="tag-name">{{ tag.name }}</span>
          <span class="tag-book-count">{{ tag.book_count || 0 }} 本</span>
          <!-- 编辑和删除按钮 -->
          <UiButton
            variant="toolbar"
            class="tag-edit-btn"
            @click="startEditTag(tag)"
          >
            编辑
          </UiButton>
          <UiButton
            variant="toolbar"
            class="tag-delete-btn"
            @click="deleteTag(tag.name)"
          >
            删除
          </UiButton>
        </div>
        
        <!-- 编辑状态：内联编辑表单 -->
        <div v-if="editingTagName === tag.name" class="tag-edit-mode">
          <UiInput
            v-model="editTagColor"
            type="color"
            class="edit-color-input"
            title="选择颜色"
          />
          <UiInput
            v-model="editTagName"
            type="text"
            class="edit-name-input"
            placeholder="标签名称"
            @keydown.enter="saveEditTag"
          />
          <UiButton
            variant="toolbar"
            class="tag-save-btn"
            @click="saveEditTag"
          >
            保存
          </UiButton>
          <UiButton
            variant="toolbar"
            class="tag-cancel-btn"
            @click="cancelEdit"
          >
            取消
          </UiButton>
        </div>
      </div>
    </div>

    <template #footer>
      <UiButton variant="secondary" @click="emit('close')">关闭</UiButton>
    </template>
  </BaseModal>
</template>

<style scoped>
.tag-manage-form {
  --tag-manage-modal-danger-shadow: rgba(220, 53, 69, .4);
  --tag-manage-modal-focus-shadow: rgba(102, 126, 234, .2);
  --tag-manage-modal-row-background: #f8f9fa;
  --tag-manage-modal-delete-start: #dc3545;
  --tag-manage-modal-delete-end: #c82333;
  --tag-manage-modal-save-end: #218838;
  --tag-manage-modal-cancel-background: #e9ecef;
  --tag-manage-modal-cancel-hover-background: #dee2e6;

  margin-bottom: 20px;
}

.form-row {
  display: flex;
  gap: 8px;
}

.tag-manage-modal__new-name-input {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 6px;
  font-size: 14px;
  outline: none;
  background: var(--color-surface-input, var(--color-surface-base));
  color: var(--color-text-default);
}

.tag-manage-modal__new-name-input:focus {
  border-color: var(--color-action-primary, var(--color-border-brand-gradient));
}

.tag-manage-modal__new-color-input {
  width: 40px;
  height: 40px;
  padding: 2px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
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
  color: var(--color-text-supporting, var(--color-text-muted));
}

.tag-manage-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  background: var(--tag-manage-modal-row-background);
  border-radius: 6px;
}

.tag-view-mode,
.tag-edit-mode {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
}

.tag-color-dot {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  flex-shrink: 0;
}

.tag-name {
  flex: 1;
  font-size: 14px;
  color: var(--color-text-default);
}

.tag-book-count {
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-muted));
  margin-right: 8px;
}

.tag-edit-btn {
  padding: 4px 12px;
  background: linear-gradient(135deg, var(--color-surface-brand-gradient-start) 0%, var(--color-surface-brand-gradient-end) 100%);
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.tag-edit-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 8px var(--shadow-brand-soft);
}

.tag-delete-btn {
  padding: 4px 12px;
  background: linear-gradient(135deg, var(--tag-manage-modal-delete-start) 0%, var(--tag-manage-modal-delete-end) 100%);
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.tag-delete-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 8px var(--tag-manage-modal-danger-shadow);
}

.edit-color-input {
  width: 32px;
  height: 32px;
  padding: 2px;
  border: 1px solid var(--color-border-muted, var(--color-border-subtle));
  border-radius: 4px;
  cursor: pointer;
  flex-shrink: 0;
}

.edit-name-input {
  flex: 1;
  padding: 6px 10px;
  border: 1px solid var(--color-action-primary, var(--color-border-brand-gradient));
  border-radius: 4px;
  font-size: 14px;
  outline: none;
  background: var(--color-surface-input, var(--color-surface-base));
  color: var(--color-text-default);
}

.edit-name-input:focus {
  box-shadow: 0 0 0 2px var(--tag-manage-modal-focus-shadow);
}

.tag-save-btn {
  padding: 4px 12px;
  background: linear-gradient(135deg, var(--color-surface-success-gradient-start) 0%, var(--tag-manage-modal-save-end) 100%);
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.tag-save-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 2px 8px var(--shadow-success-soft);
}

.tag-cancel-btn {
  padding: 4px 12px;
  background: var(--tag-manage-modal-cancel-background);
  color: var(--color-text-default);
  border: none;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.tag-cancel-btn:hover {
  background: var(--color-surface-interactive-hover, var(--tag-manage-modal-cancel-hover-background));
}
</style>
