<script setup lang="ts">
/**
 * 标签管理模态框组件
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

// 删除标签
async function deleteTag(tagId: string) {
  try {
    const success = await bookshelfStore.deleteTagApi(tagId)
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
        暂无标签，请添加新标签
      </div>
      <div
        v-for="tag in tags"
        :key="tag.id"
        class="tag-item"
      >
        <span
          class="tag-color"
          :style="{ backgroundColor: tag.color || '#667eea' }"
        ></span>
        <span class="tag-name">{{ tag.name }}</span>
        <button
          class="btn-delete"
          title="删除标签"
          @click="deleteTag(tag.id)"
        >
          ×
        </button>
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

.tag-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  background: var(--bg-secondary, #f8f9fa);
  border-radius: 6px;
}

.tag-color {
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

.btn-delete {
  width: 24px;
  height: 24px;
  padding: 0;
  background: none;
  border: none;
  border-radius: 50%;
  cursor: pointer;
  font-size: 18px;
  color: var(--text-secondary, #999);
  display: flex;
  align-items: center;
  justify-content: center;
}

.btn-delete:hover {
  background: #fee;
  color: #e74c3c;
}
</style>
