<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'
import { ref, computed } from 'vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { showToast } from '@/utils/toast'
import BaseModal from '@/components/common/BaseModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiColorInput from '@/components/ui/UiColorInput.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import { confirmProductAction } from '@/composables/useProductConfirm'
import { BOOKSHELF_DEFAULT_TAG_COLOR } from '@/constants/bookshelf'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'

const emit = defineEmits<{
  close: []
}>()

const bookshelfStore = useBookshelfStore()

const newTagName = ref('')
const newTagColor = ref(BOOKSHELF_DEFAULT_TAG_COLOR)

const editingTagName = ref<string | null>(null)
const editTagName = ref('')
const editTagColor = ref('')

const tags = computed(() => bookshelfStore.tags)

function tagMetadataItems(tag: { name: string; color?: string; book_count?: number }): ProductChipItem[] {
  const tagColor = tag.color || 'var(--color-action-brand)'

  return [
    {
      id: `tag-${tag.name}`,
      label: tag.name,
      tone: 'custom',
      backgroundColor: tagColor,
      borderColor: tagColor,
      textColor: 'var(--color-text-inverse)',
    },
    {
      id: `count-${tag.name}`,
      label: `${tag.book_count || 0} 本`,
      tone: 'neutral',
    },
  ]
}

async function createTag() {
  const name = newTagName.value.trim()
  if (!name) {
    showToast('请输入标签名称', 'warning')
    return
  }

  if (tags.value.some(t => t.name === name)) {
    showToast('标签已存在', 'warning')
    return
  }

  try {
    const tag = await bookshelfStore.createTag(name, newTagColor.value)
    if (tag) {
      showToast('标签创建成功', 'success')
      newTagName.value = ''
      newTagColor.value = BOOKSHELF_DEFAULT_TAG_COLOR
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
  editTagColor.value = tag.color || BOOKSHELF_DEFAULT_TAG_COLOR
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

async function deleteTag(tagName: string) {
  const confirmed = await confirmProductAction({
    title: '删除标签',
    message: `确定要删除标签“${tagName}”吗？此操作不会删除书籍，但会从相关书籍中移除该标签。`,
    confirmText: '删除',
    cancelText: '取消',
    tone: 'danger',
  })
  if (!confirmed) return

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
    <div class="tag-manage-modal__form">
      <UiFormGrid>
        <UiField label="标签名称" variant="settings" control-id="tag-manage-new-name">
          <UiInput
            id="tag-manage-new-name"
            v-model="newTagName"
            type="text"
            placeholder="输入新标签名称..."
            @keydown.enter="createTag"
          />
        </UiField>
        <UiField label="标签颜色" variant="settings" control-id="tag-manage-new-color">
          <UiColorInput
            input-id="tag-manage-new-color"
            v-model="newTagColor"
            title="选择颜色"
          />
        </UiField>
      </UiFormGrid>
      <ProductActionRow aria-label="新建标签操作" justify="start">
        <UiButton variant="primary" size="sm" @click="createTag">添加</UiButton>
      </ProductActionRow>
    </div>

    <div class="tag-manage-modal__list">
      <ProductStatusBanner
        v-if="tags.length === 0"
        class="tag-manage-modal__empty-state"
        tone="neutral"
        icon-name="tags"
        role="note"
      >
        暂无标签，请在上方添加
      </ProductStatusBanner>

      <ProductRecordCard
        v-for="tag in tags"
        :key="tag.name"
        class="tag-manage-modal__item"
        :aria-label="`标签 ${tag.name}`"
      >
        <div v-if="editingTagName !== tag.name" class="tag-manage-modal__view-mode">
          <ProductChipList
            class="tag-manage-modal__metadata"
            :aria-label="`${tag.name} 标签信息`"
            :items="tagMetadataItems(tag)"
          />
          <UiButton
            variant="secondary"
            size="xs"
            class="tag-manage-modal__row-edit-action"
            @click="startEditTag(tag)"
          >
            编辑
          </UiButton>
          <UiButton
            variant="danger"
            size="xs"
            class="tag-manage-modal__row-delete-action"
            @click="deleteTag(tag.name)"
          >
            删除
          </UiButton>
        </div>

        <div v-if="editingTagName === tag.name" class="tag-manage-modal__edit-mode">
          <UiFormGrid class="tag-manage-modal__edit-fields">
            <UiField label="编辑标签颜色" variant="settings" :control-id="`tag-edit-color-${tag.name}`">
              <UiColorInput
                :input-id="`tag-edit-color-${tag.name}`"
                v-model="editTagColor"
                title="选择颜色"
              />
            </UiField>
            <UiField label="编辑标签名称" variant="settings" :control-id="`tag-edit-name-${tag.name}`">
              <UiInput
                :id="`tag-edit-name-${tag.name}`"
                v-model="editTagName"
                type="text"
                size="sm"
                placeholder="标签名称"
                @keydown.enter="saveEditTag"
              />
            </UiField>
          </UiFormGrid>
          <ProductActionRow
            aria-label="编辑标签操作"
            class="tag-manage-modal__edit-actions"
            justify="start"
          >
            <UiButton
              variant="primary"
              size="xs"
              class="tag-manage-modal__edit-save-action"
              @click="saveEditTag"
            >
              保存
            </UiButton>
            <UiButton
              variant="secondary"
              size="xs"
              class="tag-manage-modal__edit-cancel-action"
              @click="cancelEdit"
            >
              取消
            </UiButton>
          </ProductActionRow>
        </div>
      </ProductRecordCard>
    </div>

    <template #footer>
      <ProductActionRow aria-label="标签管理弹窗操作" variant="dialog">
        <UiButton variant="secondary" @click="emit('close')">关闭</UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<style scoped>
.tag-manage-modal__form {
  margin-bottom: 20px;
}

.tag-manage-modal__list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 300px;
  overflow-y: auto;
}

.tag-manage-modal__empty-state {
  align-items: center;
}

.tag-manage-modal__item {
  --product-record-card-background: var(--color-surface-app);
  --product-record-card-gap: 0;
  --product-record-card-padding: 10px 12px;
  --product-record-card-radius: 6px;
}

.tag-manage-modal__view-mode,
.tag-manage-modal__edit-mode {
  width: 100%;
}

.tag-manage-modal__view-mode {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
}

.tag-manage-modal__edit-mode {
  display: grid;
  gap: 10px;
}

.tag-manage-modal__edit-fields {
  margin-bottom: 0;
}

.tag-manage-modal__metadata {
  flex: 1 1 180px;
  min-width: 0;
}

.tag-manage-modal__row-edit-action,
.tag-manage-modal__row-delete-action,
.tag-manage-modal__edit-save-action,
.tag-manage-modal__edit-cancel-action {
  flex: 0 0 auto;
}

</style>
