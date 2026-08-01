<script setup lang="ts">
import { computed } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import ProductDetailPanel from '@/components/product/ProductDetailPanel.vue'
import ProductDetailSection from '@/components/product/ProductDetailSection.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import type { NoteData, NoteType } from '@/types/insight'

const props = defineProps<{
  editingNote: NoteData | null
  noteContent: string
  notePageNum: number | null
  noteTags: string
  noteTitle: string
  noteType: NoteType
  noteTypeOptions: Array<{ label: string; value: NoteType }>
  visible: boolean
}>()

const emit = defineEmits<{
  (event: 'close'): void
  (event: 'save'): void
  (event: 'showPage', pageNum: number): void
  (event: 'update:noteContent', value: string): void
  (event: 'update:notePageNum', value: number | null): void
  (event: 'update:noteTags', value: string): void
  (event: 'update:noteTitle', value: string): void
  (event: 'update:noteType', value: NoteType): void
}>()

const noteTitleModel = computed({
  get: () => props.noteTitle,
  set: value => emit('update:noteTitle', value),
})

const noteContentModel = computed({
  get: () => props.noteContent,
  set: value => emit('update:noteContent', value),
})

const noteTypeModel = computed({
  get: () => props.noteType,
  set: value => emit('update:noteType', value),
})

const notePageNumModel = computed({
  get: () => props.notePageNum,
  set: value => emit('update:notePageNum', value),
})

const noteTagsModel = computed({
  get: () => props.noteTags,
  set: value => emit('update:noteTags', value),
})

const saveDisabled = computed(() => props.editingNote?.type !== 'qa' && !props.noteContent.trim())
const qaCitationChips = computed<ProductChipItem[]>(() => {
  return props.editingNote?.citations?.map(citation => ({
    id: citation.page,
    label: `第${citation.page}页`,
    ariaLabel: `查看第 ${citation.page} 页`,
    interactive: true,
    tone: 'primary',
  })) ?? []
})

function showCitationPage(id: string | number): void {
  emit('showPage', Number(id))
}
</script>

<template>
  <BaseModal
    :model-value="visible"
    :title="editingNote ? '编辑笔记' : '添加笔记'"
    size="small"
    custom-class="notes-panel-modal"
    frame-variant="soft"
    width="90%"
    max-width="450px"
    @close="$emit('close')"
  >
    <template #title>
      <span>{{ editingNote ? '编辑笔记' : '添加笔记' }}</span>
    </template>

    <div class="note-editor-modal__body">
      <template v-if="editingNote && editingNote.type === 'qa'">
        <ProductDetailPanel aria-label="问答笔记预览">
          <ProductDetailSection label="问题">
            {{ editingNote.question }}
          </ProductDetailSection>

          <ProductDetailSection label="回答" scroll>
            {{ editingNote.answer }}
          </ProductDetailSection>

          <ProductDetailSection
            v-if="editingNote.citations && editingNote.citations.length > 0"
            label="引用页码"
            :framed="false"
          >
            <ProductChipList
              aria-label="引用页码"
              :items="qaCitationChips"
              @select="showCitationPage"
            />
          </ProductDetailSection>

          <ProductDetailSection v-if="editingNote.comment" label="补充说明">
            {{ editingNote.comment }}
          </ProductDetailSection>
        </ProductDetailPanel>

        <UiField
          variant="settings"
          label="笔记标题"
          hint="可选"
          control-id="noteEditorQaTitle"
        >
          <UiInput
            id="noteEditorQaTitle"
            v-model="noteTitleModel"
            type="text"
            placeholder="修改标题..."
          />
        </UiField>
      </template>

      <template v-else>
        <UiField
          variant="settings"
          label="笔记类型"
        >
          <UiSelect
            :model-value="noteTypeModel"
            :options="noteTypeOptions"
            @change="emit('update:noteType', $event as NoteType)"
          />
        </UiField>

        <UiField
          variant="settings"
          label="标题"
          hint="可选"
          control-id="noteEditorTitle"
        >
          <UiInput
            id="noteEditorTitle"
            v-model="noteTitleModel"
            type="text"
            placeholder="给笔记起个标题..."
          />
        </UiField>

        <UiField
          variant="settings"
          label="内容"
          required
          control-id="noteEditorContent"
        >
          <UiTextarea
            id="noteEditorContent"
            v-model="noteContentModel"
            rows="5"
            variant="panel"
            placeholder="写下你的想法..."
          />
        </UiField>

        <UiField
          variant="settings"
          label="关联页码"
          hint="可选"
          control-id="noteEditorPageNum"
        >
          <UiNumberField
            input-id="noteEditorPageNum"
            v-model="notePageNumModel"
            nullable
            aria-label="关联页码"
            :min="1"
          />
        </UiField>

        <UiField
          variant="settings"
          label="标签"
          hint="可选"
          control-id="noteEditorTags"
        >
          <UiInput
            id="noteEditorTags"
            v-model="noteTagsModel"
            type="text"
            placeholder="多个标签用逗号分隔，如: 角色,剧情"
          />
        </UiField>
      </template>
    </div>

    <template #footer>
      <ProductActionRow
        aria-label="笔记编辑操作"
        variant="dialog"
      >
        <UiButton variant="secondary" @click="$emit('close')">取消</UiButton>
        <UiButton
          variant="primary"
          :disabled="saveDisabled"
          @click="$emit('save')"
        >
          保存
        </UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<style scoped>
.note-editor-modal__body {
  color: var(--insight-text-primary);
}
</style>
