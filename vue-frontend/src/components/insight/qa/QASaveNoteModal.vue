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
import UiIcon from '@/components/ui/UiIcon.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'

interface QANotePreviewData {
  question: string
  answer: string
  citations: Array<{ page: number }>
}

const props = withDefaults(
  defineProps<{
    noteComment: string
    noteTitle: string
    pendingQAData: QANotePreviewData | null
    renderMarkdown: (content: string) => string
    isSaving?: boolean
    visible: boolean
  }>(),
  {
    isSaving: false,
  }
)

const emit = defineEmits<{
  (event: 'close'): void
  (event: 'save'): void
  (event: 'update:noteComment', value: string): void
  (event: 'update:noteTitle', value: string): void
}>()

const noteTitleModel = computed({
  get: () => props.noteTitle,
  set: value => emit('update:noteTitle', value),
})

const noteCommentModel = computed({
  get: () => props.noteComment,
  set: value => emit('update:noteComment', value),
})

const citationChips = computed<ProductChipItem[]>(() => {
  return (
    props.pendingQAData?.citations.map(citation => ({
      id: citation.page,
      label: `第${citation.page}页`,
      tone: 'primary',
    })) ?? []
  )
})
</script>

<template>
  <BaseModal
    :model-value="visible"
    title="添加笔记"
    size="medium"
    custom-class="qa-note-modal"
    frame-variant="soft"
    footer-tone="muted"
    body-padding="spacious"
    width="90%"
    max-width="560px"
    :show-close-button="!isSaving"
    :close-on-esc="!isSaving"
    :close-on-overlay="!isSaving"
    @close="$emit('close')"
  >
    <template #title>
      <span class="qa-note-modal__title">
        <UiIcon name="file-text" />
        <span>添加笔记</span>
      </span>
    </template>

    <div class="qa-note-modal__body">
      <ProductDetailPanel v-if="pendingQAData" aria-label="问答预览">
        <ProductDetailSection label="问题">
          {{ pendingQAData.question }}
        </ProductDetailSection>

        <ProductDetailSection label="回答" scroll>
          <div v-html="renderMarkdown(pendingQAData.answer)"></div>
        </ProductDetailSection>

        <ProductDetailSection
          v-if="pendingQAData.citations.length > 0"
          label="引用页码"
          :framed="false"
        >
          <ProductChipList aria-label="引用页码" :items="citationChips" />
        </ProductDetailSection>
      </ProductDetailPanel>

      <div class="qa-note-modal__form">
        <UiField variant="settings" label="笔记标题" hint="可选" control-id="qaNoteTitle">
          <UiInput
            id="qaNoteTitle"
            v-model="noteTitleModel"
            type="text"
            placeholder="默认使用问题作为标题..."
          />
        </UiField>

        <UiField variant="settings" label="补充说明" hint="可选" control-id="qaNoteComment">
          <UiTextarea
            id="qaNoteComment"
            v-model="noteCommentModel"
            rows="3"
            variant="panel"
            placeholder="添加你的评论或补充..."
          />
        </UiField>
      </div>
    </div>

    <template #footer>
      <ProductActionRow aria-label="问答笔记保存操作" variant="dialog">
        <UiButton variant="secondary" :disabled="isSaving" @click="$emit('close')">取消</UiButton>
        <UiButton variant="primary" :loading="isSaving" :disabled="isSaving" @click="$emit('save')">
          {{ isSaving ? '保存中...' : '保存笔记' }}
        </UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<style scoped>
.qa-note-modal__title {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
</style>
