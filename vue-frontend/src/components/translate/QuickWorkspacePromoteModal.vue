<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { UiSelectOption, UiSelectValue } from '@/components/ui/selectTypes'
import {
  promoteQuickWorkspace,
  type QuickWorkspacePromotion,
} from '@/api/v2/content'
import { useBookshelfStore } from '@/stores/bookshelfStore'

const props = defineProps<{
  modelValue: boolean
}>()

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
  promoted: [result: QuickWorkspacePromotion]
  locked: [error: unknown]
}>()

const bookshelfStore = useBookshelfStore()
const mode = ref<'new_book' | 'existing_book'>('new_book')
const bookTitle = ref('')
const chapterTitle = ref('')
const targetBookId = ref('')
const saving = ref(false)
const errorMessage = ref('')

const bookOptions = computed<UiSelectOption[]>(() => (
  bookshelfStore.books.map(book => ({ label: book.title, value: book.id }))
))

const canSubmit = computed(() => (
  chapterTitle.value.trim().length > 0
  && (
    mode.value === 'new_book'
      ? bookTitle.value.trim().length > 0
      : targetBookId.value.length > 0
  )
  && !saving.value
))

watch(
  () => props.modelValue,
  async visible => {
    if (!visible) return
    errorMessage.value = ''
    if (!bookshelfStore.books.length) {
      await bookshelfStore.loadBooks()
      if (bookshelfStore.error) {
        errorMessage.value = `加载书架失败：${bookshelfStore.error}`
      }
    }
  },
  { immediate: true },
)

function setMode(value: 'new_book' | 'existing_book') {
  mode.value = value
  errorMessage.value = ''
}

function setTargetBook(value: UiSelectValue) {
  targetBookId.value = String(value)
}

function close() {
  if (saving.value) return
  emit('update:modelValue', false)
}

function handleChapterTitleEnter(event: KeyboardEvent): void {
  if (event.isComposing) return
  void submit()
}

async function submit() {
  if (!canSubmit.value) return
  saving.value = true
  errorMessage.value = ''
  try {
    const result = await promoteQuickWorkspace(
      mode.value === 'new_book'
        ? {
            mode: 'new_book',
            title: bookTitle.value.trim(),
            chapterTitle: chapterTitle.value.trim(),
          }
        : {
            mode: 'existing_book',
            bookId: targetBookId.value,
            chapterTitle: chapterTitle.value.trim(),
          },
    )
    emit('promoted', result)
    emit('update:modelValue', false)
  } catch (error) {
    if (
      error
      && typeof error === 'object'
      && 'status' in error
      && error.status === 423
    ) {
      emit('locked', error)
      return
    }
    errorMessage.value = error instanceof Error ? error.message : '保存到书架失败'
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <BaseModal
    :model-value="modelValue"
    title="保存到书架"
    size="small"
    :close-on-overlay="!saving"
    :close-on-esc="!saving"
    @close="close"
  >
    <div class="quick-promote">
      <ProductActionRow aria-label="保存目标">
        <UiButton
          :variant="mode === 'new_book' ? 'primary' : 'secondary'"
          :aria-pressed="mode === 'new_book'"
          :disabled="saving"
          @click="setMode('new_book')"
        >
          新建书籍
        </UiButton>
        <UiButton
          :variant="mode === 'existing_book' ? 'primary' : 'secondary'"
          :aria-pressed="mode === 'existing_book'"
          :disabled="saving"
          @click="setMode('existing_book')"
        >
          已有书籍
        </UiButton>
      </ProductActionRow>

      <UiField
        v-if="mode === 'new_book'"
        label="书籍名称"
        variant="settings"
        control-id="quick-promote-book-title"
      >
        <UiInput
          id="quick-promote-book-title"
          v-model="bookTitle"
          maxlength="500"
          :disabled="saving"
          placeholder="输入新书名称"
        />
      </UiField>

      <UiField
        v-else
        label="目标书籍"
        variant="settings"
        control-id="quick-promote-target-book"
      >
        <UiSelect
          id="quick-promote-target-book"
          :model-value="targetBookId"
          :options="bookOptions"
          :disabled="saving"
          placeholder="选择已有书籍"
          @change="setTargetBook"
        />
      </UiField>

      <UiField
        label="章节名称"
        variant="settings"
        control-id="quick-promote-chapter-title"
      >
        <UiInput
          id="quick-promote-chapter-title"
          v-model="chapterTitle"
          maxlength="500"
          :disabled="saving"
          placeholder="输入章节名称"
          @keydown.enter="handleChapterTitleEnter"
        />
      </UiField>

      <ProductStatusBanner
        v-if="mode === 'existing_book'"
        tone="warning"
      >
        保存到已有书籍时，该书原有术语表保持不变；快速工作区术语表会被清空。
      </ProductStatusBanner>
      <ProductStatusBanner v-if="errorMessage" tone="danger">
        {{ errorMessage }}
      </ProductStatusBanner>
    </div>

    <template #footer>
      <ProductActionRow aria-label="保存到书架操作" variant="dialog">
        <UiButton variant="secondary" :disabled="saving" @click="close">取消</UiButton>
        <UiButton variant="primary" :disabled="!canSubmit" @click="submit">
          {{ saving ? '保存中…' : '保存到书架' }}
        </UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<style scoped>
.quick-promote {
  display: grid;
  gap: 16px;
}
</style>
