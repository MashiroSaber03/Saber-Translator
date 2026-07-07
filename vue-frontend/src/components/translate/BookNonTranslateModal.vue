<template>
  <BaseModal
    v-model="isOpen"
    title="禁翻表"
    size="large"
    :close-on-overlay="true"
    :close-on-esc="true"
    @close="handleClose"
  >
    <div class="constraint-modal-body">
      <ProductStatusBanner tone="info" role="note">
        命中当前文本的禁翻内容会被保护为占位符，翻译完成后再还原。
      </ProductStatusBanner>
      <UiCheckbox
        :model-value="draft.enabled"
        label="启用禁翻表"
        @change="toggleEnabled"
      />
      <TranslationConstraintTable
        :model-value="draft.entries"
        :columns="columns"
        :empty-row="emptyRow"
        export-base-name="禁翻表"
        dedupe-key="pattern"
        row-key-prefix="book-non-translate"
        @update:model-value="updateEntries"
      />
    </div>
    <template #footer>
      <ProductActionRow
        variant="dialog"
        aria-label="禁翻表操作"
      >
        <UiButton variant="secondary" @click="handleClose">取消</UiButton>
        <UiButton variant="primary" :disabled="isSaving" @click="handleSave">保存</UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import { computed, ref, watch } from 'vue'

import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'
import TranslationConstraintTable from '@/components/settings/shared/TranslationConstraintTable.vue'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'
import type { NonTranslateEntry } from '@/types/translationConstraints'
import { deepClone } from '@/utils/deepClone'
import { getStringField, validateRegexEntries } from '@/utils/translationConstraintTable'
import { showToast } from '@/utils/toast'

const props = defineProps<{ modelValue: boolean }>()
const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'saved'): void
}>()

const constraintStore = useBookTranslationConstraintsStore()
const isOpen = ref(props.modelValue)
const isSaving = computed(() => constraintStore.isSaving)
const draft = ref({
  enabled: false,
  entries: [] as NonTranslateEntry[],
})

const columns = [
  { key: 'pattern', label: '内容/规则' },
  { key: 'note', label: '备注' },
  {
    key: 'matchMode',
    label: '匹配方式',
    type: 'select' as const,
    options: [
      { label: '普通字符串', value: 'text' },
      { label: '正则表达式', value: 'regex' },
    ],
  },
]

const emptyRow = {
  pattern: '',
  note: '',
  matchMode: 'text',
} satisfies NonTranslateEntry

function toMatchMode(value: string): NonTranslateEntry['matchMode'] {
  return value === 'regex' ? 'regex' : 'text'
}

function toNonTranslateEntry(row: object): NonTranslateEntry {
  return {
    pattern: getStringField(row, 'pattern'),
    note: getStringField(row, 'note'),
    matchMode: toMatchMode(getStringField(row, 'matchMode')),
  }
}

watch(
  () => props.modelValue,
  value => {
    isOpen.value = value
    if (value) {
      syncDraft()
    }
  },
  { immediate: true }
)

watch(isOpen, value => {
  if (!value && props.modelValue) {
    emit('update:modelValue', false)
  }
})

function syncDraft(): void {
  draft.value = deepClone(constraintStore.nonTranslate)
}

function toggleEnabled(checked: boolean): void {
  draft.value.enabled = checked
}

function updateEntries(entries: object[]): void {
  draft.value.entries = entries.map(toNonTranslateEntry)
}

function handleClose(): void {
  isOpen.value = false
  emit('update:modelValue', false)
}

async function handleSave(): Promise<void> {
  const error = validateRegexEntries(draft.value.entries, { patternField: 'pattern' })
  if (error) {
    showToast(error, 'error')
    return
  }

  const ok = await constraintStore.saveBookConstraints({
    ...deepClone(constraintStore.constraints),
    non_translate: deepClone(draft.value),
  })
  if (!ok) {
    showToast('保存禁翻表失败', 'error')
    return
  }

  showToast('禁翻表已保存', 'success')
  emit('saved')
  handleClose()
}
</script>

<style scoped>
.constraint-modal-body {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

</style>
