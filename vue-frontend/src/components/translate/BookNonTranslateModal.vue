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
      <div class="constraint-description">
        命中当前文本的禁翻内容会被保护为占位符，翻译完成后再还原。
      </div>
      <label class="checkbox-label">
        <input :checked="draft.enabled" type="checkbox" @change="toggleEnabled" />
        启用禁翻表
      </label>
      <TranslationConstraintTable
        :model-value="draft.entries as unknown as Record<string, string>[]"
        :columns="columns"
        :empty-row="emptyRow"
        export-base-name="禁翻表"
        dedupe-key="pattern"
        row-key-prefix="book-non-translate"
        @update:model-value="updateEntries"
      />
    </div>
    <template #footer>
      <button class="btn btn-secondary" @click="handleClose">取消</button>
      <button class="btn btn-primary" :disabled="isSaving" @click="handleSave">保存</button>
    </template>
  </BaseModal>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'

import BaseModal from '@/components/common/BaseModal.vue'
import TranslationConstraintTable from '@/components/settings/shared/TranslationConstraintTable.vue'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'
import type { NonTranslateEntry } from '@/types/translationConstraints'
import { validateRegexEntries } from '@/utils/translationConstraintTable'
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
}

watch(
  () => props.modelValue,
  (value) => {
    isOpen.value = value
    if (value) {
      syncDraft()
    }
  },
  { immediate: true },
)

watch(isOpen, (value) => {
  if (!value && props.modelValue) {
    emit('update:modelValue', false)
  }
})

function syncDraft(): void {
  draft.value = JSON.parse(JSON.stringify(constraintStore.nonTranslate))
}

function toggleEnabled(event: Event): void {
  draft.value.enabled = (event.target as HTMLInputElement).checked
}

function updateEntries(entries: Record<string, string>[]): void {
  draft.value.entries = entries as unknown as NonTranslateEntry[]
}

function handleClose(): void {
  isOpen.value = false
  emit('update:modelValue', false)
}

async function handleSave(): Promise<void> {
  const error = validateRegexEntries(draft.value.entries as any, { patternField: 'pattern' })
  if (error) {
    showToast(error, 'error')
    return
  }

  const ok = await constraintStore.saveBookConstraints({
    ...JSON.parse(JSON.stringify(constraintStore.constraints)),
    non_translate: JSON.parse(JSON.stringify(draft.value)),
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

.constraint-description {
  color: var(--text-secondary);
  font-size: 13px;
  line-height: 1.5;
}

.checkbox-label {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
</style>
