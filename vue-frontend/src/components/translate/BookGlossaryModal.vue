<template>
  <BaseModal
    :model-value="modelValue"
    title="术语表"
    size="large"
    :close-on-overlay="!isSaving"
    :close-on-esc="!isSaving"
    @close="handleClose"
  >
    <div class="constraint-modal-body">
      <ProductStatusBanner class="constraint-modal-body__description" tone="info" role="note">
        命中当前文本的术语会追加到翻译提示词中，并在翻译完成后做术语检查。
      </ProductStatusBanner>
      <UiCheckbox
        :model-value="draft.enabled"
        label="启用术语表"
        @change="toggleEnabled"
      />
      <UiCheckbox
        :model-value="draft.autoExtractEnabled"
        label="自动添加术语"
        @change="toggleAutoExtractEnabled"
      />
      <ProductStatusBanner class="constraint-modal-body__description" tone="neutral" role="note">
        仅书架模式生效。开启后会在当前页正式翻译前，自动从 OCR
        结果中提取专有名词和人名并写入本书术语表。
      </ProductStatusBanner>
      <UiField
        variant="dialog"
        label="自动术语提取提示词"
        control-id="autoGlossaryPrompt"
      >
        <p class="constraint-modal-body__field-description">
          默认会显示内置提示词，你可以直接在此基础上修改；如果你把内容全部删空后保存，系统会自动恢复为默认提示词。
        </p>
        <UiTextarea
          id="autoGlossaryPrompt"
          :model-value="draft.autoExtractPrompt"
          variant="panel"
          :rows="6"
          placeholder="请输入自动术语提取提示词"
          @update:model-value="updateAutoExtractPrompt"
        />
      </UiField>
      <ProductActionRow
        class="constraint-modal-body__reset-row"
        aria-label="自动术语提取提示词操作"
      >
        <UiButton
          type="button"
          variant="secondary"
          size="sm"
          block
          class="reset-auto-glossary-prompt-btn"
          @click="resetAutoExtractPrompt"
        >
          重置为默认提示词
        </UiButton>
      </ProductActionRow>
      <TranslationConstraintTable
        :model-value="draft.entries"
        :columns="columns"
        :empty-row="emptyRow"
        export-base-name="术语表"
        dedupe-key="source"
        row-key-prefix="book-glossary"
        @update:model-value="updateEntries"
      />
    </div>
    <template #footer>
      <ProductActionRow
        variant="dialog"
        aria-label="术语表操作"
      >
        <UiButton variant="secondary" :disabled="isSaving" @click="handleClose">取消</UiButton>
        <UiButton
          variant="primary"
          :disabled="isSaving"
          data-testid="save-book-glossary-button"
          @click="handleSave"
        >
          保存
        </UiButton>
      </ProductActionRow>
    </template>
  </BaseModal>
</template>

<script setup lang="ts">
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import { computed, ref, watch } from 'vue'

import { DEFAULT_AUTO_GLOSSARY_PROMPT } from '@/constants'
import BaseModal from '@/components/common/BaseModal.vue'
import UiButton from '@/components/ui/UiButton.vue'
import TranslationConstraintTable from '@/components/settings/shared/TranslationConstraintTable.vue'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'
import type { GlossaryEntry } from '@/types/translationConstraints'
import { deepClone } from '@/utils/deepClone'
import { getStringField, validateRegexEntries } from '@/utils/translationConstraintTable'
import { showToast } from '@/utils/toast'

const props = defineProps<{ modelValue: boolean }>()
const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'saved'): void
}>()

const constraintStore = useBookTranslationConstraintsStore()
const isSaving = computed(() => constraintStore.isSaving)
const draft = ref({
  enabled: false,
  autoExtractEnabled: false,
  autoExtractPrompt: DEFAULT_AUTO_GLOSSARY_PROMPT,
  entries: [] as GlossaryEntry[],
})

const columns = [
  { key: 'source', label: '原文' },
  { key: 'target', label: '译文' },
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
  source: '',
  target: '',
  note: '',
  matchMode: 'text',
} satisfies GlossaryEntry

function toMatchMode(value: string): GlossaryEntry['matchMode'] {
  return value === 'regex' ? 'regex' : 'text'
}

function toGlossaryEntry(row: object): GlossaryEntry {
  return {
    source: getStringField(row, 'source'),
    target: getStringField(row, 'target'),
    note: getStringField(row, 'note'),
    matchMode: toMatchMode(getStringField(row, 'matchMode')),
  }
}

watch(
  () => props.modelValue,
  value => {
    if (value) {
      syncDraft()
    }
  },
  { immediate: true }
)

function syncDraft(): void {
  draft.value = deepClone(constraintStore.glossary)
}

function toggleEnabled(checked: boolean): void {
  draft.value.enabled = checked
}

function toggleAutoExtractEnabled(checked: boolean): void {
  draft.value.autoExtractEnabled = checked
}

function updateAutoExtractPrompt(value: string): void {
  draft.value.autoExtractPrompt = value
}

function resetAutoExtractPrompt(): void {
  draft.value.autoExtractPrompt = DEFAULT_AUTO_GLOSSARY_PROMPT
}

function updateEntries(entries: object[]): void {
  draft.value.entries = entries.map(toGlossaryEntry)
}

function handleClose(): void {
  if (isSaving.value) return
  emit('update:modelValue', false)
}

async function handleSave(): Promise<void> {
  const error = validateRegexEntries(draft.value.entries, { patternField: 'source' })
  if (error) {
    showToast(error, 'error')
    return
  }

  try {
    await constraintStore.saveBookConstraints({
      ...deepClone(constraintStore.constraints),
      glossary: deepClone(draft.value),
    })
  } catch (saveError) {
    showToast(saveError instanceof Error ? saveError.message : '保存术语表失败', 'error')
    return
  }

  showToast('术语表已保存', 'success')
  emit('saved')
  handleClose()
}
</script>

<style scoped>
.constraint-modal-body {
  display: flex;
  flex-direction: column;
  gap: 14px;

  --ui-textarea-panel-line-height: normal;
  --ui-textarea-panel-padding: 10px 12px;
  --ui-textarea-height: 120px;
}

.constraint-modal-body__description {
  --product-status-banner-gap: 0;
  --product-status-banner-padding: 0;
  --product-status-banner-border: 0;
  --product-status-banner-background: transparent;
  --product-status-banner-icon-display: none;
  --product-status-banner-body-color: var(--color-text-secondary);
  --product-status-banner-body-font-size: 13px;
}

.constraint-modal-body__reset-row {
  align-items: stretch;
  margin-top: -10px;
  margin-bottom: 13px;
}

.constraint-modal-body__field-description {
  margin: 6px 0 10px;
  color: var(--color-text-supporting);
  font-size: 12px;
  line-height: 1.45;
}

</style>
