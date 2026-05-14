<template>
  <BaseModal
    v-model="isOpen"
    title="术语表"
    size="large"
    :close-on-overlay="true"
    :close-on-esc="true"
    @close="handleClose"
  >
    <div class="constraint-modal-body">
      <div class="constraint-description">
        命中当前文本的术语会追加到翻译提示词中，并在翻译完成后做术语检查。
      </div>
      <label class="checkbox-label">
        <input :checked="draft.enabled" type="checkbox" @change="toggleEnabled" />
        启用术语表
      </label>
      <label class="checkbox-label">
        <input :checked="draft.autoExtractEnabled" type="checkbox" @change="toggleAutoExtractEnabled" />
        自动添加术语
      </label>
      <div class="constraint-description">
        仅书架模式生效。开启后会在当前页正式翻译前，自动从 OCR 结果中提取专有名词和人名并写入本书术语表。
      </div>
      <div class="settings-item">
        <label for="autoGlossaryPrompt">自动术语提取提示词</label>
        <div class="constraint-description">
          默认会显示内置提示词，你可以直接在此基础上修改；如果你把内容全部删空后保存，系统会自动恢复为默认提示词。
        </div>
        <textarea
          id="autoGlossaryPrompt"
          class="auto-glossary-prompt"
          :value="draft.autoExtractPrompt"
          rows="6"
          placeholder="请输入自动术语提取提示词"
          @input="updateAutoExtractPrompt"
        />
        <button type="button" class="btn btn-secondary btn-sm reset-auto-glossary-prompt-btn" @click="resetAutoExtractPrompt">
          重置为默认提示词
        </button>
      </div>
      <TranslationConstraintTable
        :model-value="draft.entries as unknown as Record<string, string>[]"
        :columns="columns"
        :empty-row="emptyRow"
        export-base-name="术语表"
        dedupe-key="source"
        row-key-prefix="book-glossary"
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

import { DEFAULT_AUTO_GLOSSARY_PROMPT } from '@/constants'
import BaseModal from '@/components/common/BaseModal.vue'
import TranslationConstraintTable from '@/components/settings/shared/TranslationConstraintTable.vue'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'
import type { GlossaryEntry } from '@/types/translationConstraints'
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
  draft.value = JSON.parse(JSON.stringify(constraintStore.glossary))
}

function toggleEnabled(event: Event): void {
  draft.value.enabled = (event.target as HTMLInputElement).checked
}

function toggleAutoExtractEnabled(event: Event): void {
  draft.value.autoExtractEnabled = (event.target as HTMLInputElement).checked
}

function updateAutoExtractPrompt(event: Event): void {
  draft.value.autoExtractPrompt = (event.target as HTMLTextAreaElement).value
}

function resetAutoExtractPrompt(): void {
  draft.value.autoExtractPrompt = DEFAULT_AUTO_GLOSSARY_PROMPT
}

function updateEntries(entries: Record<string, string>[]): void {
  draft.value.entries = entries as unknown as GlossaryEntry[]
}

function handleClose(): void {
  isOpen.value = false
  emit('update:modelValue', false)
}

async function handleSave(): Promise<void> {
  const error = validateRegexEntries(draft.value.entries as any, { patternField: 'source' })
  if (error) {
    showToast(error, 'error')
    return
  }

  const ok = await constraintStore.saveBookConstraints({
    ...JSON.parse(JSON.stringify(constraintStore.constraints)),
    glossary: JSON.parse(JSON.stringify(draft.value)),
  })
  if (!ok) {
    showToast('保存术语表失败', 'error')
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

.settings-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.auto-glossary-prompt {
  width: 100%;
  min-height: 120px;
  padding: 10px 12px;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  resize: vertical;
  box-sizing: border-box;
}
</style>
