<template>
  <div class="glossary-settings">
    <div class="settings-group">
      <div class="settings-group-title">术语表</div>
      <div class="settings-item">
        <label class="checkbox-label">
          <input :checked="settings.enabled" type="checkbox" @change="toggleEnabled" />
          启用术语表
        </label>
        <div class="input-hint">
          命中当前文本的术语会追加到翻译提示词中，并在翻译完成后做术语检查。
        </div>
      </div>
      <div class="settings-item">
        <TranslationConstraintTable
          :model-value="settings.entries"
          :columns="columns"
          :empty-row="emptyRow"
          export-base-name="术语表"
          dedupe-key="source"
          row-key-prefix="glossary"
          @update:model-value="updateEntries"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useSettingsStore } from '@/stores/settingsStore'
import TranslationConstraintTable from './shared/TranslationConstraintTable.vue'
import type { GlossaryEntry } from '@/types/translationConstraints'
import { validateRegexEntries } from '@/utils/translationConstraintTable'

const settingsStore = useSettingsStore()

const settings = computed(() => settingsStore.settings.glossary)
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

function toggleEnabled(event: Event): void {
  settingsStore.setGlossaryEnabled((event.target as HTMLInputElement).checked)
}

function updateEntries(entries: Record<string, string>[]): void {
  settingsStore.setGlossaryEntries(entries as unknown as GlossaryEntry[])
}

function validateSettings(): { success: boolean; error?: string } {
  const error = validateRegexEntries(settings.value.entries as any, { patternField: 'source' })
  if (error) {
    return { success: false, error }
  }
  return { success: true }
}

defineExpose({
  validateSettings,
})
</script>

<style scoped>
.checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
}

.input-hint {
  margin-top: 8px;
  color: var(--text-secondary);
  font-size: 12px;
}
</style>
