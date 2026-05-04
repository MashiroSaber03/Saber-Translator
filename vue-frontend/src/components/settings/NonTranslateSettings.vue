<template>
  <div class="non-translate-settings">
    <div class="settings-group">
      <div class="settings-group-title">禁翻表</div>
      <div class="settings-item">
        <label class="checkbox-label">
          <input :checked="settings.enabled" type="checkbox" @change="toggleEnabled" />
          启用禁翻表
        </label>
        <div class="input-hint">
          命中当前文本的禁翻内容会被保护为占位符，翻译完成后再还原。
        </div>
      </div>
      <div class="settings-item">
        <TranslationConstraintTable
          :model-value="settings.entries"
          :columns="columns"
          :empty-row="emptyRow"
          export-base-name="禁翻表"
          dedupe-key="pattern"
          row-key-prefix="non-translate"
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
import type { NonTranslateEntry } from '@/types/translationConstraints'
import { validateRegexEntries } from '@/utils/translationConstraintTable'

const settingsStore = useSettingsStore()

const settings = computed(() => settingsStore.settings.nonTranslate)
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

function toggleEnabled(event: Event): void {
  settingsStore.setNonTranslateEnabled((event.target as HTMLInputElement).checked)
}

function updateEntries(entries: Record<string, string>[]): void {
  settingsStore.setNonTranslateEntries(entries as unknown as NonTranslateEntry[])
}

function validateSettings(): { success: boolean; error?: string } {
  const error = validateRegexEntries(settings.value.entries as any, { patternField: 'pattern' })
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
