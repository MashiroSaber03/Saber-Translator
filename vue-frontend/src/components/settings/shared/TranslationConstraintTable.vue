<template>
  <div class="constraint-table">
    <div class="constraint-toolbar">
      <UiInput
        v-model="searchText"
        class="constraint-search"
        type="text"
        placeholder="搜索表格内容..."
      />
      <div class="constraint-actions">
        <UiButton variant="secondary" type="button" @click="addRow" size="sm">新增</UiButton>
        <UiButton variant="secondary" type="button" @click="triggerImport('json')" size="sm">导入 JSON</UiButton>
        <UiButton variant="secondary" type="button" @click="triggerImport('xlsx')" size="sm">导入 XLSX</UiButton>
        <UiButton variant="secondary" type="button" @click="exportJson" size="sm">导出 JSON</UiButton>
        <UiButton variant="secondary" type="button" @click="exportXlsx" size="sm">导出 XLSX</UiButton>
      </div>
      <UiFileInput
        ref="jsonImportInput"
        class="hidden-input"
        accept=".json,application/json"
        @change="handleImport($event, 'json')"
      />
      <UiFileInput
        ref="xlsxImportInput"
        class="hidden-input"
        accept=".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        @change="handleImport($event, 'xlsx')"
      />
    </div>

    <table class="settings-table">
      <thead>
        <tr>
          <th
            v-for="column in columns"
            :key="column.key"
            :aria-sort="sortKey === column.key ? (sortDirection === 'asc' ? 'ascending' : 'descending') : 'none'"
          >
            <UiButton
              variant="toolbar"
              type="button"
              class="sortable-header-button"
              :aria-label="`按${column.label}排序`"
              @click="toggleSort(column.key)"
            >
              {{ column.label }}
            </UiButton>
          </th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="({ row, originalIndex }, index) in filteredRows" :key="`${rowKeyPrefix}-${index}`">
          <td v-for="column in columns" :key="column.key">
            <div
              v-if="column.type === 'select'"
              class="select-cell"
            >
              <CustomSelect
                :model-value="String(row[column.key] ?? '')"
                :options="column.options || []"
                fit
                @change="updateCell(originalIndex, column.key, String($event))"
              />
            </div>
            <UiTextarea
              v-else-if="column.type === 'textarea'"
              class="constraint-cell-field"
              :model-value="String(row[column.key] ?? '')"
              :rows="2"
              @update:model-value="updateCell(originalIndex, column.key, $event)"
            />
            <UiInput
              v-else
              class="constraint-cell-field"
              :value="String(row[column.key] ?? '')"
              type="text"
              @input="updateCell(originalIndex, column.key, ($event.target as HTMLInputElement).value)"
            />
          </td>
          <td class="action-cell">
            <UiButton variant="danger" type="button" @click="removeRow(originalIndex)" size="sm">删除</UiButton>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup lang="ts">
import UiInput from '@/components/ui/UiInput.vue'
import UiFileInput from '@/components/ui/UiFileInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import { computed, ref } from 'vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { useToast } from '@/utils/toast'
import {
  exportRowsToJson,
  exportRowsToXlsxBuffer,
  importRowsFromJson,
  importRowsFromXlsxBuffer,
  type TranslationConstraintColumn,
} from '@/utils/translationConstraintTable'

type TableRow = Record<string, string>

type EditableColumn = TranslationConstraintColumn & {
  type?: 'text' | 'textarea' | 'select'
  options?: Array<{ label: string; value: string }>
}

const props = defineProps<{
  modelValue: TableRow[]
  columns: EditableColumn[]
  emptyRow: TableRow
  exportBaseName: string
  rowKeyPrefix?: string
  dedupeKey?: string
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: TableRow[]): void
}>()

const toast = useToast()
const searchText = ref('')
const sortKey = ref<string>('')
const sortDirection = ref<'asc' | 'desc'>('asc')
const jsonImportInput = ref<HTMLInputElement | null>(null)
const xlsxImportInput = ref<HTMLInputElement | null>(null)

const filteredRows = computed(() => {
  const rowsWithIndex = props.modelValue.map((row, originalIndex) => ({ row, originalIndex }))
  const query = searchText.value.trim().toLowerCase()
  const rows = query
    ? rowsWithIndex.filter(({ row }) =>
        props.columns.some(column =>
          String(row[column.key] ?? '').toLowerCase().includes(query),
        ),
      )
    : rowsWithIndex

  if (!sortKey.value) {
    return rows
  }

  return [...rows].sort((left, right) => {
    const leftValue = String(left.row[sortKey.value] ?? '').toLowerCase()
    const rightValue = String(right.row[sortKey.value] ?? '').toLowerCase()
    const compare = leftValue.localeCompare(rightValue, undefined, { numeric: true })
    return sortDirection.value === 'asc' ? compare : compare * -1
  })
})

function emitRows(rows: TableRow[]): void {
  emit('update:modelValue', rows)
}

function addRow(): void {
  emitRows([...props.modelValue, { ...props.emptyRow }])
}

function removeRow(index: number): void {
  const nextRows = [...props.modelValue]
  nextRows.splice(index, 1)
  emitRows(nextRows)
}

function updateCell(index: number, key: string, value: string): void {
  const nextRows = props.modelValue.map((row, rowIndex) =>
    rowIndex === index
      ? {
          ...row,
          [key]: value,
        }
      : row,
  )
  emitRows(nextRows)
}

function toggleSort(key: string): void {
  if (sortKey.value === key) {
    sortDirection.value = sortDirection.value === 'asc' ? 'desc' : 'asc'
    return
  }

  sortKey.value = key
  sortDirection.value = 'asc'
}

function triggerImport(format: 'json' | 'xlsx'): void {
  if (format === 'json') {
    jsonImportInput.value?.click()
    return
  }
  xlsxImportInput.value?.click()
}

async function handleImport(event: Event, format: 'json' | 'xlsx'): Promise<void> {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) {
    return
  }

  try {
    const importedRows = format === 'json'
      ? importRowsFromJson(await file.text(), props.columns)
      : importRowsFromXlsxBuffer(await file.arrayBuffer(), props.columns)
    emitRows(mergeImportedRows(importedRows))
    toast.success(`已导入 ${importedRows.length} 条记录`)
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '导入失败')
  } finally {
    input.value = ''
  }
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

function exportJson(): void {
  const blob = new Blob([exportRowsToJson(props.modelValue)], {
    type: 'application/json;charset=utf-8',
  })
  downloadBlob(blob, `${props.exportBaseName}.json`)
}

function exportXlsx(): void {
  const buffer = exportRowsToXlsxBuffer(props.modelValue, props.columns)
  const blob = new Blob([buffer], {
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  })
  downloadBlob(blob, `${props.exportBaseName}.xlsx`)
}

function mergeImportedRows(importedRows: TableRow[]): TableRow[] {
  if (!props.dedupeKey) {
    return importedRows
  }

  const existingRows = [...props.modelValue]
  const existingKeys = new Set(
    existingRows
      .map(row => String(row[props.dedupeKey as string] ?? '').trim())
      .filter(Boolean),
  )

  const mergedRows = [...existingRows]
  for (const row of importedRows) {
    const key = String(row[props.dedupeKey] ?? '').trim()
    if (!key || existingKeys.has(key)) {
      continue
    }
    existingKeys.add(key)
    mergedRows.push(row)
  }
  return mergedRows
}
</script>

<style scoped>
.constraint-table {
  --translation-constraint-table-border-default: #cfd6e4;
  --translation-constraint-table-border-strong: #5b73f2;
  --translation-constraint-table-shadow-default: rgba(88, 125, 255, .18);
  --translation-constraint-table-text-primary: #1f2430;
}

.constraint-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.constraint-search {
  flex: 1 1 260px;
  min-width: 220px;
  min-height: 40px;
  padding: 0 12px;
  border: 1px solid var(--translation-constraint-table-border-default);
  border-radius: 8px;
  background: var(--color-surface-plain);
  color: var(--translation-constraint-table-text-primary);
  font-size: 14px;
  transition: border-color 0.15s, box-shadow 0.15s;
  box-sizing: border-box;
}

.constraint-search:focus {
  outline: none;
  border-color: var(--translation-constraint-table-border-strong);
  box-shadow: 0 0 0 2px var(--translation-constraint-table-shadow-default);
}

.constraint-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.hidden-input {
  display: none;
}

.settings-table {
  width: 100%;
  border-collapse: collapse;
}

.settings-table th,
.settings-table td {
  border: 1px solid var(--color-border-muted);
  padding: 8px;
  vertical-align: top;
}

.constraint-cell-field {
  width: 100%;
  min-height: 40px;
  padding: 0 12px;
  border: 1px solid var(--translation-constraint-table-border-default);
  border-radius: 8px;
  background: var(--color-surface-plain);
  color: var(--translation-constraint-table-text-primary);
  font-size: 14px;
  transition: border-color 0.15s, box-shadow 0.15s;
  box-sizing: border-box;
}

.constraint-cell-field:is(textarea) {
  min-height: 72px;
  padding: 10px 12px;
  resize: vertical;
}

.constraint-cell-field:focus {
  outline: none;
  border-color: var(--translation-constraint-table-border-strong);
  box-shadow: 0 0 0 2px var(--translation-constraint-table-shadow-default);
}

.select-cell {
  min-width: 0;
}

.sortable-header-button {
  width: 100%;
  color: inherit;
  font-weight: inherit;
  text-align: left;
  user-select: none;
}

.action-cell {
  width: 88px;
}
</style>
