<template>
  <div class="translation-constraint-table">
    <div class="translation-constraint-table__toolbar">
      <ProductSearchField
        v-model="searchText"
        class="translation-constraint-table__search-field"
        aria-label="搜索约束表格"
        placeholder="搜索表格内容..."
      />
      <ProductActionRow
        class="translation-constraint-table__actions"
        aria-label="约束表格操作"
        justify="start"
      >
        <UiButton variant="secondary" type="button" @click="addRow" size="sm">新增</UiButton>
        <UiButton variant="secondary" type="button" @click="triggerImport('json')" size="sm">
          导入 JSON
        </UiButton>
        <UiButton variant="secondary" type="button" @click="triggerImport('xlsx')" size="sm">
          导入 XLSX
        </UiButton>
        <UiButton variant="secondary" type="button" @click="exportJson" size="sm">
          导出 JSON
        </UiButton>
        <UiButton variant="secondary" type="button" @click="exportXlsx" size="sm">
          导出 XLSX
        </UiButton>
      </ProductActionRow>
      <UiFileInput
        ref="jsonImportInput"
        accept=".json,application/json"
        hidden
        @files-change="handleImport($event, 'json')"
      />
      <UiFileInput
        ref="xlsxImportInput"
        accept=".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        hidden
        @files-change="handleImport($event, 'xlsx')"
      />
    </div>

    <table class="translation-constraint-table__table">
      <thead>
        <tr>
          <th
            v-for="column in columns"
            :key="column.key"
            :aria-sort="
              sortKey === column.key
                ? sortDirection === 'asc'
                  ? 'ascending'
                  : 'descending'
                : 'none'
            "
          >
            <UiButton
              variant="toolbar"
              type="button"
              class="translation-constraint-table__sort-action"
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
        <tr
          v-for="({ row, originalIndex }, index) in filteredRows"
          :key="`${rowKeyPrefix}-${index}`"
        >
          <td v-for="column in columns" :key="column.key">
            <div v-if="column.type === 'select'" class="translation-constraint-table__select-cell">
              <UiSelect
                :model-value="getCellValue(row, column.key)"
                :options="column.options || []"
                @change="updateCell(originalIndex, column.key, String($event))"
              />
            </div>
            <UiTextarea
              v-else-if="column.type === 'textarea'"
              class="translation-constraint-table__cell-field"
              :model-value="getCellValue(row, column.key)"
              :rows="2"
              @update:model-value="updateCell(originalIndex, column.key, $event)"
            />
            <UiInput
              v-else
              class="translation-constraint-table__cell-field"
              :model-value="getCellValue(row, column.key)"
              type="text"
              @update:model-value="updateCell(originalIndex, column.key, $event)"
            />
          </td>
          <td class="translation-constraint-table__action-cell">
            <UiButton variant="danger" type="button" @click="removeRow(originalIndex)" size="sm">
              删除
            </UiButton>
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
import UiSelect from '@/components/ui/UiSelect.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductSearchField from '@/components/product/ProductSearchField.vue'
import { computed, ref } from 'vue'
import { useToast } from '@/utils/toast'
import { triggerBlobDownload } from '@/utils/browserDownload'
import {
  exportRowsToJson,
  exportRowsToXlsxBuffer,
  importRowsFromJson,
  importRowsFromXlsxBuffer,
  getStringField,
  type TranslationConstraintColumn,
} from '@/utils/translationConstraintTable'

type TableRow = object

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
const jsonImportInput = ref<InstanceType<typeof UiFileInput> | null>(null)
const xlsxImportInput = ref<InstanceType<typeof UiFileInput> | null>(null)

function getCellValue(row: TableRow, key: string): string {
  return getStringField(row, key)
}

function withCellValue(row: TableRow, key: string, value: string): TableRow {
  return {
    ...row,
    [key]: value,
  }
}

function toStringRows(rows: TableRow[]): Record<string, string>[] {
  return rows.map(row => {
    const nextRow: Record<string, string> = {}
    for (const column of props.columns) {
      nextRow[column.key] = getCellValue(row, column.key)
    }
    return nextRow
  })
}

const filteredRows = computed(() => {
  const rowsWithIndex = props.modelValue.map((row, originalIndex) => ({ row, originalIndex }))
  const query = searchText.value.trim().toLowerCase()
  const rows = query
    ? rowsWithIndex.filter(({ row }) =>
        props.columns.some(column => getCellValue(row, column.key).toLowerCase().includes(query))
      )
    : rowsWithIndex

  if (!sortKey.value) {
    return rows
  }

  return [...rows].sort((left, right) => {
    const leftValue = getCellValue(left.row, sortKey.value).toLowerCase()
    const rightValue = getCellValue(right.row, sortKey.value).toLowerCase()
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

function updateCell(index: number, key: string, value: string | number | boolean): void {
  const nextValue = String(value)
  const nextRows = props.modelValue.map((row, rowIndex) =>
    rowIndex === index ? withCellValue(row, key, nextValue) : row
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

async function handleImport(files: File[], format: 'json' | 'xlsx'): Promise<void> {
  const file = files[0]
  if (!file) {
    return
  }

  try {
    const importedRows =
      format === 'json'
        ? importRowsFromJson(await file.text(), props.columns)
        : importRowsFromXlsxBuffer(await file.arrayBuffer(), props.columns)
    emitRows(mergeImportedRows(importedRows))
    toast.success(`已导入 ${importedRows.length} 条记录`)
  } catch (error) {
    toast.error(error instanceof Error ? error.message : '导入失败')
  } finally {
    clearImportInput(format)
  }
}

function clearImportInput(format: 'json' | 'xlsx'): void {
  const inputRef = format === 'json' ? jsonImportInput : xlsxImportInput
  inputRef.value?.clear()
}

function exportJson(): void {
  const blob = new Blob([exportRowsToJson(toStringRows(props.modelValue))], {
    type: 'application/json;charset=utf-8',
  })
  triggerBlobDownload(blob, `${props.exportBaseName}.json`)
}

function exportXlsx(): void {
  const buffer = exportRowsToXlsxBuffer(toStringRows(props.modelValue), props.columns)
  const blob = new Blob([buffer], {
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  })
  triggerBlobDownload(blob, `${props.exportBaseName}.xlsx`)
}

function mergeImportedRows(importedRows: TableRow[]): TableRow[] {
  if (!props.dedupeKey) {
    return importedRows
  }

  const existingRows = [...props.modelValue]
  const existingKeys = new Set(
    existingRows.map(row => getCellValue(row, props.dedupeKey as string).trim()).filter(Boolean)
  )

  const mergedRows = [...existingRows]
  for (const row of importedRows) {
    const key = getCellValue(row, props.dedupeKey).trim()
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
.translation-constraint-table {
  --translation-constraint-table-field-border: var(--color-border-input);
  --translation-constraint-table-field-focus-border: var(--color-border-brand);
  --translation-constraint-table-field-focus-ring: var(--color-focus-brand-subtle);
  --translation-constraint-table-field-text: var(--color-text-strong);
}

.translation-constraint-table__toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.translation-constraint-table__search-field {
  flex: 1 1 260px;
  min-width: 220px;
}

.translation-constraint-table__actions {
  gap: 8px;
}

.translation-constraint-table__table {
  width: 100%;
  border-collapse: collapse;
}

.translation-constraint-table__table th,
.translation-constraint-table__table td {
  border: 1px solid var(--color-border-muted);
  padding: 8px;
  vertical-align: top;
}

.translation-constraint-table__cell-field {
  width: 100%;
  min-height: 40px;
  padding: 0 12px;
  border: 1px solid var(--translation-constraint-table-field-border);
  border-radius: 8px;
  background: var(--color-surface-base);
  color: var(--translation-constraint-table-field-text);
  font-size: 14px;
  transition:
    border-color 0.15s,
    box-shadow 0.15s;
  box-sizing: border-box;
}

.translation-constraint-table__cell-field:is(textarea) {
  min-height: 72px;
  padding: 10px 12px;
  resize: vertical;
}

.translation-constraint-table__cell-field:focus {
  outline: none;
  border-color: var(--translation-constraint-table-field-focus-border);
  box-shadow: 0 0 0 2px var(--translation-constraint-table-field-focus-ring);
}

.translation-constraint-table__select-cell {
  min-width: 0;
}

.translation-constraint-table__sort-action {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  color: inherit;
  font-weight: inherit;
  text-align: center;
  user-select: none;
}

.translation-constraint-table__action-cell {
  width: 88px;
}
</style>
