<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import AppHeader from '@/components/common/AppHeader.vue'
import {
  createGlossaryEntry,
  deleteGlossaryEntry,
  getGlossaryEntries,
  updateGlossaryEntry,
  type GlossaryCreatePayload
} from '@/api/glossary'
import type { GlossaryEntry } from '@/types'
import { showToast } from '@/utils/toast'

const loading = ref(false)
const entries = ref<GlossaryEntry[]>([])
const query = ref('')
const filterBookId = ref('')

const form = reactive<GlossaryCreatePayload>({
  source_text: '',
  target_text: '',
  note: '',
  entry_type: 'term',
  scope: 'global',
  book_id: '',
  enabled: true
})

const editingId = ref<string | null>(null)
const editDraft = reactive<Partial<GlossaryCreatePayload>>({})

async function loadEntries(): Promise<void> {
  loading.value = true
  try {
    const res = await getGlossaryEntries({
      query: query.value || undefined,
      book_id: filterBookId.value || undefined,
      include_global: true
    })
    if (!res.success) throw new Error(res.error || '加载术语失败')
    entries.value = res.entries || []
  } catch (error) {
    showToast((error as Error).message, 'error')
  } finally {
    loading.value = false
  }
}

async function createEntry(): Promise<void> {
  if (!form.source_text?.trim() || !form.target_text?.trim()) {
    showToast('请填写原文和译文', 'warning')
    return
  }
  if (form.scope === 'book' && !form.book_id?.trim()) {
    showToast('book 级术语需要 book_id', 'warning')
    return
  }
  try {
    const payload: GlossaryCreatePayload = {
      ...form,
      source_text: form.source_text.trim(),
      target_text: form.target_text.trim(),
      note: form.note?.trim() || '',
      book_id: form.scope === 'book' ? form.book_id?.trim() : undefined
    }
    const res = await createGlossaryEntry(payload)
    if (!res.success || !res.entry) throw new Error(res.error || '创建失败')
    showToast('术语条目已创建', 'success')
    form.source_text = ''
    form.target_text = ''
    form.note = ''
    await loadEntries()
  } catch (error) {
    showToast((error as Error).message, 'error')
  }
}

function startEdit(entry: GlossaryEntry): void {
  editingId.value = entry.id
  editDraft.source_text = entry.source_text
  editDraft.target_text = entry.target_text
  editDraft.note = entry.note || ''
  editDraft.scope = entry.scope
  editDraft.book_id = entry.book_id || ''
  editDraft.entry_type = entry.entry_type
  editDraft.enabled = entry.enabled
}

function cancelEdit(): void {
  editingId.value = null
  Object.keys(editDraft).forEach(k => {
    delete (editDraft as Record<string, unknown>)[k]
  })
}

async function saveEdit(entryId: string): Promise<void> {
  try {
    const payload: Partial<GlossaryCreatePayload> = {
      source_text: editDraft.source_text?.trim(),
      target_text: editDraft.target_text?.trim(),
      note: editDraft.note || '',
      scope: editDraft.scope,
      entry_type: editDraft.entry_type,
      enabled: editDraft.enabled,
      book_id: editDraft.scope === 'book' ? editDraft.book_id?.trim() : undefined
    }
    const res = await updateGlossaryEntry(entryId, payload)
    if (!res.success) throw new Error(res.error || '更新失败')
    showToast('条目已更新', 'success')
    cancelEdit()
    await loadEntries()
  } catch (error) {
    showToast((error as Error).message, 'error')
  }
}

async function removeEntry(entryId: string): Promise<void> {
  if (!confirm('确认删除此条目？')) return
  try {
    const res = await deleteGlossaryEntry(entryId)
    if (!res.success) throw new Error(res.error || '删除失败')
    showToast('条目已删除', 'success')
    await loadEntries()
  } catch (error) {
    showToast((error as Error).message, 'error')
  }
}

onMounted(() => {
  loadEntries()
})
</script>

<template>
  <div class="glossary-page">
    <AppHeader variant="bookshelf" />

    <main class="glossary-main">
      <section class="panel">
        <h1>术语库与翻译记忆</h1>
        <p>维护书籍级术语与全局翻译记忆，供质量分析和翻译前建议使用。</p>

        <div class="create-grid">
          <div class="field">
            <label for="glossary-source">原文术语</label>
            <input id="glossary-source" v-model="form.source_text" placeholder="原文术语" />
          </div>
          <div class="field">
            <label for="glossary-target">建议译文</label>
            <input id="glossary-target" v-model="form.target_text" placeholder="建议译文" />
          </div>
          <div class="field">
            <label for="glossary-entry-type">条目类型</label>
            <select id="glossary-entry-type" v-model="form.entry_type">
              <option value="term">术语</option>
              <option value="memory">翻译记忆</option>
            </select>
          </div>
          <div class="field">
            <label for="glossary-scope">作用范围</label>
            <select id="glossary-scope" v-model="form.scope">
              <option value="global">全局</option>
              <option value="book">书籍级</option>
            </select>
          </div>
          <div v-if="form.scope === 'book'" class="field">
            <label for="glossary-book-id">book_id</label>
            <input id="glossary-book-id" v-model="form.book_id" placeholder="book_id" />
          </div>
          <div class="field">
            <label for="glossary-note">备注（可选）</label>
            <input id="glossary-note" v-model="form.note" placeholder="备注（可选）" />
          </div>
          <div class="field action-field">
            <span class="sr-only">创建条目</span>
            <button type="button" @click="createEntry">新增条目</button>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="filter-row">
          <div class="field">
            <label for="glossary-query" class="sr-only">搜索条目</label>
            <input id="glossary-query" v-model="query" placeholder="按原文/译文/备注搜索" />
          </div>
          <div class="field">
            <label for="glossary-filter-book" class="sr-only">筛选 book_id</label>
            <input id="glossary-filter-book" v-model="filterBookId" placeholder="筛选 book_id（可选）" />
          </div>
          <button type="button" :disabled="loading" @click="loadEntries">刷新</button>
        </div>

        <div class="table-wrap">
          <table>
            <caption class="sr-only">术语与翻译记忆条目列表</caption>
            <thead>
              <tr>
                <th scope="col">类型</th>
                <th scope="col">范围</th>
                <th scope="col">原文</th>
                <th scope="col">译文</th>
                <th scope="col">备注</th>
                <th scope="col">操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="entry in entries" :key="entry.id">
                <td>{{ entry.entry_type }}</td>
                <td>{{ entry.scope }} <span v-if="entry.book_id">({{ entry.book_id }})</span></td>
                <td v-if="editingId !== entry.id">{{ entry.source_text }}</td>
                <td v-else>
                  <label :for="`edit-source-${entry.id}`" class="sr-only">编辑原文</label>
                  <input :id="`edit-source-${entry.id}`" v-model="editDraft.source_text" />
                </td>
                <td v-if="editingId !== entry.id">{{ entry.target_text }}</td>
                <td v-else>
                  <label :for="`edit-target-${entry.id}`" class="sr-only">编辑译文</label>
                  <input :id="`edit-target-${entry.id}`" v-model="editDraft.target_text" />
                </td>
                <td v-if="editingId !== entry.id">{{ entry.note || '-' }}</td>
                <td v-else>
                  <label :for="`edit-note-${entry.id}`" class="sr-only">编辑备注</label>
                  <input :id="`edit-note-${entry.id}`" v-model="editDraft.note" />
                </td>
                <td class="actions">
                  <template v-if="editingId !== entry.id">
                    <button type="button" @click="startEdit(entry)">编辑</button>
                    <button type="button" class="danger" @click="removeEntry(entry.id)">删除</button>
                  </template>
                  <template v-else>
                    <button type="button" @click="saveEdit(entry.id)">保存</button>
                    <button type="button" class="secondary" @click="cancelEdit">取消</button>
                  </template>
                </td>
              </tr>
              <tr v-if="entries.length === 0">
                <td colspan="6" class="empty">暂无条目</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </main>
  </div>
</template>

<style scoped>
.glossary-page {
  min-height: 100vh;
  background: linear-gradient(160deg, #f6fbf8 0%, #f7f7ff 100%);
}

.glossary-main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  display: grid;
  gap: 16px;
}

.panel {
  background: #fff;
  border: 1px solid #dbe5f1;
  border-radius: 14px;
  padding: 16px;
}

.panel h1 {
  margin: 0 0 8px;
}

.panel p {
  margin: 0 0 12px;
  color: #556579;
}

.create-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 10px;
  align-items: center;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.field label {
  font-size: 13px;
  color: #556579;
}

.action-field {
  justify-content: flex-end;
}

.create-grid input,
.create-grid select,
.filter-row input {
  border: 1px solid #c8d6e6;
  border-radius: 10px;
  padding: 9px 10px;
}

button {
  border: none;
  border-radius: 10px;
  background: #1677ff;
  color: #fff;
  padding: 9px 12px;
  cursor: pointer;
}

button.secondary {
  background: #5d6d83;
}

button.danger {
  background: #e74c3c;
}

.filter-row {
  display: grid;
  grid-template-columns: 2fr 1fr auto;
  gap: 10px;
  align-items: end;
  margin-bottom: 10px;
}

.table-wrap {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th,
td {
  border-bottom: 1px solid #edf1f6;
  text-align: left;
  padding: 10px 8px;
  font-size: 14px;
}

td input {
  width: 100%;
  border: 1px solid #c8d6e6;
  border-radius: 8px;
  padding: 6px 8px;
}

.actions {
  display: flex;
  gap: 6px;
  white-space: nowrap;
}

.empty {
  text-align: center;
  color: #5a6b80;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

@media (max-width: 768px) {
  .glossary-main {
    padding: 12px;
  }

  .filter-row {
    grid-template-columns: 1fr;
  }
}
</style>
