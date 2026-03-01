<script setup lang="ts">
import { reactive, ref } from 'vue'
import AppHeader from '@/components/common/AppHeader.vue'
import { getCompareRun, runCompare } from '@/api/compare'
import type { CompareRun } from '@/types'
import { showToast } from '@/utils/toast'

interface CandidateDraft {
  id: string
  name: string
  text: string
}

const loading = ref(false)
const runIdInput = ref('')
const runResult = ref<CompareRun | null>(null)

const form = reactive({
  session: '',
  pageIndex: '',
  baselineName: '基线版本',
  baselineText: ''
})

const candidates = ref<CandidateDraft[]>([
  { id: 'candidate_1', name: '模型A', text: '' },
  { id: 'candidate_2', name: '模型B', text: '' }
])

function addCandidate(): void {
  const index = candidates.value.length + 1
  candidates.value.push({
    id: `candidate_${index}`,
    name: `候选${index}`,
    text: ''
  })
}

function removeCandidate(id: string): void {
  if (candidates.value.length <= 1) {
    showToast('至少保留一个候选版本', 'warning')
    return
  }
  candidates.value = candidates.value.filter(item => item.id !== id)
}

function setAsBaseline(text: string, name: string): void {
  form.baselineText = text
  form.baselineName = `${name}（设为基线）`
  showToast('已设为新的基线版本', 'success')
}

async function runComparison(): Promise<void> {
  if (!form.baselineText.trim()) {
    showToast('请先填写基线译文', 'warning')
    return
  }
  const validCandidates = candidates.value
    .filter(item => item.text.trim())
    .map(item => ({ id: item.id, name: item.name, text: item.text }))
  if (validCandidates.length === 0) {
    showToast('请至少填写一个候选译文', 'warning')
    return
  }

  loading.value = true
  try {
    const res = await runCompare({
      session: form.session.trim() || undefined,
      page_index: form.pageIndex ? Number(form.pageIndex) : undefined,
      baseline: {
        id: 'baseline',
        name: form.baselineName.trim() || '基线',
        text: form.baselineText
      },
      candidates: validCandidates
    })
    if (!res.success || !res.run) {
      throw new Error(res.error || '对比失败')
    }
    runResult.value = res.run
    runIdInput.value = res.run.id
    showToast('版本对比完成', 'success')
  } catch (error) {
    showToast((error as Error).message, 'error')
  } finally {
    loading.value = false
  }
}

async function loadRunById(): Promise<void> {
  if (!runIdInput.value.trim()) {
    showToast('请输入 run_id', 'warning')
    return
  }
  loading.value = true
  try {
    const res = await getCompareRun(runIdInput.value.trim())
    if (!res.success || !res.run) {
      throw new Error(res.error || '未找到对比结果')
    }
    runResult.value = res.run
    showToast('已加载对比结果', 'success')
  } catch (error) {
    showToast((error as Error).message, 'error')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="compare-page">
    <AppHeader variant="bookshelf" />

    <main class="compare-main">
      <section class="panel">
        <h1>版本对比</h1>
        <p>同页多版本译文对比，支持设为基线并持久化查询。</p>

        <div class="top-row">
          <div class="field">
            <label for="compare-session">session（可选）</label>
            <input id="compare-session" v-model="form.session" placeholder="session（可选）" />
          </div>
          <div class="field">
            <label for="compare-page-index">page_index（可选）</label>
            <input id="compare-page-index" v-model="form.pageIndex" placeholder="page_index（可选）" />
          </div>
        </div>

        <label class="label" for="compare-baseline-name">基线名称</label>
        <input id="compare-baseline-name" v-model="form.baselineName" placeholder="基线名称" />
        <label class="label" for="compare-baseline-text">基线译文</label>
        <textarea id="compare-baseline-text" v-model="form.baselineText" rows="4" placeholder="输入基线译文"></textarea>

        <div class="candidates">
          <div v-for="candidate in candidates" :key="candidate.id" class="candidate-item">
            <label :for="`compare-candidate-name-${candidate.id}`" class="sr-only">候选名称</label>
            <input :id="`compare-candidate-name-${candidate.id}`" v-model="candidate.name" placeholder="候选名称" />
            <label :for="`compare-candidate-text-${candidate.id}`" class="sr-only">候选译文</label>
            <textarea :id="`compare-candidate-text-${candidate.id}`" v-model="candidate.text" rows="3" placeholder="候选译文"></textarea>
            <button type="button" class="danger" @click="removeCandidate(candidate.id)">移除</button>
          </div>
        </div>

        <div class="actions">
          <button type="button" :disabled="loading" @click="addCandidate">添加候选</button>
          <button type="button" :disabled="loading" @click="runComparison">开始对比</button>
        </div>
      </section>

      <section class="panel">
        <h2>读取历史 Run</h2>
        <div class="top-row">
          <div class="field">
            <label for="compare-run-id">run_id</label>
            <input id="compare-run-id" v-model="runIdInput" placeholder="输入 run_id" />
          </div>
          <div class="field field-action">
            <span class="sr-only">加载历史对比结果</span>
            <button type="button" :disabled="loading" @click="loadRunById">加载</button>
          </div>
        </div>
      </section>

      <section v-if="runResult" class="panel">
        <h2>对比结果：{{ runResult.id }}</h2>
        <p class="baseline"><strong>基线：</strong>{{ runResult.baseline.name }}</p>
        <pre class="base-text">{{ runResult.baseline.text }}</pre>

        <article v-for="candidate in runResult.candidates" :key="candidate.id" class="result-item">
          <header>
            <strong>{{ candidate.name }}</strong>
            <span>相似度 {{ (candidate.similarity * 100).toFixed(2) }}%</span>
          </header>
          <pre>{{ candidate.text }}</pre>
          <button type="button" class="secondary" @click="setAsBaseline(candidate.text, candidate.name)">
            设为基线
          </button>
          <div class="diff-list">
            <div v-for="(seg, idx) in candidate.diff_segments" :key="`${candidate.id}-${idx}`" class="diff-item">
              <span class="op" :class="`op-${seg.op}`">{{ seg.op }}</span>
              <div class="diff-content">
                <div><strong>基线片段:</strong> {{ seg.base || '∅' }}</div>
                <div><strong>候选片段:</strong> {{ seg.candidate || '∅' }}</div>
              </div>
            </div>
          </div>
        </article>
      </section>
    </main>
  </div>
</template>

<style scoped>
.compare-page {
  min-height: 100vh;
  background: linear-gradient(150deg, #f6fbff 0%, #f9f6ff 100%);
}

.compare-main {
  max-width: 1100px;
  margin: 0 auto;
  padding: 20px;
  display: grid;
  gap: 16px;
}

.panel {
  background: #fff;
  border: 1px solid #dce6f2;
  border-radius: 14px;
  padding: 16px;
}

.panel h1,
.panel h2 {
  margin: 0 0 8px;
}

.panel p {
  margin: 0 0 12px;
  color: #5a6a7f;
}

.top-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-bottom: 10px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.field-action {
  justify-content: flex-end;
}

input,
textarea {
  width: 100%;
  border: 1px solid #c8d6e6;
  border-radius: 10px;
  padding: 9px 10px;
  box-sizing: border-box;
}

.label {
  display: block;
  font-size: 13px;
  color: #5a6a7f;
  margin: 8px 0 6px;
}

.field label {
  font-size: 13px;
  color: #5a6a7f;
}

.candidates {
  display: grid;
  gap: 10px;
  margin-top: 12px;
}

.candidate-item {
  border: 1px solid #e6edf5;
  border-radius: 10px;
  padding: 10px;
  display: grid;
  gap: 8px;
}

.actions {
  margin-top: 12px;
  display: flex;
  gap: 8px;
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
  background: #5f6f85;
}

button.danger {
  background: #dd4d3e;
}

button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.baseline {
  margin: 0 0 8px;
}

.base-text,
.result-item pre {
  white-space: pre-wrap;
  background: #f6f9fc;
  border-radius: 10px;
  padding: 10px;
  margin: 0 0 10px;
}

.result-item {
  border: 1px solid #e6edf5;
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 10px;
}

.result-item header {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 8px;
}

.diff-list {
  display: grid;
  gap: 6px;
  margin-top: 10px;
}

.diff-item {
  border: 1px dashed #d5e1ef;
  border-radius: 8px;
  padding: 8px;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 8px;
  align-items: start;
}

.op {
  border-radius: 6px;
  padding: 2px 6px;
  font-size: 12px;
  font-weight: 600;
}

.op-replace {
  background: #ffe6b3;
  color: #8f5400;
}

.op-insert {
  background: #d9f5d2;
  color: #2b7a2d;
}

.op-delete {
  background: #ffdede;
  color: #b52f2f;
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
  .compare-main {
    padding: 12px;
  }

  .top-row {
    grid-template-columns: 1fr;
  }
}
</style>
