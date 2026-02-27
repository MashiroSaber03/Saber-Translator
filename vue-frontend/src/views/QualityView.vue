<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import AppHeader from '@/components/common/AppHeader.vue'
import { analyzeQuality, getQualityReport } from '@/api/quality'
import type { QualityIssue, QualityReport } from '@/types'
import { showToast } from '@/utils/toast'

const router = useRouter()

const sessionPath = ref('')
const loading = ref(false)
const report = ref<QualityReport | null>(null)

const summary = computed(() => report.value?.summary)

function parseBookshelfSession(path: string): { bookId?: string; chapterId?: string } {
  const parts = path.replace(/\\/g, '/').split('/')
  if (parts.length >= 5 && parts[0] === 'bookshelf' && parts[2] === 'chapters') {
    return {
      bookId: parts[1],
      chapterId: parts[3]
    }
  }
  return {}
}

function jumpToIssue(issue: QualityIssue): void {
  if (!report.value) return
  const { bookId, chapterId } = parseBookshelfSession(report.value.session)
  if (!bookId || !chapterId) {
    showToast('当前会话不是书架章节路径，无法自动跳转', 'info')
    return
  }
  router.push({
    path: '/reader',
    query: {
      book: bookId,
      chapter: chapterId,
      page: String(issue.page_index),
      bubble: issue.bubble_index !== undefined && issue.bubble_index !== null ? String(issue.bubble_index) : undefined
    }
  })
}

async function runAnalyze(): Promise<void> {
  if (!sessionPath.value.trim()) {
    showToast('请先输入 session 路径', 'warning')
    return
  }
  loading.value = true
  try {
    const res = await analyzeQuality(sessionPath.value.trim())
    if (!res.success || !res.report) {
      throw new Error(res.error || '质量分析失败')
    }
    report.value = res.report
    showToast(`质量分析完成，发现 ${res.report.issue_count} 个问题`, 'success')
  } catch (error) {
    showToast((error as Error).message, 'error')
  } finally {
    loading.value = false
  }
}

async function loadReport(): Promise<void> {
  if (!sessionPath.value.trim()) {
    showToast('请先输入 session 路径', 'warning')
    return
  }
  loading.value = true
  try {
    const res = await getQualityReport(sessionPath.value.trim())
    if (!res.success || !res.report) {
      throw new Error(res.error || '未找到报告')
    }
    report.value = res.report
    showToast('已加载质量报告', 'success')
  } catch (error) {
    showToast((error as Error).message, 'error')
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="quality-page">
    <AppHeader variant="bookshelf" />

    <main class="quality-main">
      <section class="panel">
        <h1>质量工作台</h1>
        <p>输入会话路径，分析 OCR 置信度、长度异常、术语一致性和可疑断句。</p>

        <div class="controls">
          <label for="quality-session-path" class="sr-only">会话路径</label>
          <input
            id="quality-session-path"
            v-model="sessionPath"
            type="text"
            placeholder="例如: bookshelf/book123/chapters/ch_01/session"
          />
          <button type="button" :disabled="loading" @click="runAnalyze">开始分析</button>
          <button type="button" class="secondary" :disabled="loading" @click="loadReport">加载已有报告</button>
        </div>
      </section>

      <section v-if="report && summary" class="panel">
        <div class="summary-grid">
          <div class="card">
            <span class="label">总问题</span>
            <strong>{{ summary.total }}</strong>
          </div>
          <div class="card high">
            <span class="label">高优先级</span>
            <strong>{{ summary.by_severity.high || 0 }}</strong>
          </div>
          <div class="card medium">
            <span class="label">中优先级</span>
            <strong>{{ summary.by_severity.medium || 0 }}</strong>
          </div>
          <div class="card low">
            <span class="label">低优先级</span>
            <strong>{{ summary.by_severity.low || 0 }}</strong>
          </div>
        </div>
      </section>

      <section v-if="report" class="panel">
        <h2>问题列表</h2>
        <div v-if="report.issues.length === 0" class="empty">未发现明显问题。</div>
        <div v-else class="issues">
          <article v-for="issue in report.issues" :key="issue.id" class="issue-item">
            <div class="meta">
              <span class="tag" :class="`tag-${issue.severity}`">{{ issue.severity }}</span>
              <span>{{ issue.type }}</span>
              <span>页 {{ issue.page_index }}</span>
              <span v-if="issue.bubble_index !== undefined && issue.bubble_index !== null">
                气泡 {{ issue.bubble_index }}
              </span>
            </div>
            <p class="message">{{ issue.message }}</p>
            <div class="texts">
              <div>
                <strong>原文</strong>
                <pre>{{ issue.original_text || '-' }}</pre>
              </div>
              <div>
                <strong>译文</strong>
                <pre>{{ issue.translated_text || '-' }}</pre>
              </div>
            </div>
            <button type="button" class="jump-btn" @click="jumpToIssue(issue)">跳转到章节定位</button>
          </article>
        </div>
      </section>
    </main>
  </div>
</template>

<style scoped>
.quality-page {
  min-height: 100vh;
  background: linear-gradient(160deg, #f7fbff 0%, #eef7f1 100%);
}

.quality-main {
  max-width: 1100px;
  margin: 0 auto;
  padding: 20px;
  display: grid;
  gap: 16px;
}

.panel {
  background: #fff;
  border: 1px solid #dfe8f2;
  border-radius: 14px;
  padding: 16px;
}

.panel h1,
.panel h2 {
  margin: 0 0 8px;
}

.panel p {
  margin: 0 0 12px;
  color: #4f6075;
}

.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.controls input {
  flex: 1;
  min-width: 320px;
  border: 1px solid #c7d7ea;
  border-radius: 10px;
  padding: 10px 12px;
}

.controls button {
  border: none;
  border-radius: 10px;
  padding: 10px 14px;
  background: #1677ff;
  color: #fff;
  cursor: pointer;
}

.controls button.secondary {
  background: #4c627a;
}

.controls button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px;
}

.card {
  border-radius: 12px;
  padding: 14px;
  background: #f3f8ff;
}

.card.high {
  background: #ffe9e9;
}

.card.medium {
  background: #fff4df;
}

.card.low {
  background: #edf9eb;
}

.card .label {
  color: #516177;
  font-size: 13px;
}

.card strong {
  display: block;
  margin-top: 6px;
  font-size: 22px;
}

.empty {
  color: #4f6075;
  padding: 8px 0;
}

.issues {
  display: grid;
  gap: 12px;
}

.issue-item {
  border: 1px solid #dfe8f2;
  border-radius: 12px;
  padding: 12px;
}

.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  font-size: 13px;
  color: #5a6b80;
}

.tag {
  border-radius: 8px;
  padding: 2px 8px;
  font-weight: 600;
}

.tag-high {
  background: #ffdbdb;
  color: #c02424;
}

.tag-medium {
  background: #ffe9c7;
  color: #a95d00;
}

.tag-low {
  background: #dff4da;
  color: #2f7a2f;
}

.message {
  margin: 8px 0;
  font-weight: 600;
}

.texts {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

.texts pre {
  white-space: pre-wrap;
  margin: 6px 0 0;
  background: #f7f9fc;
  border-radius: 8px;
  padding: 8px;
  font-size: 12px;
}

.jump-btn {
  margin-top: 10px;
  border: 1px solid #9fb7d3;
  background: #fff;
  color: #2d4d73;
  border-radius: 8px;
  padding: 6px 10px;
  cursor: pointer;
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
  .quality-main {
    padding: 12px;
  }

  .controls input {
    min-width: 100%;
  }

  .texts {
    grid-template-columns: 1fr;
  }
}
</style>
