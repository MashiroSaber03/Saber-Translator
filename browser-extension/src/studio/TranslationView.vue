<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import SelectControl from './SelectControl.vue'
import ImportDialog from './ImportDialog.vue'
import { downloadOriginals } from './downloadOriginals'
import type { StudioState, StudioAction } from './protocol'
import type {
  BrowserLibraryBook,
  BrowserSessionImportCommand,
  BrowserSessionImportResult,
  DomainPreference,
} from '../types'
const props = defineProps<{
  state: StudioState
  actionTarget?: HTMLElement
  request: <T = void>(action: StudioAction, payload?: unknown) => Promise<T>
}>()
const selected = ref<string[]>([])
const importOpen = ref(false)
const importIds = ref<string[] | null>(null)
const error = ref('')
const actionPending = ref(false)
const downloadStage = ref('')
let actionSequence = 0
const busy = computed(() => props.state.notice.tone === 'busy' || actionPending.value)
const session = computed(() => props.state.session)
const processing = computed(() =>
  session.value?.pages.some(page => page.state === 'queued' || page.state === 'translating')
)
const methods = [
  { value: 'adapter', label: '站点适配 / 通用' },
  { value: 'similar', label: '点选同类图片' },
  { value: 'dom-agent', label: 'DOM Agent' },
]
watch(
  () => props.state.candidates.map(item => item.id).join(','),
  () => {
    selected.value = props.state.candidates.map(item => item.id)
  },
  { immediate: true }
)
async function act(action: StudioAction, payload?: unknown) {
  if (actionPending.value && action !== 'cancel') return
  const sequence = ++actionSequence
  actionPending.value = true
  error.value = ''
  try {
    await props.request(action, payload)
  } catch (e) {
    if (sequence === actionSequence) error.value = (e as Error).message
  } finally {
    if (sequence === actionSequence) actionPending.value = false
  }
}
function preference(patch: Partial<DomainPreference>) {
  void act('preference', patch)
}
const loadBooks = () => props.request<BrowserLibraryBook[]>('books')
async function submitImport(command: BrowserSessionImportCommand) {
  error.value = ''
  if (importIds.value === null) return props.request<BrowserSessionImportResult>('import', command)
  actionPending.value = true
  try {
    return await props.request<BrowserSessionImportResult>('import-selected', { ids: [...importIds.value], command })
  } finally {
    actionPending.value = false
  }
}
async function downloadSelected() {
  if (busy.value || !selected.value.length) return
  actionPending.value = true
  error.value = ''
  downloadStage.value = '正在准备原图…'
  let sessionId: string | undefined
  let downloaded = false
  try {
    sessionId = await props.request<string>('prepare-download', [...selected.value])
    downloadStage.value = '正在下载 ZIP…'
    await downloadOriginals(sessionId, props.state.title)
    downloaded = true
  } catch (e) {
    error.value = (e as Error).message
  } finally {
    try {
      if (sessionId) await props.request('finish-download', downloaded ? '已发起原图 ZIP 下载' : undefined)
    } catch (e) {
      error.value = (e as Error).message
    }
    downloadStage.value = ''
    actionPending.value = false
  }
}
</script>
<template>
  <div class="page-intro">
    <span class="eyebrow">当前页面</span>
    <h2>{{ state.title }}</h2>
  </div>
  <div
    class="notice"
    :class="{ error: state.notice.tone === 'error' || error, working: busy }"
    role="status"
  >
    <span class="status-symbol" aria-hidden="true">{{
      busy ? '◌' : state.notice.tone === 'error' ? '!' : '✦'
    }}</span>
    <div>
      <strong>{{ error || state.notice.title }}</strong>
      <p>{{ state.notice.message }}</p>
    </div>
  </div>
  <div v-if="state.uploadError" class="notice error" role="status">
    {{ state.uploadError.count }} 张图片上传失败：{{ state.uploadError.message }}
  </div>
  <div v-if="state.retryStart || state.uploadError" class="actions">
    <button v-if="state.uploadError" class="button" @click="act('retry-uploads')">重试上传</button
    ><button v-if="state.retryStart" class="button" @click="act('retry-start')">重试启动</button
    ><button class="text-button" @click="act('diagnostics')">复制诊断</button>
  </div>
  <template v-if="state.view === 'idle'">
    <div v-if="state.preference.rule" class="notice" role="note">
      <div>
        <p>此网站有已保存的图片识别规则</p>
        <button class="text-button" :disabled="busy" @click="act('discover-saved')">使用上次规则</button>
      </div>
    </div>
    <section class="setting-group">
      <h3>识别与翻译</h3>
      <label class="field"
        >识别方式<SelectControl
          :model-value="state.preference.method"
          :options="methods"
          label="识别方式"
          :disabled="busy"
          @update:model-value="
            value => preference({ method: value as DomainPreference['method'] })
          "
      /></label>
      <div class="field">
        <span>翻译模式</span>
        <div class="segmented" role="group" aria-label="翻译模式">
          <button
            :class="{ active: state.preference.mode === 'standard' }"
            :aria-pressed="state.preference.mode === 'standard'"
            :disabled="busy"
            @click="preference({ mode: 'standard' })"
          >
            标准翻译</button
          ><button
            :class="{ active: state.preference.mode === 'hq' }"
            :aria-pressed="state.preference.mode === 'hq'"
            :disabled="busy"
            @click="preference({ mode: 'hq' })"
          >
            高质量翻译
          </button>
        </div>
      </div>
      <label class="switch-field"
        >启用术语表<input
          type="checkbox"
          class="switch"
          :checked="state.preference.glossaryEnabled"
          :disabled="busy"
          @change="
            preference({
              glossaryEnabled: ($event.target as HTMLInputElement).checked,
            })
          " /></label
      ><label class="switch-field"
        >自动添加术语<input
          type="checkbox"
          class="switch"
          :checked="state.preference.autoTermsEnabled"
          :disabled="busy"
          @change="
            preference({
              autoTermsEnabled: ($event.target as HTMLInputElement).checked,
            })
          "
      /></label>
    </section>
    <Teleport :to="actionTarget ?? 'body'" :disabled="!actionTarget"><div class="flow-actions">
    <button
      class="button primary full"
      :disabled="busy"
      @click="act('discover', state.preference.method)"
    >
      <span aria-hidden="true">✦</span
      >{{ state.preference.method === 'similar' ? '选择一张漫画图片' : '识别漫画图片' }}
    </button>
    <p class="footnote">确认图片后才开始翻译，退出页面清理临时数据</p>
    </div></Teleport>
  </template>
  <template v-if="state.view === 'candidates'"
    ><div class="section-heading">
      <h3>
        已选图片
        <span class="badge">{{ selected.length }} / {{ state.candidates.length }}</span>
      </h3>
      <div>
        <button class="text-button" :disabled="busy" @click="selected = state.candidates.map(item => item.id)">
          全选</button
        ><button class="text-button" :disabled="busy" @click="selected = []">清空</button>
      </div>
    </div>
    <div class="candidate-grid">
      <label
        v-for="(item, index) in state.candidates"
        :key="item.id"
        class="candidate"
        :class="{ selected: selected.includes(item.id) }"
        ><img
          v-if="item.sourceUrl"
          :src="item.sourceUrl"
          alt=""
          loading="lazy"
          @error="($event.target as HTMLImageElement).style.opacity = '0'"
        /><span class="candidate-fallback">{{ String(index + 1).padStart(2, '0') }}</span
        ><input
          v-model="selected"
          :disabled="busy"
          type="checkbox"
          :value="item.id"
          :aria-label="`选择第 ${index + 1} 张图片`"
        /><span class="candidate-meta">{{ item.width }} × {{ item.height }}</span></label
      >
    </div>
    <Teleport :to="actionTarget ?? 'body'" :disabled="!actionTarget"><div class="flow-actions">
      <div class="selection-actions">
        <button class="button" :disabled="busy || !selected.length" @click="importIds = [...selected]; importOpen = true">仅导入书架</button>
        <button class="button" title="将勾选原图按网页顺序打包为 ZIP，需连接本机 Saber" :disabled="busy || !selected.length" @click="downloadSelected">{{ downloadStage || '下载原图（ZIP）' }}</button>
      <button class="button" :disabled="busy" @click="act('back')">重新选择</button
      ><button
        class="button primary"
        :disabled="!selected.length || busy"
        @click="act('confirm', selected)"
      >
        开始翻译 · {{ selected.length }} 张
      </button>
      </div>
      <p v-if="downloadStage" class="muted" role="status">{{ downloadStage }}</p>
      <p v-if="error" class="notice error" role="alert">{{ error }}</p>
    </div></Teleport></template
  >
  <template v-if="state.view === 'progress'">
    <div v-if="state.preparation" class="progress-card">
      <div class="section-heading">
        <strong>正在准备图片</strong
        ><span>{{ state.preparation.processed }} / {{ state.preparation.total }}</span>
      </div>
      <progress :max="state.preparation.total || 1" :value="state.preparation.processed" />
      <p class="muted">
        成功 {{ state.preparation.processed - state.preparation.failed }} · 失败
        {{ state.preparation.failed }}
      </p>
    </div>
    <template v-if="session"
      ><div class="metrics">
        <div>
          <strong>{{ session.counts.total }}</strong
          ><span>全部图片</span>
        </div>
        <div>
          <strong>{{ session.counts.completed }}</strong
          ><span>已完成</span>
        </div>
        <div>
          <strong>{{ session.counts.queued + session.counts.translating }}</strong
          ><span>处理中</span>
        </div>
        <div :class="{ error: session.counts.failed }">
          <strong>{{ session.counts.failed }}</strong
          ><span>失败</span>
        </div>
      </div>
      <progress :max="session.counts.total || 1" :value="session.counts.completed" />
      <Teleport :to="actionTarget ?? 'body'" :disabled="!actionTarget"><div class="flow-actions"><div class="actions">
        <button
          class="button grow"
          :disabled="!session.counts.completed"
          @click="act('toggle-global')"
        >
          {{ state.translated ? '显示原图' : '显示译图' }}</button
        ><button
          v-if="!state.imported"
          class="button primary grow"
          :disabled="processing || busy || !session.pages.some(page => page.pageId)"
          @click="importIds = null; importOpen = true"
        >
          导入到书架
        </button>
      </div>
      </div></Teleport>
      <div v-if="!state.imported" class="actions">
        <button class="text-button" :disabled="actionPending || session.state === 'cancelled'" @click="act(state.discoveryStopped ? 'resume-discovery' : 'stop-discovery')">{{ state.discoveryStopped ? '继续发现' : '停止继续发现' }}</button
        ><button
          class="text-button danger"
          :disabled="!processing && session.state !== 'idle'"
          @click="act('cancel')"
        >
          取消任务
        </button>
      </div>
      <p v-if="!state.imported" class="muted">{{ state.discoveryStopped ? '继续发现已停止' : '正在自动发现后续图片' }}</p>
      <button class="button" :disabled="processing || busy" @click="act('reselect')">重新选图</button>
      <p class="footnote">重新选图会清除当前页面的临时结果；需要保留请先导入书架。</p>
      <button class="button" :disabled="processing || busy" @click="act('restart')">按新配置重新翻译</button>
      <p class="footnote">重新翻译会替换当前页面的临时结果，使用已保存的新配置；已导入书架的内容不受影响。</p>
      <details v-if="!state.imported" class="disclosure">
        <summary>
          逐页查看<span class="badge">{{ session.pages.length }}</span
          ><span class="chevron" />
        </summary>
        <div v-for="page in session.pages" :key="page.id" class="page-row">
          <div>
            <strong>第 {{ page.ordinal }} 张</strong>
            <p class="muted">
              {{
                page.error?.message ||
                {
                  queued: '排队中',
                  translating: '翻译中',
                  completed: '已完成',
                  failed: '失败',
                  cancelled: '已取消',
                }[page.state]
              }}
            </p>
          </div>
          <div class="actions">
            <button
              v-if="page.state === 'completed' && page.resultReady"
              class="text-button"
              @click="act('toggle-page', page.id)"
            >
              {{ state.originalPageIds.includes(page.id) ? '显示译图' : '查看原图' }}</button
            ><button
              v-if="['completed', 'failed', 'cancelled'].includes(page.state)"
              class="text-button"
              :disabled="processing || actionPending"
              @click="act('retry-page', page.id)"
            >
              {{ page.state === 'completed' ? '按原配置重翻' : '按原配置重试' }}
            </button>
          </div>
        </div>
      </details>
      <details
        v-if="state.preference.glossaryEnabled || state.preference.autoTermsEnabled"
        class="disclosure"
      >
        <summary>
          实时术语<span class="badge">{{ state.terms.length }}</span
          ><span class="chevron" />
        </summary>
        <p v-if="!state.terms.length" class="muted">尚未提取术语</p>
        <div v-for="(term, i) in state.terms" :key="i" class="term-row">
          <span>{{ term.source }}</span
          ><span>→</span><strong>{{ term.target }}</strong>
        </div>
      </details>
    </template>
  </template>
  <details class="disclosure utility">
    <summary>页面操作<span class="chevron" /></summary>
    <div class="actions">
      <button v-if="state.preference.rule" class="text-button" @click="act('delete-adaptation')">
        删除已保存适配</button
      ><button class="text-button" @click="act('diagnostics')">复制诊断</button>
    </div>
  </details>
  <ImportDialog
    v-if="importOpen"
    :title="state.title"
    :load-books="loadBooks"
    :submit="submitImport"
    @close="importOpen = false"
  />
</template>
