<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { isRequestCanceled } from '@/api/client'
import { useRuntimeStore } from '@/stores/runtimeStore'
import { cancelLocalModelInstall, getLocalModels, installLocalModels, prepareValenRuntime,
  type LocalModelCatalog, type LocalModelInstallRequest } from '@/api/v2/models'

const props = defineProps<{ active: boolean }>()
const runtime = useRuntimeStore()
const catalog = ref<LocalModelCatalog | null>(null)
const source = ref<NonNullable<LocalModelInstallRequest['source']>>('auto')
const error = ref('')
const submitting = ref(false)
const loading = ref(false)
const task = computed(() => catalog.value?.task)
const taskActive = computed(() => ['running', 'cancelling'].includes(task.value?.status ?? ''))
const busy = computed(() => submitting.value || taskActive.value)
const missing = computed(() => catalog.value?.models.filter(model => !model.optional && !model.installed) ?? [])
const missingBytes = computed(() => missing.value.reduce((sum, model) => sum + model.sizeBytes, 0))
const sources = computed(() => catalog.value?.sources.map(item => ({ value: item.id,
  label: item.configured ? item.label : `${item.label}（尚未配置）`, disabled: !item.configured })) ?? [])
const currentLabel = computed(() => catalog.value?.models.find(model => model.id === task.value?.currentModelId)?.label ?? '')
const progress = computed(() => Math.min(100, Math.round((task.value?.downloadedBytes ?? 0) / Math.max(1, task.value?.totalBytes ?? 0) * 100)))
let timer: ReturnType<typeof setTimeout> | undefined
let controller: AbortController | undefined
let disposed = false

function formatBytes(bytes: number): string {
  return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} GiB` : `${(bytes / 1024 ** 2).toFixed(1)} MiB`
}

function stopPolling(): void {
  clearTimeout(timer)
  controller?.abort()
  controller = undefined
}

async function refresh(): Promise<void> {
  stopPolling()
  if (!props.active || disposed) return
  const request = new AbortController()
  controller = request
  loading.value = true
  try {
    const result = await getLocalModels(request.signal)
    if (request.signal.aborted) return
    catalog.value = result
    const installed = result.models.filter(model => model.installed).map(model => model.id)
    if (JSON.stringify(installed) !== JSON.stringify(runtime.capabilities?.installedModels)) {
      void runtime.load(true).catch(() => {})
    }
    error.value = ''
  } catch (cause) {
    if (!isRequestCanceled(cause)) error.value = cause instanceof Error ? cause.message : '无法读取模型状态'
  } finally {
    if (controller === request) {
      controller = undefined
      loading.value = false
      if (props.active && taskActive.value) timer = setTimeout(() => void refresh(), 1000)
    }
  }
}

async function run(action: () => ReturnType<typeof installLocalModels>): Promise<void> {
  submitting.value = true
  error.value = ''
  try {
    const result = await action()
    if (catalog.value) catalog.value.task = result
    await refresh()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : '无法执行模型管理操作'
  } finally { submitting.value = false }
}

function install(modelIds: string[], verify = false): Promise<void> {
  return run(() => installLocalModels({ modelIds, source: source.value, verify }))
}

async function cancel(): Promise<void> {
  submitting.value = true
  try {
    const result = await cancelLocalModelInstall()
    if (catalog.value) catalog.value.task = result
    await refresh()
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : '取消失败'
  } finally { submitting.value = false }
}

watch([() => props.active, () => runtime.capabilities?.installedModels], ([active]) => {
  if (active) void refresh(); else stopPolling()
}, { immediate: true })
onBeforeUnmount(() => { disposed = true; stopPolling() })
</script>

<template>
  <div class="local-models">
    <ProductFormSection>
      <template #title>模型管理</template>
      <p class="local-models__hint">按需下载安装，或一键安装翻译所需模型。关闭设置后下载会继续。</p>
      <p v-if="catalog" class="local-models__path">模型下载目录：{{ catalog.directory }}</p>
      <UiField variant="settings" label="模型下载源" hint="模型下载后自动校验并解压。离线安装可将完整 ZIP 解压到上述目录后刷新。">
        <UiSelect v-model="source" :options="sources" :disabled="busy" aria-label="模型下载源" />
      </UiField>
      <div class="local-models__actions">
        <UiButton variant="primary" :disabled="busy || !missing.length" @click="install(missing.map(model => model.id))">{{ missing.length ? `一键安装翻译模型（${formatBytes(missingBytes)}）` : '翻译模型已全部安装' }}</UiButton>
        <UiButton :disabled="loading || submitting" @click="refresh">刷新状态</UiButton>
      </div>
    </ProductFormSection>
    <ProductStatusBanner v-if="error" tone="danger" role="alert">{{ error }}</ProductStatusBanner>
    <ProductStatusBanner v-if="busy && task" tone="info" role="status" aria-live="polite">
      <strong>{{ task.status === 'cancelling' ? '正在取消' : task.phase === 'preparing_runtime' ? '正在准备 Valen 推理环境' : task.phase === 'extracting' ? '正在解压安装' : task.phase === 'verifying' ? '正在校验' : '正在下载' }} {{ currentLabel }}</strong>
      <template v-if="task.phase !== 'preparing_runtime'">
        <div class="local-models__progress"><progress :value="progress" max="100" aria-label="模型安装进度" /><span>{{ progress }}%</span></div>
        <p class="local-models__hint">{{ formatBytes(task.downloadedBytes ?? 0) }} / {{ formatBytes(task.totalBytes ?? 0) }} · {{ task.currentFile }}</p>
      </template>
      <template #actions><UiButton size="sm" :disabled="task.status === 'cancelling' || submitting" @click="cancel">取消操作</UiButton></template>
    </ProductStatusBanner>
    <ProductStatusBanner v-else-if="task?.status === 'completed'" tone="success" role="status">{{ task.phase === 'preparing_runtime' ? 'Valen 推理环境已就绪' : task.verify ? '校验完成' : '安装完成' }}</ProductStatusBanner>
    <ProductStatusBanner v-else-if="task?.status === 'cancelled'" tone="neutral" role="status">已取消。已安装的模型不受影响，再次下载会继续未完成的文件。</ProductStatusBanner>
    <ProductStatusBanner v-else-if="task?.status === 'failed'" tone="warning" role="status">部分模型未完成，请查看下方原因并重试。{{ task.error }}</ProductStatusBanner>
    <ProductFormSection v-if="catalog">
      <template #title>本地模型 · {{ catalog.models.filter(model => model.installed).length }} / {{ catalog.models.length }} 已安装</template>
      <p class="local-models__hint">其他检测器还需 Default 生成精确掩膜；自动提取文字颜色需要 48px OCR。只编辑已有文字、手动涂白和导出无需下载模型。</p>
      <ul class="local-models__list" aria-label="本地模型列表">
        <li v-for="model in catalog.models" :key="model.id" class="local-models__row" :data-model-id="model.id">
          <div class="local-models__info">
            <div class="local-models__name"><strong>{{ model.label }}</strong><span>{{ formatBytes(model.sizeBytes) }}</span></div>
            <p class="local-models__hint">{{ model.description }}</p>
            <p v-if="task?.results[model.id]?.error" class="local-models__error">{{ task.results[model.id]?.error }}</p>
            <p v-else-if="!model.installed && model.partialBytes" class="local-models__hint">已有 {{ formatBytes(model.partialBytes) }} 下载缓存，可继续下载</p>
          </div>
          <div class="local-models__row-actions">
            <span class="local-models__state">{{ model.installed ? '已安装' : '未安装' }}</span>
            <UiButton v-if="model.installed" size="sm" :disabled="busy" :aria-label="`校验 ${model.label}`" @click="install([model.id], true)">校验</UiButton>
            <UiButton size="sm" :variant="model.installed ? 'secondary' : 'primary'" :disabled="busy" :aria-label="`下载或修复 ${model.label}`" @click="install([model.id])">{{ model.installed ? '修复' : model.partialBytes ? '继续下载' : '下载' }}</UiButton>
          </div>
        </li>
      </ul>
      <UiField
        v-if="catalog.models.some(model => model.optional)"
        variant="settings" label="Valen 独立推理环境"
        hint="准备时需要 Python 3.12 和下载依赖。已有对应版本的完整 Valen 模型可放入上述目录的同名文件夹，然后刷新并校验。"
      >
        <UiButton :disabled="busy || catalog.valenRuntimeReady" @click="run(prepareValenRuntime)">
          {{ catalog.valenRuntimeReady ? '推理环境已就绪' : '准备 Valen 推理环境' }}
        </UiButton>
      </UiField>
    </ProductFormSection>
  </div>
</template>

<style scoped>
.local-models {
  --product-form-section-margin-bottom: 0;

  display: grid;
  gap: 16px;
}

.local-models__hint,
.local-models__path {
  margin: 6px 0;
  color: var(--color-text-supporting);
  font-size: 0.85rem;
  line-height: 1.6;
  overflow-wrap: anywhere;
}

.local-models__path { margin-bottom: 16px; }

.local-models__actions,
.local-models__row-actions,
.local-models__name {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}

.local-models__name span,
.local-models__state {
  color: var(--color-text-supporting);
  font-size: 0.8rem;
}

.local-models__list {
  list-style: none;
  padding: 0;
  margin: 16px 0 0;
}

.local-models__row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 16px 0;
  border-top: 1px solid var(--color-border-muted);
}

.local-models__info { flex: 1; min-width: 0; }

.local-models__error {
  margin: 6px 0 0;
  color: var(--color-status-error);
  overflow-wrap: anywhere;
  font-size: 0.85rem;
}

.local-models__progress { display: flex; align-items: center; gap: 10px; margin-top: 8px; }

.local-models__progress progress { flex: 1; min-width: 0; accent-color: var(--color-action-primary); }

@media (--breakpoint-sm-down) {
  .local-models__row { align-items: flex-start; flex-direction: column; }
}
</style>
