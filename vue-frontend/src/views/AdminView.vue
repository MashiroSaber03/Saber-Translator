<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import {
  createAdminInvite,
  createUserRecoveryCode,
  getAssetQuota,
  getPublicUserPolicy,
  getRegistrationPolicy,
  getSchedulingPolicy,
  listAdminInvites,
  listAdminUsers,
  revokeAdminInvite,
  setAdminUserStatus,
  setAssetQuota,
  setPublicUserPolicy,
  setRegistrationPolicy,
  setSchedulingPolicy,
  type AdminInvite,
  type AdminUser,
  type PublicFeatureKey,
  type PublicModelKey,
  type PublicUserPolicy,
  type QueueDiscipline,
  type SchedulingOverview,
  type SchedulingPolicy,
} from '@/api/v2/auth'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import UiSwitch from '@/components/ui/UiSwitch.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'
import type { UiSelectOption, UiSelectValue } from '@/components/ui/selectTypes'
import { useRuntimeStore } from '@/stores/runtimeStore'
import { deepClone } from '@/utils/deepClone'
import { jobKindLabel } from '@/utils/taskDisplay'

const GIB = 1024 ** 3
type UserStatusFilter = 'all' | AdminUser['taskStatus'] | 'disabled'

interface AdminPageData {
  users: AdminUser[]
  invites: AdminInvite[]
  assetQuotaGiB: number | null
  registrationRequiresInvite: boolean
  publicUserPolicy: PublicUserPolicy
  scheduling: SchedulingOverview
}

const runtime = useRuntimeStore()
const adminData = ref<AdminPageData | null>(null)
const latestInvite = ref('')
const oneTimeMessage = ref('')
const error = ref('')
const busy = ref(false)
const loading = ref(false)
const userSearch = ref('')
const userStatusFilter = ref<UserStatusFilter>('all')

const users = computed(() => adminData.value?.users ?? [])
const invites = computed(() => adminData.value?.invites ?? [])
const assetQuotaGiB = computed<number | null>({
  get: () => adminData.value?.assetQuotaGiB ?? null,
  set: value => {
    if (adminData.value) adminData.value.assetQuotaGiB = value
  },
})
const registrationRequiresInvite = computed({
  get: () => adminData.value?.registrationRequiresInvite ?? false,
  set: (value: boolean) => {
    if (adminData.value) adminData.value.registrationRequiresInvite = value
  },
})
const publicUserPolicy = computed(() => adminData.value?.publicUserPolicy ?? null)
const scheduling = computed(() => adminData.value?.scheduling ?? null)
const controlsDisabled = computed(() => busy.value || loading.value)

const userStatusOptions: Array<UiSelectOption & { value: UserStatusFilter }> = [
  { label: '全部状态', value: 'all' },
  { label: '处理中', value: 'active' },
  { label: '排队中', value: 'queued' },
  { label: '已暂停', value: 'paused' },
  { label: '待恢复', value: 'interrupted' },
  { label: '空闲', value: 'idle' },
  { label: '已禁用', value: 'disabled' },
]

const queueDisciplineOptions: Array<UiSelectOption & { value: QueueDiscipline }> = [
  { label: '按用户轮转', value: 'owner_round_robin' },
  { label: '先进先出', value: 'fifo' },
]

const filteredUsers = computed(() => {
  const query = userSearch.value.trim().toLocaleLowerCase()
  return users.value
    .filter(user => {
      if (query && !user.username.toLocaleLowerCase().includes(query)) return false
      if (userStatusFilter.value === 'all') return true
      if (userStatusFilter.value === 'disabled') return user.status === 'disabled'
      return user.status === 'active' && user.taskStatus === userStatusFilter.value
    })
    .sort((left, right) => userDisplayPriority(left) - userDisplayPriority(right))
})

const featureControls: Array<{
  key: PublicFeatureKey
  label: string
  description: string
}> = [
  { key: 'translation', label: '翻译', description: '进入翻译页并创建翻译、消字任务' },
  { key: 'insight', label: '漫画分析', description: '进入漫画分析页并运行分析任务' },
  { key: 'characterStudio', label: '角色工坊', description: '进入角色工坊并使用相关功能' },
  { key: 'editMode', label: '编辑模式', description: '进入气泡编辑并提交编辑操作' },
]

const modelControls: Array<{
  key: PublicModelKey
  label: string
}> = [
  { key: 'detector_default', label: 'Default（DBNet）检测' },
  { key: 'detector_ctd', label: 'CTD 检测' },
  { key: 'detector_yolo', label: 'YOLO 检测' },
  { key: 'aux_ysg_yolo', label: 'YSG YOLO 辅助检测' },
  { key: 'saber_yolo', label: 'Saber YOLO 精修' },
  { key: 'manga_ocr', label: 'Manga OCR' },
  { key: 'ocr_48px', label: '48px OCR' },
  { key: 'paddle_ocr', label: 'Paddle OCR' },
  { key: 'paddleocr_vl', label: 'PaddleOCR-VL' },
  { key: 'lama_mpe', label: 'LAMA（速度优化）' },
  { key: 'litelama', label: 'LAMA（通用）' },
]

function formatBytes(value: number): string {
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`
  if (value < GIB) return `${(value / 1024 ** 2).toFixed(1)} MB`
  return `${(value / GIB).toFixed(2)} GB`
}

function inviteStatusLabel(status: string): string {
  return (
    {
      active: '可使用',
      used: '已使用',
      revoked: '已撤销',
      expired: '已过期',
    }[status] ?? status
  )
}

async function reload(): Promise<void> {
  if (loading.value) return
  loading.value = true
  error.value = ''
  try {
    const [userRows, inviteRows, quota, registrationPolicy, userPolicy, scheduler] =
      await Promise.all([
        listAdminUsers(),
        listAdminInvites(),
        getAssetQuota(),
        getRegistrationPolicy(),
        getPublicUserPolicy(),
        getSchedulingPolicy(),
      ])
    adminData.value = {
      users: userRows,
      invites: inviteRows,
      assetQuotaGiB: quota.assetQuotaBytes / GIB,
      registrationRequiresInvite: registrationPolicy.registrationRequiresInvite,
      publicUserPolicy: deepClone(userPolicy),
      scheduling: deepClone(scheduler),
    }
    runtime.setRegistrationRequiresInvite(registrationPolicy.registrationRequiresInvite)
    runtime.setPublicUserPolicy(deepClone(userPolicy))
    runtime.setMaxDeepLearningConcurrency(scheduler.policy.maxDeepLearningConcurrency)
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : '管理数据加载失败'
  } finally {
    loading.value = false
  }
}

async function run(action: () => Promise<void>): Promise<void> {
  busy.value = true
  error.value = ''
  try {
    await action()
    await reload()
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : '操作失败'
  } finally {
    busy.value = false
  }
}

function quotaBytes(value: number | null): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed) || parsed <= 0) throw new Error('资产额度必须大于 0GB')
  return Math.round(parsed * GIB)
}

async function saveAssetQuota(): Promise<void> {
  await run(async () => {
    await setAssetQuota(quotaBytes(assetQuotaGiB.value))
    oneTimeMessage.value = '每用户资产额度已保存。'
  })
}

function formatDateTime(value: string | null): string {
  if (!value) return '从未创建任务'
  return new Date(value).toLocaleString()
}

function taskStatusLabel(status: AdminUser['taskStatus']): string {
  return {
    active: '处理中',
    queued: '排队中',
    paused: '已暂停',
    interrupted: '待恢复',
    idle: '空闲',
  }[status]
}

function taskDetail(user: AdminUser): string {
  const details: string[] = []
  if (user.currentTaskKind) details.push(jobKindLabel(user.currentTaskKind))
  if (user.currentTaskStartedAt) details.push(`${formatDateTime(user.currentTaskStartedAt)} 开始`)
  if (user.queuedTaskCount > 0) details.push(`${user.queuedTaskCount} 个排队`)
  if (user.pausedTaskCount > 0) details.push(`${user.pausedTaskCount} 个暂停`)
  if (user.interruptedTaskCount > 0) details.push(`${user.interruptedTaskCount} 个待恢复`)
  return details.join(' · ') || '暂无进行中的任务'
}

function userDisplayPriority(user: AdminUser): number {
  if (user.status === 'disabled') return 5
  return {
    active: 0,
    queued: 1,
    paused: 2,
    interrupted: 3,
    idle: 4,
  }[user.taskStatus]
}

function updateUserStatusFilter(value: UiSelectValue): void {
  const option = userStatusOptions.find(candidate => candidate.value === value)
  if (option) userStatusFilter.value = option.value
}

async function saveRegistrationPolicy(): Promise<void> {
  await run(async () => {
    const result = await setRegistrationPolicy(registrationRequiresInvite.value)
    runtime.setRegistrationRequiresInvite(result.registrationRequiresInvite)
    oneTimeMessage.value = result.registrationRequiresInvite
      ? '已开启邀请码注册，新用户注册时必须填写有效邀请码。'
      : '已关闭邀请码注册，新用户现在可以直接注册。'
  })
}

function updateFeature(key: PublicFeatureKey, value: boolean): void {
  if (publicUserPolicy.value) publicUserPolicy.value.features[key] = value
}

function updateModel(key: PublicModelKey, value: boolean): void {
  if (publicUserPolicy.value) publicUserPolicy.value.models[key] = value
}

async function savePublicUserPolicy(): Promise<void> {
  if (!publicUserPolicy.value) return
  await run(async () => {
    const result = await setPublicUserPolicy(publicUserPolicy.value as PublicUserPolicy)
    runtime.setPublicUserPolicy(result)
    oneTimeMessage.value = '普通用户的功能与性能设置已保存。'
  })
}

type NumericSchedulingKey = Exclude<keyof SchedulingPolicy, 'queueDiscipline'>

function updateQueueDiscipline(value: UiSelectValue): void {
  if (!scheduling.value) return
  const option = queueDisciplineOptions.find(candidate => candidate.value === value)
  if (option) scheduling.value.policy.queueDiscipline = option.value
}

function updateSchedulingNumber(key: NumericSchedulingKey, value: number | null): void {
  if (!scheduling.value || value === null || !Number.isInteger(value)) return
  scheduling.value.policy[key] = value
}

function schedulerWaitingLabel(reason: SchedulingOverview['status']['waitingReason']): string {
  if (reason === null) return '正常调度'
  return {
    worker_offline: 'Worker 离线',
    low_memory: '可用内存不足',
    queue_blocked: '任务等待写入锁',
  }[reason]
}

async function saveScheduler(): Promise<void> {
  if (!scheduling.value) return
  await run(async () => {
    const result = await setSchedulingPolicy(scheduling.value!.policy)
    runtime.setMaxDeepLearningConcurrency(result.policy.maxDeepLearningConcurrency)
    oneTimeMessage.value = '调度设置已保存，最迟约 2 秒对新调度片段生效。'
  })
}

async function refreshSchedulerStatus(): Promise<void> {
  if (!adminData.value || busy.value || loading.value) return
  try {
    const latest = await getSchedulingPolicy()
    if (adminData.value) adminData.value.scheduling.status = latest.status
  } catch {
    // Keep the last good snapshot; explicit refresh still reports errors.
  }
}

async function createInvite(): Promise<void> {
  await run(async () => {
    const result = await createAdminInvite()
    latestInvite.value = result.code
    oneTimeMessage.value = '邀请码只在创建时显示，请立即复制。'
  })
}

async function recoveryCode(user: AdminUser): Promise<void> {
  await run(async () => {
    const result = await createUserRecoveryCode(user.id)
    oneTimeMessage.value = `${user.username} 的一次性恢复码：${result.recoveryCode}`
  })
}

async function toggleUserStatus(user: AdminUser): Promise<void> {
  await run(async () => {
    await setAdminUserStatus(user.id, user.status === 'active' ? 'disabled' : 'active')
  })
}

let schedulerRefreshTimer: number | undefined
onMounted(() => {
  void reload()
  schedulerRefreshTimer = window.setInterval(refreshSchedulerStatus, 5000)
})
onBeforeUnmount(() => {
  if (schedulerRefreshTimer !== undefined) window.clearInterval(schedulerRefreshTimer)
})
</script>

<template>
  <main class="admin-page">
    <div class="admin-shell">
      <header class="admin-header">
        <div>
          <p class="page-brand">Saber Translator</p>
          <h1>管理后台</h1>
          <p class="page-description">管理账户、资产额度与访问凭据。</p>
        </div>
        <RouterLink class="back-link" to="/">返回应用</RouterLink>
      </header>

      <ProductStatusBanner v-if="error" tone="danger" role="alert">{{ error }}</ProductStatusBanner>
      <ProductStatusBanner v-if="oneTimeMessage" tone="neutral" role="status">
        {{ oneTimeMessage }}
      </ProductStatusBanner>

      <section v-if="!adminData" class="admin-card admin-card--loading">
        <div class="admin-loading-state" :role="error ? 'alert' : 'status'" aria-live="polite">
          <UiSpinner v-if="loading" :decorative="false" label="正在读取管理数据" size="24" />
          <h2>{{ loading ? '正在读取管理数据' : '管理数据尚未就绪' }}</h2>
          <p>
            {{ loading ? '正在同步账户与公网配置。' : '请检查服务状态后重新加载。' }}
          </p>
          <UiButton v-if="!loading" size="sm" @click="reload">重新加载</UiButton>
        </div>
      </section>

      <template v-else>
        <section class="admin-card admin-card--quota">
          <div>
            <p class="section-kicker">全局设置</p>
            <h2>每用户资产额度</h2>
            <p>仅限制持久化资产总量；不限制用户数量和书本数量。</p>
          </div>
          <div class="inline-control">
            <div class="quota-field">
              <UiNumberField
                v-model="assetQuotaGiB"
                class="admin-quota-input"
                :min="0.1"
                :step="0.1"
                size="sm"
                aria-label="每用户资产额度 GB"
                :disabled="controlsDisabled"
              />
              <span>GB / 用户</span>
            </div>
            <UiButton
              variant="primary"
              size="sm"
              :disabled="controlsDisabled"
              @click="saveAssetQuota"
            >
              保存额度
            </UiButton>
          </div>
        </section>

        <section class="admin-card admin-card--quota">
          <div>
            <p class="section-kicker">注册访问</p>
            <h2>邀请码注册</h2>
            <p v-if="registrationRequiresInvite">新用户必须填写管理员创建的有效邀请码。</p>
            <p v-else>当前允许自由注册，新用户无需邀请码。</p>
          </div>
          <div class="inline-control registration-control">
            <span>{{ registrationRequiresInvite ? '已开启' : '已关闭' }}</span>
            <UiSwitch
              v-model="registrationRequiresInvite"
              accessibility-label="注册必须使用邀请码"
              :disabled="controlsDisabled"
            />
            <UiButton
              variant="primary"
              size="sm"
              :disabled="controlsDisabled"
              @click="saveRegistrationPolicy"
            >
              保存注册设置
            </UiButton>
          </div>
        </section>

        <section v-if="scheduling" class="admin-card admin-card--block scheduler-card">
          <div class="section-heading">
            <div>
              <p class="section-kicker">任务调度</p>
              <h2>负载与公平性</h2>
              <p>全局只有一个持久任务占用计算槽；参数在页面片段边界生效。</p>
            </div>
            <UiButton
              variant="primary"
              size="sm"
              :disabled="controlsDisabled"
              @click="saveScheduler"
            >
              保存调度设置
            </UiButton>
          </div>

          <div class="scheduler-status-grid" aria-label="调度器状态">
            <div>
              <span>Worker</span>
              <strong>{{ scheduling.status.workerOnline ? '在线' : '离线' }}</strong>
            </div>
            <div>
              <span>当前状态</span>
              <strong>{{ schedulerWaitingLabel(scheduling.status.waitingReason) }}</strong>
            </div>
            <div>
              <span>队列</span>
              <strong>
                {{ scheduling.status.queuedJobCount }} 个任务 ·
                {{ scheduling.status.queuedUserCount }} 位用户
              </strong>
            </div>
            <div>
              <span>可用内存</span>
              <strong>
                {{ scheduling.status.availableMemoryMiB }} /
                {{ scheduling.status.totalMemoryMiB }} MiB
              </strong>
            </div>
          </div>
          <p v-if="scheduling.status.currentTask" class="scheduler-current-task">
            正在处理 {{ scheduling.status.currentTask.ownerUsername }} 的
            {{ jobKindLabel(scheduling.status.currentTask.kind) }}任务；另有
            {{ scheduling.status.pausedJobCount }} 个暂停任务。
          </p>
          <p v-else class="scheduler-current-task">
            当前没有执行中的持久任务；另有 {{ scheduling.status.pausedJobCount }} 个暂停任务。
          </p>

          <div class="scheduler-grid">
            <div class="scheduler-field">
              <div>
                <strong>队列规则</strong>
                <span>多人公网优先按用户轮转；先进先出会让一个任务完整跑完。</span>
              </div>
              <UiSelect
                class="scheduler-select"
                :model-value="scheduling.policy.queueDiscipline"
                :options="queueDisciplineOptions"
                size="sm"
                aria-label="队列规则"
                :disabled="controlsDisabled"
                @update:model-value="updateQueueDiscipline"
              />
            </div>
            <div class="scheduler-field">
              <div>
                <strong>每轮页数</strong>
                <span>完成这些页面并排空并行步骤后，再切换到下一位用户。</span>
              </div>
              <UiNumberField
                class="policy-number-input"
                :model-value="scheduling.policy.pageQuantum"
                :min="1"
                :max="20"
                :step="1"
                size="sm"
                aria-label="每轮页数"
                :disabled="controlsDisabled"
                @update:model-value="updateSchedulingNumber('pageQuantum', $event)"
              />
            </div>
            <div class="scheduler-field">
              <div>
                <strong>交互操作插队数</strong>
                <span>每个页面片段后最多处理的编辑操作；0 表示长任务运行时不插队。</span>
              </div>
              <UiNumberField
                class="policy-number-input"
                :model-value="scheduling.policy.interactiveBurst"
                :min="0"
                :max="3"
                :step="1"
                size="sm"
                aria-label="交互操作插队数"
                :disabled="controlsDisabled"
                @update:model-value="updateSchedulingNumber('interactiveBurst', $event)"
              />
            </div>
            <div class="scheduler-field">
              <div>
                <strong>深度学习并发</strong>
                <span>检测、OCR、颜色与修复共用的全局上限，对管理员同样生效。</span>
              </div>
              <UiNumberField
                class="policy-number-input"
                :model-value="scheduling.policy.maxDeepLearningConcurrency"
                :min="1"
                :max="8"
                :step="1"
                size="sm"
                aria-label="深度学习并发"
                :disabled="controlsDisabled"
                @update:model-value="updateSchedulingNumber('maxDeepLearningConcurrency', $event)"
              />
            </div>
            <div class="scheduler-field">
              <div>
                <strong>轻量操作并发</strong>
                <span>翻译气泡、角色工坊等 API 后台操作的并发上限。</span>
              </div>
              <UiNumberField
                class="policy-number-input"
                :model-value="scheduling.policy.apiOperationConcurrency"
                :min="1"
                :max="8"
                :step="1"
                size="sm"
                aria-label="轻量操作并发"
                :disabled="controlsDisabled"
                @update:model-value="updateSchedulingNumber('apiOperationConcurrency', $event)"
              />
            </div>
            <div class="scheduler-field">
              <div>
                <strong>模型空闲释放</strong>
                <span>Worker 空闲多久后卸载模型缓存，单位为秒。</span>
              </div>
              <UiNumberField
                class="policy-number-input"
                :model-value="scheduling.policy.modelIdleSeconds"
                :min="60"
                :max="3600"
                :step="30"
                size="sm"
                aria-label="模型空闲释放秒数"
                :disabled="controlsDisabled"
                @update:model-value="updateSchedulingNumber('modelIdleSeconds', $event)"
              />
            </div>
            <div class="scheduler-field">
              <div>
                <strong>最低可用内存</strong>
                <span>低于此值时释放模型并暂停领取新片段；0 表示关闭保护。</span>
              </div>
              <UiNumberField
                class="policy-number-input"
                :model-value="scheduling.policy.minAvailableMemoryMiB"
                :min="0"
                :step="512"
                size="sm"
                aria-label="最低可用内存 MiB"
                :disabled="controlsDisabled"
                @update:model-value="updateSchedulingNumber('minAvailableMemoryMiB', $event)"
              />
            </div>
          </div>
          <p class="scheduler-recommendation">
            当前机器建议：按用户轮转、每轮 1 页、插队 1 次、深度学习 1、轻量操作 2、 180
            秒释放模型、保留 2048 MiB 可用内存。
          </p>
          <p
            v-if="
              publicUserPolicy?.settings.parallel.allowed && scheduling.policy.pageQuantum === 1
            "
            class="scheduler-warning"
          >
            普通用户已可使用并行模式。每轮 1 页最公平，但跨页流水线收益较小；需要吞吐时可改为 2 页。
          </p>
        </section>

        <section v-if="publicUserPolicy" class="admin-card admin-card--block policy-card">
          <div class="section-heading">
            <div>
              <p class="section-kicker">公网普通用户</p>
              <h2>功能与性能控制</h2>
              <p>只限制普通用户；管理员和本地模式不受影响。保存后对新操作立即生效。</p>
            </div>
            <UiButton
              variant="primary"
              size="sm"
              :disabled="controlsDisabled"
              @click="savePublicUserPolicy"
            >
              保存功能设置
            </UiButton>
          </div>

          <div class="policy-section">
            <div class="policy-section__heading">
              <h3>页面与模式</h3>
              <p>关闭后隐藏入口，后端同时拒绝直接调用。</p>
            </div>
            <div class="policy-grid policy-grid--features">
              <div v-for="control in featureControls" :key="control.key" class="policy-row">
                <div>
                  <strong>{{ control.label }}</strong>
                  <span>{{ control.description }}</span>
                </div>
                <UiSwitch
                  :model-value="publicUserPolicy.features[control.key]"
                  :accessibility-label="`允许普通用户使用${control.label}`"
                  :disabled="controlsDisabled"
                  @change="updateFeature(control.key, $event)"
                />
              </div>
            </div>
          </div>

          <div class="policy-section">
            <div class="policy-section__heading">
              <h3>本地模型</h3>
              <p>关闭后普通用户不能在翻译任务或编辑操作中调用该模型。</p>
            </div>
            <div class="policy-grid policy-grid--models">
              <div
                v-for="control in modelControls"
                :key="control.key"
                class="policy-row policy-row--compact"
              >
                <strong>{{ control.label }}</strong>
                <UiSwitch
                  :model-value="publicUserPolicy.models[control.key]"
                  :accessibility-label="`允许普通用户调用${control.label}`"
                  :disabled="controlsDisabled"
                  @change="updateModel(control.key, $event)"
                />
              </div>
            </div>
          </div>

          <div class="policy-section">
            <div class="policy-section__heading">
              <h3>可修改设置与并行</h3>
              <p>这里只控制普通用户能否开启并行；并发上限由全局调度设置统一管理。</p>
            </div>
            <div class="policy-grid policy-grid--settings">
              <div class="policy-row">
                <div>
                  <strong>允许修改“禁用自动缩放”</strong>
                  <span>关闭后由管理员统一指定该设置。</span>
                </div>
                <UiSwitch
                  v-model="publicUserPolicy.settings.lamaDisableResize.editable"
                  accessibility-label="允许普通用户修改 LAMA 禁用自动缩放"
                  :disabled="controlsDisabled"
                />
              </div>
              <div v-if="!publicUserPolicy.settings.lamaDisableResize.editable" class="policy-row">
                <div>
                  <strong>禁用自动缩放的固定值</strong>
                  <span>{{
                    publicUserPolicy.settings.lamaDisableResize.value ? '强制开启' : '强制关闭'
                  }}</span>
                </div>
                <UiSwitch
                  v-model="publicUserPolicy.settings.lamaDisableResize.value"
                  accessibility-label="普通用户 LAMA 禁用自动缩放固定值"
                  :disabled="controlsDisabled"
                />
              </div>
              <div class="policy-row">
                <div>
                  <strong>允许并行模式</strong>
                  <span>关闭后普通用户的任务固定按顺序执行。</span>
                </div>
                <UiSwitch
                  v-model="publicUserPolicy.settings.parallel.allowed"
                  accessibility-label="允许普通用户使用并行模式"
                  :disabled="controlsDisabled"
                />
              </div>
            </div>
          </div>
        </section>

        <section class="admin-card admin-card--block">
          <div class="section-heading">
            <div>
              <p class="section-kicker">账户管理</p>
              <h2>用户</h2>
              <p>查看账户、当前任务和近期任务记录；统计范围与任务中心保留的历史一致。</p>
            </div>
            <UiButton size="sm" :disabled="controlsDisabled" @click="reload">刷新数据</UiButton>
          </div>
          <div class="user-toolbar">
            <UiInput
              v-model="userSearch"
              class="user-search"
              type="search"
              size="sm"
              placeholder="搜索用户名"
              aria-label="搜索用户名"
            />
            <div class="user-status-filter">
              <UiSelect
                :model-value="userStatusFilter"
                :options="userStatusOptions"
                size="sm"
                aria-label="按用户状态筛选"
                @update:model-value="updateUserStatusFilter"
              />
            </div>
            <span class="user-result-count" aria-live="polite">
              显示 {{ filteredUsers.length }} / {{ users.length }} 位用户
            </span>
          </div>

          <div v-if="filteredUsers.length" class="user-grid">
            <article v-for="user in filteredUsers" :key="user.id" class="user-card">
              <header class="user-card__header">
                <div class="user-card__identity">
                  <strong class="user-card__name">{{ user.username }}</strong>
                  <span>
                    {{ user.role === 'admin' ? '管理员' : '普通用户' }} · 注册于
                    {{ formatDateTime(user.createdAt) }}
                  </span>
                </div>
                <span class="status-pill" :class="`status-pill--${user.status}`">
                  <span class="status-dot" aria-hidden="true" />
                  {{ user.status === 'active' ? '正常' : '已禁用' }}
                </span>
              </header>

              <div class="user-card__task">
                <span class="status-pill" :class="`status-pill--task-${user.taskStatus}`">
                  <span class="status-dot" aria-hidden="true" />
                  {{ taskStatusLabel(user.taskStatus) }}
                </span>
                <span class="task-detail" :title="taskDetail(user)">{{ taskDetail(user) }}</span>
              </div>

              <div class="user-card__stats" aria-label="任务数量">
                <div class="user-stat">
                  <strong>{{ user.activeTaskCount }}</strong>
                  <span>处理中</span>
                </div>
                <div class="user-stat">
                  <strong>{{ user.queuedTaskCount }}</strong>
                  <span>排队</span>
                </div>
                <div class="user-stat">
                  <strong>{{ user.pausedTaskCount }}</strong>
                  <span>暂停</span>
                </div>
                <div class="user-stat">
                  <strong>{{ user.interruptedTaskCount }}</strong>
                  <span>待恢复</span>
                </div>
                <div class="user-stat">
                  <strong>{{ user.completedTaskCount }}</strong>
                  <span>近期完成</span>
                </div>
                <div class="user-stat" :class="{ 'user-stat--issue': user.issueTaskCount > 0 }">
                  <strong>{{ user.issueTaskCount }}</strong>
                  <span>近期异常</span>
                </div>
              </div>

              <dl class="user-card__meta">
                <div>
                  <dt>资产使用</dt>
                  <dd>
                    <span class="usage-value">{{ formatBytes(user.assetUsageBytes) }}</span>
                    <span class="usage-limit"> / {{ formatBytes(user.assetQuotaBytes) }}</span>
                  </dd>
                </div>
                <div>
                  <dt>最近任务</dt>
                  <dd>
                    <time v-if="user.lastTaskAt" class="last-task" :datetime="user.lastTaskAt">
                      {{ formatDateTime(user.lastTaskAt) }}
                    </time>
                    <span v-else class="last-task last-task--empty">从未创建任务</span>
                  </dd>
                </div>
              </dl>

              <div class="user-card__actions">
                <UiButton size="sm" :disabled="controlsDisabled" @click="recoveryCode(user)">
                  生成恢复码
                </UiButton>
                <UiButton
                  v-if="user.role !== 'admin'"
                  size="sm"
                  :variant="user.status === 'active' ? 'danger' : 'secondary'"
                  :disabled="controlsDisabled"
                  @click="toggleUserStatus(user)"
                >
                  {{ user.status === 'active' ? '禁用' : '启用' }}
                </UiButton>
              </div>
            </article>
          </div>
          <p v-else class="empty-state user-empty-state">
            {{ users.length ? '没有符合条件的用户' : '暂无用户' }}
          </p>
        </section>

        <section class="admin-card admin-card--block">
          <div class="section-heading">
            <div>
              <p class="section-kicker">注册访问</p>
              <h2>邀请码</h2>
              <p>邀请码 7 天有效且只能使用一次；关闭邀请码注册时不会被使用。</p>
            </div>
            <UiButton
              variant="primary"
              size="sm"
              :disabled="controlsDisabled"
              @click="createInvite"
            >
              创建邀请码
            </UiButton>
          </div>
          <div v-if="latestInvite" class="invite-code">
            <span>刚刚创建</span>
            <code>{{ latestInvite }}</code>
          </div>
          <div class="invite-list">
            <div v-for="invite in invites" :key="invite.id" class="invite-row">
              <code>{{ invite.id }}…</code>
              <span class="status-pill" :class="`status-pill--${invite.status}`">
                <span class="status-dot" aria-hidden="true" />
                {{ inviteStatusLabel(invite.status) }}
              </span>
              <time :datetime="invite.expiresAt">{{
                new Date(invite.expiresAt).toLocaleString()
              }}</time>
              <UiButton
                v-if="invite.status === 'active'"
                size="sm"
                :disabled="controlsDisabled"
                @click="
                  run(async () => {
                    await revokeAdminInvite(invite.id)
                  })
                "
              >
                撤销
              </UiButton>
            </div>
            <p v-if="!invites.length" class="empty-state">暂无邀请码</p>
          </div>
        </section>
      </template>
    </div>
  </main>
</template>

<style scoped>
.admin-page {
  --admin-surface: #fff;
  --admin-surface-subtle: #fafafa;
  --admin-surface-warm: #fafaf9;
  --admin-surface-muted: #f4f4f5;
  --admin-surface-used: #f5f5f4;
  --admin-text: #18181b;
  --admin-text-default: #27272a;
  --admin-text-secondary: #52525b;
  --admin-text-muted: #71717a;
  --admin-text-faint: #a1a1aa;
  --admin-border: #e4e4e7;
  --admin-border-strong: #d4d4d8;
  --admin-divider: #ececee;
  --admin-divider-soft: #f0f0f1;
  --admin-status-active-surface: #f0fdf4;
  --admin-status-active-text: #166534;
  --admin-status-active-dot: #22c55e;
  --admin-status-queued-surface: #fffbeb;
  --admin-status-queued-text: #92400e;
  --admin-status-queued-dot: #f59e0b;
  --admin-status-interrupted-surface: #fff7ed;
  --admin-status-interrupted-text: #9a3412;
  --admin-status-interrupted-dot: #f97316;
  --admin-warning-border: #fde68a;
  --admin-warning-surface: #fffbeb;
  --admin-warning-text: #92400e;
  --ui-button-primary-background: #18181b;
  --ui-button-primary-color: #fff;
  --ui-button-primary-shadow: none;
  --ui-button-primary-hover-background: #27272a;
  --ui-button-primary-hover-transform: none;
  --ui-button-primary-hover-shadow: none;
  --ui-button-secondary-background: #fff;
  --ui-button-secondary-color: #27272a;
  --ui-button-secondary-border: 1px solid #d4d4d8;
  --ui-button-secondary-hover-background: #f4f4f5;
  --ui-button-secondary-hover-border-color: #a1a1aa;
  --ui-button-danger-background: #fff;
  --ui-button-danger-color: #b42318;
  --ui-button-danger-border: 1px solid #f0b8b2;
  --ui-button-danger-shadow: none;
  --ui-button-danger-hover-background: #fff7f6;
  --ui-button-danger-hover-shadow: none;
  --ui-button-radius: 8px;
  --ui-input-background: #fff;
  --ui-input-color: #18181b;
  --ui-input-border: 1px solid #d4d4d8;
  --ui-input-focus-border: #18181b;
  --ui-input-focus-shadow: rgb(24 24 27 / 10%);
  --ui-switch-track-background: #d4d4d8;
  --ui-switch-track-checked-background: #18181b;
  --ui-switch-thumb-background: #fff;

  min-height: 100dvh;
  padding: 0 clamp(20px, 4vw, 56px) 64px;
  background: var(--admin-surface);
  color: var(--admin-text);
}

.admin-shell {
  width: min(100%, 1180px);
  margin: 0 auto;
}

.admin-header,
.section-heading,
.admin-card--quota,
.inline-control,
.quota-field,
.invite-row {
  display: flex;
  align-items: center;
}

.admin-header,
.section-heading,
.admin-card--quota {
  justify-content: space-between;
}

.admin-header {
  min-height: 150px;
  gap: 24px;
  padding: 32px 0 28px;
  border-bottom: 1px solid var(--admin-border);
}

.page-brand,
.section-kicker,
h1,
h2,
.admin-card p {
  margin: 0;
}

.page-brand,
.section-kicker {
  color: var(--admin-text-muted);
  font-size: 0.75rem;
  font-weight: 650;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

h1 {
  margin-top: 6px;
  font-size: clamp(2rem, 4vw, 2.75rem);
  font-weight: 650;
  letter-spacing: -0.045em;
  line-height: 1.08;
}

.page-description,
.admin-card p {
  margin-top: 8px;
  color: var(--admin-text-muted);
  line-height: 1.6;
}

.back-link {
  flex: 0 0 auto;
  padding: 9px 13px;
  border: 1px solid var(--admin-border-strong);
  border-radius: 8px;
  color: var(--admin-text-default);
  font-size: 0.875rem;
  font-weight: 550;
  text-decoration: none;
}

.back-link:hover {
  border-color: var(--admin-text-faint);
  background: var(--admin-surface-muted);
}

.product-status-banner {
  --product-status-banner-background: #fafafa;
  --product-status-banner-border: 1px solid #e4e4e7;
  --product-status-banner-accent: #52525b;
  --product-status-banner-radius: 10px;

  margin-top: 20px;
  overflow-wrap: anywhere;
}

.admin-card {
  margin-top: 20px;
  padding: 24px;
  border: 1px solid var(--admin-border);
  border-radius: 14px;
  background: var(--admin-surface);
}

.admin-card--quota {
  gap: 32px;
  background: var(--admin-surface-warm);
}

.admin-card--block {
  display: block;
}

.admin-card--loading {
  display: grid;
  min-height: 220px;
  place-items: center;
}

.admin-loading-state {
  display: flex;
  max-width: 420px;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  text-align: center;
}

.admin-loading-state .ui-spinner {
  margin-bottom: 4px;
  color: var(--admin-text);
}

.admin-loading-state .ui-button {
  margin-top: 6px;
}

.section-heading {
  gap: 24px;
}

.section-kicker {
  margin-bottom: 7px;
}

h2 {
  font-size: 1.1rem;
  font-weight: 650;
  letter-spacing: -0.015em;
}

.inline-control,
.quota-field {
  gap: 10px;
}

.inline-control {
  flex-wrap: wrap;
}

.quota-field > span {
  color: var(--admin-text-secondary);
  font-size: 0.875rem;
  white-space: nowrap;
}

.registration-control > span {
  color: var(--admin-text-secondary);
  font-size: 0.875rem;
  white-space: nowrap;
}

.admin-quota-input {
  --ui-number-field-input-width: 116px;
}

.scheduler-card {
  padding-bottom: 20px;
}

.scheduler-status-grid,
.scheduler-grid {
  display: grid;
  gap: 1px;
  overflow: hidden;
  border: 1px solid var(--admin-border);
  border-radius: 10px;
  background: var(--admin-border);
}

.scheduler-status-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin-top: 20px;
}

.scheduler-status-grid > div {
  min-width: 0;
  padding: 14px 16px;
  background: var(--admin-surface-subtle);
}

.scheduler-status-grid span,
.scheduler-status-grid strong {
  display: block;
}

.scheduler-status-grid span {
  color: var(--admin-text-muted);
  font-size: 0.72rem;
}

.scheduler-status-grid strong {
  margin-top: 5px;
  overflow: hidden;
  color: var(--admin-text-default);
  font-size: 0.86rem;
  font-variant-numeric: tabular-nums;
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.scheduler-current-task {
  padding: 12px 2px 18px;
  border-bottom: 1px solid var(--admin-divider);
  font-size: 0.8rem;
}

.scheduler-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 18px;
}

.scheduler-field {
  display: flex;
  min-width: 0;
  min-height: 82px;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 14px 16px;
  background: var(--admin-surface);
}

.scheduler-field > div {
  min-width: 0;
}

.scheduler-field strong,
.scheduler-field span {
  display: block;
}

.scheduler-field strong {
  color: var(--admin-text-default);
  font-size: 0.875rem;
  font-weight: 600;
}

.scheduler-field span {
  margin-top: 4px;
  color: var(--admin-text-muted);
  font-size: 0.76rem;
  line-height: 1.45;
}

.scheduler-select {
  flex: 0 0 150px;
  width: 150px;
}

.scheduler-recommendation,
.scheduler-warning {
  padding: 11px 13px;
  border-radius: 8px;
  font-size: 0.78rem;
}

.admin-card .scheduler-recommendation {
  margin-top: 14px;
  background: var(--admin-surface-muted);
}

.admin-card .scheduler-warning {
  margin-top: 8px;
  border: 1px solid var(--admin-warning-border);
  background: var(--admin-warning-surface);
  color: var(--admin-warning-text);
}

.policy-card {
  padding-bottom: 8px;
}

.policy-section {
  padding: 22px 0;
  border-top: 1px solid var(--admin-divider);
}

.policy-section:first-of-type {
  margin-top: 22px;
}

.policy-section__heading {
  margin-bottom: 14px;
}

.policy-section__heading h3 {
  margin: 0;
  font-size: 0.95rem;
  font-weight: 650;
}

.policy-section__heading p {
  margin-top: 5px;
  font-size: 0.82rem;
}

.policy-grid {
  display: grid;
  gap: 1px;
  overflow: hidden;
  border: 1px solid var(--admin-border);
  border-radius: 10px;
  background: var(--admin-border);
}

.policy-grid--features,
.policy-grid--settings {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.policy-grid--models {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.policy-row {
  display: flex;
  min-width: 0;
  min-height: 74px;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  padding: 14px 16px;
  background: var(--admin-surface);
}

.policy-row--compact {
  min-height: 56px;
}

.policy-row strong,
.policy-row span {
  display: block;
}

.policy-row strong {
  color: var(--admin-text-default);
  font-size: 0.875rem;
  font-weight: 600;
}

.policy-row span {
  margin-top: 4px;
  color: var(--admin-text-muted);
  font-size: 0.76rem;
  line-height: 1.45;
}

.policy-number-input {
  flex: 0 0 auto;

  --ui-number-field-input-width: 84px;
}

.user-toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin-top: 20px;
  padding-top: 18px;
  border-top: 1px solid var(--admin-divider);
}

.user-search {
  flex: 0 1 300px;
}

.user-status-filter {
  flex: 0 0 164px;
  width: 164px;
}

.user-result-count {
  margin-left: auto;
  color: var(--admin-text-muted);
  font-size: 0.78rem;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.user-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}

.user-card {
  display: flex;
  min-width: 0;
  flex-direction: column;
  padding: 16px;
  border: 1px solid var(--admin-border);
  border-radius: 11px;
  background: var(--admin-surface);
}

.user-card__header,
.user-card__task,
.user-card__actions {
  display: flex;
  align-items: center;
}

.user-card__header {
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.user-card__identity {
  min-width: 0;
}

.user-card__name {
  display: block;
  overflow: hidden;
  color: var(--admin-text);
  font-size: 0.94rem;
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.user-card__identity > span {
  display: block;
  margin-top: 4px;
  color: var(--admin-text-faint);
  font-size: 0.72rem;
  line-height: 1.4;
}

.user-card__task {
  min-width: 0;
  gap: 8px;
  margin-top: 14px;
}

.task-detail {
  min-width: 0;
  flex: 1;
  overflow: hidden;
  color: var(--admin-text-muted);
  font-size: 0.75rem;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.user-card__stats {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  row-gap: 10px;
  margin: 14px 0 12px;
  padding: 11px 0;
  border-top: 1px solid var(--admin-divider-soft);
  border-bottom: 1px solid var(--admin-divider-soft);
}

.user-stat {
  min-width: 0;
  padding: 0 4px;
  border-left: 1px solid var(--admin-divider-soft);
  text-align: center;
}

.user-stat:nth-child(3n + 1) {
  border-left: 0;
}

.user-stat strong,
.user-stat span {
  display: block;
}

.user-stat strong {
  color: var(--admin-text);
  font-size: 0.9rem;
  font-variant-numeric: tabular-nums;
  font-weight: 650;
}

.user-stat span {
  margin-top: 3px;
  overflow: hidden;
  color: var(--admin-text-muted);
  font-size: 0.65rem;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.user-stat--issue strong,
.user-stat--issue span {
  color: var(--admin-status-interrupted-text);
}

.user-card__meta {
  display: grid;
  gap: 7px;
  margin: 0;
}

.user-card__meta > div {
  display: flex;
  min-width: 0;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}

.user-card__meta dt,
.user-card__meta dd {
  margin: 0;
}

.user-card__meta dt {
  flex: 0 0 auto;
  color: var(--admin-text-muted);
  font-size: 0.72rem;
}

.user-card__meta dd {
  min-width: 0;
  overflow: hidden;
  color: var(--admin-text-default);
  font-size: 0.76rem;
  text-align: right;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.usage-value {
  color: var(--admin-text-default);
  font-variant-numeric: tabular-nums;
}

.usage-limit {
  color: var(--admin-text-faint);
}

.last-task {
  color: var(--admin-text-muted);
  font-size: 0.76rem;
}

.last-task--empty {
  color: var(--admin-text-faint);
}

.user-card__actions {
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px solid var(--admin-divider-soft);
}

.user-empty-state {
  margin-top: 14px;
  border-top: 1px solid var(--admin-divider);
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  width: fit-content;
  padding: 4px 8px;
  border-radius: 999px;
  background: var(--admin-surface-muted);
  color: var(--admin-text-secondary);
  font-size: 0.75rem;
  font-weight: 600;
  white-space: nowrap;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--admin-text-faint);
}

.status-pill--active {
  background: var(--admin-status-active-surface);
  color: var(--admin-status-active-text);
}

.status-pill--active .status-dot {
  background: var(--admin-status-active-dot);
}

.status-pill--task-active {
  background: var(--admin-surface-muted);
  color: var(--admin-text);
}

.status-pill--task-active .status-dot {
  background: var(--admin-text);
}

.status-pill--task-queued {
  background: var(--admin-status-queued-surface);
  color: var(--admin-status-queued-text);
}

.status-pill--task-queued .status-dot {
  background: var(--admin-status-queued-dot);
}

.status-pill--task-paused {
  background: var(--admin-surface-muted);
  color: var(--admin-text-secondary);
}

.status-pill--task-paused .status-dot {
  background: var(--admin-text-muted);
}

.status-pill--task-interrupted {
  background: var(--admin-status-interrupted-surface);
  color: var(--admin-status-interrupted-text);
}

.status-pill--task-interrupted .status-dot {
  background: var(--admin-status-interrupted-dot);
}

.status-pill--task-idle {
  background: var(--admin-surface-subtle);
  color: var(--admin-text-muted);
}

.status-pill--disabled,
.status-pill--revoked,
.status-pill--expired {
  background: var(--admin-surface-muted);
  color: var(--admin-text-muted);
}

.status-pill--used {
  background: var(--admin-surface-used);
  color: var(--admin-text-secondary);
}

.invite-code {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-top: 20px;
  padding: 14px 16px;
  border: 1px solid var(--admin-border-strong);
  border-radius: 10px;
  background: var(--admin-surface-subtle);
}

.invite-code span {
  color: var(--admin-text-muted);
  font-size: 0.78rem;
  font-weight: 600;
}

.invite-code code {
  overflow-wrap: anywhere;
  color: var(--admin-text);
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 0.95rem;
  user-select: all;
}

.invite-list {
  margin-top: 16px;
}

.invite-row {
  gap: 16px;
  padding: 13px 0;
  border-top: 1px solid var(--admin-divider);
}

.invite-row > code {
  color: var(--admin-text-default);
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 0.82rem;
}

.invite-row time {
  margin-left: auto;
  color: var(--admin-text-muted);
  font-size: 0.8rem;
  font-variant-numeric: tabular-nums;
}

.empty-state {
  padding: 28px 0 8px;
  text-align: center;
}

@media (--breakpoint-preview-down) {
  .admin-page {
    padding: 0 16px 40px;
  }

  .admin-header {
    min-height: auto;
    align-items: flex-start;
    padding: 24px 0 22px;
  }

  .admin-card {
    padding: 18px;
    border-radius: 12px;
  }

  .admin-card--quota,
  .section-heading {
    align-items: flex-start;
    flex-direction: column;
  }

  .admin-card--quota,
  .section-heading,
  .inline-control {
    gap: 16px;
  }

  .inline-control {
    align-items: stretch;
    width: 100%;
  }

  .quota-field {
    justify-content: space-between;
  }

  .policy-grid--features,
  .policy-grid--settings,
  .policy-grid--models,
  .scheduler-grid,
  .scheduler-status-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .user-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .invite-row {
    align-items: flex-start;
    flex-direction: column;
    gap: 8px;
  }

  .invite-row time {
    margin-left: 0;
  }
}

@media (--breakpoint-sm-down) {
  .admin-header {
    flex-direction: column;
    gap: 18px;
  }

  .back-link {
    width: 100%;
    text-align: center;
  }

  .inline-control {
    display: grid;
  }

  .inline-control .ui-button {
    width: 100%;
  }

  .invite-code {
    align-items: flex-start;
    flex-direction: column;
    gap: 6px;
  }

  .policy-grid--features,
  .policy-grid--settings,
  .policy-grid--models,
  .scheduler-grid,
  .scheduler-status-grid {
    grid-template-columns: 1fr;
  }

  .scheduler-field {
    align-items: flex-start;
    flex-direction: column;
  }

  .scheduler-select {
    width: 100%;
  }

  .user-toolbar {
    align-items: stretch;
  }

  .user-search,
  .user-status-filter {
    flex: 1 1 100%;
    width: 100%;
  }

  .user-result-count {
    margin-left: 0;
  }

  .user-grid {
    grid-template-columns: 1fr;
  }
}
</style>
