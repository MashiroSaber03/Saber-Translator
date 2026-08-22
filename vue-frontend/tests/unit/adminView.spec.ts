import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import UiSelect from '@/components/ui/UiSelect.vue'
import AdminView from '@/views/AdminView.vue'
import type { AdminUser, PublicUserPolicy } from '@/api/v2/auth'

const authMocks = vi.hoisted(() => ({
  listAdminUsers: vi.fn(),
  listAdminInvites: vi.fn(),
  getAssetQuota: vi.fn(),
  getRegistrationPolicy: vi.fn(),
  getPublicUserPolicy: vi.fn(),
  createAdminInvite: vi.fn(),
  createUserRecoveryCode: vi.fn(),
  revokeAdminInvite: vi.fn(),
  setAdminUserStatus: vi.fn(),
  setAssetQuota: vi.fn(),
  setRegistrationPolicy: vi.fn(),
  setPublicUserPolicy: vi.fn(),
}))

vi.mock('@/api/v2/auth', () => authMocks)

const viewSource = readFileSync(resolve(process.cwd(), 'src/views/AdminView.vue'), 'utf8')
const apiSource = readFileSync(resolve(process.cwd(), 'src/api/v2/auth.ts'), 'utf8')

const publicUserPolicy: PublicUserPolicy = {
  features: {
    translation: true,
    insight: true,
    characterStudio: true,
    editMode: true,
  },
  models: {
    detector_default: true,
    detector_ctd: true,
    detector_yolo: true,
    aux_ysg_yolo: true,
    saber_yolo: true,
    manga_ocr: true,
    ocr_48px: true,
    paddle_ocr: true,
    paddleocr_vl: true,
    lama_mpe: true,
    litelama: true,
  },
  settings: {
    lamaDisableResize: { editable: true, value: false },
    parallel: { allowed: false, maxDeepLearningConcurrency: 1 },
  },
}

function adminUser(
  username: string,
  taskStatus: AdminUser['taskStatus'],
  status: AdminUser['status'] = 'active'
): AdminUser {
  return {
    id: `user-${username}`,
    username,
    role: 'user',
    status,
    assetUsageBytes: 1024,
    assetQuotaBytes: 2 * 1024 ** 3,
    createdAt: '2026-08-22T00:00:00Z',
    taskStatus,
    activeTaskCount: taskStatus === 'active' ? 1 : 0,
    queuedTaskCount: taskStatus === 'queued' ? 1 : 0,
    interruptedTaskCount: taskStatus === 'interrupted' ? 1 : 0,
    completedTaskCount: 3,
    issueTaskCount: 0,
    currentTaskKind: taskStatus === 'idle' ? null : 'translate',
    currentTaskStartedAt: taskStatus === 'active' ? '2026-08-22T01:00:00Z' : null,
    lastTaskAt: '2026-08-22T01:00:00Z',
  }
}

describe('AdminView user task overview', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    authMocks.listAdminUsers.mockResolvedValue([])
    authMocks.listAdminInvites.mockResolvedValue([])
    authMocks.getAssetQuota.mockResolvedValue({ assetQuotaBytes: 2 * 1024 ** 3 })
    authMocks.getRegistrationPolicy.mockResolvedValue({ registrationRequiresInvite: false })
    authMocks.getPublicUserPolicy.mockResolvedValue(structuredClone(publicUserPolicy))
  })

  it('defines the retained task activity returned for each administrator user row', () => {
    expect(apiSource).toContain("taskStatus: 'active' | 'queued' | 'interrupted' | 'idle'")
    expect(apiSource).toContain('completedTaskCount: number')
    expect(apiSource).toContain('issueTaskCount: number')
    expect(apiSource).toContain('currentTaskKind: string | null')
    expect(apiSource).toContain('lastTaskAt: string | null')
  })

  it('shows workload status, current task, retained totals, and the latest task time', () => {
    expect(viewSource).toContain('taskStatusLabel(user.taskStatus)')
    expect(viewSource).toContain('jobKindLabel(user.currentTaskKind)')
    expect(viewSource).toContain('近期完成')
    expect(viewSource).toContain('近期异常')
    expect(viewSource).toContain('<dt>最近任务</dt>')
  })

  it('sorts important states first and filters the compact cards without another API request', async () => {
    authMocks.listAdminUsers.mockResolvedValue([
      adminUser('idle-user', 'idle'),
      adminUser('disabled-user', 'idle', 'disabled'),
      adminUser('queued-user', 'queued'),
      adminUser('active-user', 'active'),
      adminUser('interrupted-user', 'interrupted'),
    ])

    const wrapper = mount(AdminView, {
      global: {
        stubs: {
          RouterLink: { template: '<a><slot /></a>' },
        },
      },
    })
    await flushPromises()

    const visibleNames = () => wrapper.findAll('.user-card__name').map(node => node.text())
    expect(visibleNames()).toEqual([
      'active-user',
      'queued-user',
      'interrupted-user',
      'idle-user',
      'disabled-user',
    ])
    expect(wrapper.text()).toContain('显示 5 / 5 位用户')

    await wrapper.get('[aria-label="搜索用户名"]').setValue('QUEUED')
    expect(visibleNames()).toEqual(['queued-user'])

    await wrapper.get('[aria-label="搜索用户名"]').setValue('')
    wrapper.getComponent(UiSelect).vm.$emit('update:modelValue', 'interrupted')
    await wrapper.vm.$nextTick()
    expect(visibleNames()).toEqual(['interrupted-user'])

    wrapper.getComponent(UiSelect).vm.$emit('update:modelValue', 'disabled')
    await wrapper.vm.$nextTick()
    expect(visibleNames()).toEqual(['disabled-user'])
    expect(authMocks.listAdminUsers).toHaveBeenCalledTimes(1)

    wrapper.unmount()
  })

  it('does not render server-backed controls before one complete admin snapshot is ready', async () => {
    let resolveQuota!: (value: { assetQuotaBytes: number }) => void
    authMocks.getAssetQuota.mockReturnValueOnce(
      new Promise(resolve => {
        resolveQuota = resolve
      })
    )

    const wrapper = mount(AdminView, {
      global: {
        stubs: {
          RouterLink: { template: '<a><slot /></a>' },
        },
      },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('正在读取管理数据')
    expect(wrapper.text()).not.toContain('每用户资产额度')
    expect(wrapper.find('[aria-label="每用户资产额度 GB"]').exists()).toBe(false)
    expect(wrapper.find('[aria-label="注册必须使用邀请码"]').exists()).toBe(false)

    resolveQuota({ assetQuotaBytes: 5.5 * 1024 ** 3 })
    await flushPromises()

    const quotaInput = wrapper.get('[aria-label="每用户资产额度 GB"]')
    expect((quotaInput.element as HTMLInputElement).value).toBe('5.5')
    expect(wrapper.get('[aria-label="注册必须使用邀请码"]').attributes('aria-checked')).toBe(
      'false'
    )
    expect(wrapper.text()).toContain('当前允许自由注册')
    wrapper.unmount()
  })

  it('keeps server-backed controls unmounted when initial admin loading fails', async () => {
    authMocks.getAssetQuota.mockRejectedValueOnce(new Error('额度接口不可用'))

    const wrapper = mount(AdminView, {
      global: {
        stubs: {
          RouterLink: { template: '<a><slot /></a>' },
        },
      },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('额度接口不可用')
    expect(wrapper.text()).toContain('管理数据尚未就绪')
    expect(wrapper.text()).toContain('重新加载')
    expect(wrapper.find('[aria-label="每用户资产额度 GB"]').exists()).toBe(false)
    expect(wrapper.find('[aria-label="注册必须使用邀请码"]').exists()).toBe(false)
    wrapper.unmount()
  })

  it('exposes one global ordinary-user policy without per-user permission layers', () => {
    expect(apiSource).toContain('export interface PublicUserPolicy')
    expect(apiSource).toContain("'translation' | 'insight' | 'characterStudio' | 'editMode'")
    expect(apiSource).toContain('getPublicUserPolicy')
    expect(apiSource).toContain('setPublicUserPolicy')
    expect(viewSource).toContain('功能与性能控制')
    expect(viewSource).toContain('页面与模式')
    expect(viewSource).toContain('本地模型')
    expect(viewSource).toContain('允许修改“禁用自动缩放”')
    expect(viewSource).toContain('深度学习并发上限')
    expect(viewSource).toContain('只限制普通用户；管理员和本地模式不受影响')
    expect(viewSource).not.toContain('用户组')
    expect(viewSource).not.toContain('权限模板')
  })
})
