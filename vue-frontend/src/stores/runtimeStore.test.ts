import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, expect, it, vi } from 'vitest'
import { useRuntimeStore } from './runtimeStore'
import { usePublicUserAccess } from '../composables/usePublicUserAccess'
import { INPAINT_MODEL_KEYS } from '../utils/localModelAvailability'

const api = vi.hoisted(() => ({ getCapabilities: vi.fn() }))
vi.mock('@/api/v2/auth', () => api)
beforeEach(() => { setActivePinia(createPinia()); vi.clearAllMocks() })

it('refreshes one shared inventory and retains it on transient errors', async () => {
  const store = useRuntimeStore()
  store.assumeLocalForTests()
  const initial = { ...store.capabilities! }
  const updated = { ...initial, installedModels: ['paddle_ocr'] }
  let finish!: (value: typeof updated) => void
  api.getCapabilities.mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
  const first = store.load(true)
  const second = store.load(true)
  expect(api.getCapabilities).toHaveBeenCalledTimes(1)
  finish(updated)
  await Promise.all([first, second])
  expect(store.capabilities!.installedModels).toEqual(['paddle_ocr'])
  const current = store.capabilities
  api.getCapabilities.mockResolvedValueOnce(updated)
  await store.load(true)
  expect(store.capabilities).toBe(current)
  await store.load()
  expect(api.getCapabilities).toHaveBeenCalledTimes(2)
  api.getCapabilities.mockRejectedValueOnce(new Error('offline'))
  await expect(store.load(true)).rejects.toThrow('offline')
  expect(store.capabilities!.installedModels).toEqual(['paddle_ocr'])
})

it('keeps pure fill and remote choices, hides missing models, and respects public policy', () => {
  const store = useRuntimeStore()
  store.assumeLocalForTests()
  store.capabilities = { ...store.capabilities!, installedModels: ['lama_manga'] }
  const access = usePublicUserAccess()
  const options = [{ label: '纯色', value: 'solid' }, { label: '漫画', value: 'lama_manga' }, { label: '通用', value: 'litelama' }]
  expect(access.modelOptions(options, INPAINT_MODEL_KEYS).map(option => option.value)).toEqual(['solid', 'lama_manga'])
  store.capabilities.profile = 'public'
  store.capabilities.publicUserPolicy.models.lama_manga = false
  expect(access.modelOptions(options, INPAINT_MODEL_KEYS)).toEqual([
    options[0], { ...options[1], label: '漫画（管理员已关闭）', disabled: true },
  ])
  store.capabilities.installedModels.push('detector_ctd', 'detector_default')
  store.capabilities.publicUserPolicy.models.detector_default = false
  expect(access.modelAllowed('detector_ctd')).toBe(false)
})
