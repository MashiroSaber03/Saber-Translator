import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'

import type { PublicUserPolicy, RuntimeCapabilities } from '@/api/v2/auth'
import ParallelSettings from '@/components/settings/ParallelSettings.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import { usePublicUserAccess } from '@/composables/usePublicUserAccess'
import { useAuthStore } from '@/stores/authStore'
import { useRuntimeStore } from '@/stores/runtimeStore'
import { useSettingsStore } from '@/stores/settings'

function policy(): PublicUserPolicy {
  return {
    features: {
      translation: false,
      insight: true,
      characterStudio: true,
      editMode: false,
    },
    models: {
      detector_default: false,
      detector_ctd: true,
      detector_yolo: true,
      aux_ysg_yolo: true,
      saber_yolo: true,
      manga_ocr: true,
      ocr_48px: true,
      paddle_ocr: true,
      paddleocr_vl: false,
      lama_mpe: false,
      litelama: true,
      lama_manga: true,
    },
    settings: {
      lamaDisableResize: { editable: false, value: true },
      parallel: { allowed: false },
    },
  }
}

function capabilities(profile: 'local' | 'public' = 'public'): RuntimeCapabilities {
  return {
    profile,
    requiresAuth: profile === 'public',
    browserCredentials: profile === 'public',
    registrationRequiresInvite: true,
    publicUserPolicy: policy(),
    scheduling: { maxDeepLearningConcurrency: 2 },
    features: {
      plugins: profile === 'local',
      webImport: profile === 'local',
      localProviders: profile === 'local',
    },
  }
}

describe('ordinary public-user policy projection', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    useRuntimeStore().capabilities = capabilities()
    useAuthStore().user = { id: 'user-1', username: 'alice', role: 'user' }
  })

  it('projects feature, model, and setting restrictions for ordinary users', () => {
    const access = usePublicUserAccess()

    expect(access.featureAllowed('translation')).toBe(false)
    expect(access.featureAllowed('insight')).toBe(true)
    expect(access.modelAllowed('paddleocr_vl')).toBe(false)
    expect(access.lamaDisableResizeEditable()).toBe(false)
    expect(access.lamaDisableResizeValue()).toBe(true)
    expect(access.parallelAllowed()).toBe(false)
    expect(access.maxDeepLearningConcurrency()).toBe(2)
    expect(
      access.modelOptions([{ label: 'PaddleOCR-VL', value: 'paddleocr_vl' }], {
        paddleocr_vl: 'paddleocr_vl',
      })
    ).toEqual([
      {
        label: 'PaddleOCR-VL（管理员已关闭）',
        value: 'paddleocr_vl',
        disabled: true,
      },
    ])
  })

  it('does not apply ordinary-user permissions to admins, but keeps the global concurrency cap', () => {
    const auth = useAuthStore()
    const runtime = useRuntimeStore()
    const access = usePublicUserAccess()

    auth.user = { id: 'admin-1', username: 'admin', role: 'admin' }
    expect(access.featureAllowed('translation')).toBe(true)
    expect(access.modelAllowed('detector_default')).toBe(true)
    expect(access.parallelAllowed()).toBe(true)
    expect(access.maxDeepLearningConcurrency()).toBe(2)

    auth.user = { id: 'user-1', username: 'alice', role: 'user' }
    runtime.capabilities = capabilities('local')
    expect(access.featureAllowed('translation')).toBe(true)
    expect(access.modelAllowed('detector_default')).toBe(true)
    expect(access.maxDeepLearningConcurrency()).toBeNull()
  })

  it('keeps a public administrator limit lower than the pool count', () => {
    const store = useSettingsStore()
    store.settings.parallel.enabled = true
    const runtime = useRuntimeStore()
    runtime.capabilities = capabilities('public')
    runtime.capabilities.publicUserPolicy.settings.parallel.allowed = true
    const wrapper = mount(ParallelSettings)
    const field = wrapper.getComponent(UiNumberField)
    expect(field.props('max')).toBe(2)
    field.vm.$emit('update:modelValue', 4)
    expect(store.settings.parallel.deepLearningLockSize).toBe(2)
  })
})
