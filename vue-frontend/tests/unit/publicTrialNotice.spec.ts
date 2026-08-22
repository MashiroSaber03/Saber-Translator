import { enableAutoUnmount, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { createMemoryHistory, createRouter } from 'vue-router'
import { afterEach, describe, expect, it } from 'vitest'
import { nextTick } from 'vue'
import AccountDock from '@/components/common/AccountDock.vue'
import PublicTrialNotice from '@/components/common/PublicTrialNotice.vue'
import { useAuthStore } from '@/stores/authStore'
import { useRuntimeStore } from '@/stores/runtimeStore'
import type { RuntimeCapabilities } from '@/api/v2/auth'

enableAutoUnmount(afterEach)

const PUBLIC_CAPABILITIES: RuntimeCapabilities = {
  profile: 'public',
  requiresAuth: true,
  browserCredentials: true,
  registrationRequiresInvite: false,
  publicUserPolicy: {
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
      lamaDisableResize: { editable: false, value: false },
      parallel: { allowed: false },
    },
  },
  scheduling: { maxDeepLearningConcurrency: 1 },
  features: { plugins: false, webImport: false, localProviders: false },
}

function mountAccountDock(profile: 'local' | 'public') {
  const pinia = createPinia()
  setActivePinia(pinia)
  const runtime = useRuntimeStore()
  if (profile === 'public') runtime.capabilities = PUBLIC_CAPABILITIES
  else runtime.assumeLocalForTests()
  useAuthStore().user = { id: 'user-1', username: 'alice', role: 'user' }
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', component: { template: '<div />' } },
      { path: '/account', component: { template: '<div />' } },
      { path: '/login', component: { template: '<div />' } },
    ],
  })
  return mount(AccountDock, {
    attachTo: document.body,
    global: { plugins: [pinia, router] },
  })
}

describe('PublicTrialNotice', () => {
  afterEach(() => {
    document.body.innerHTML = ''
  })

  it('links to the official site and repository without an opener', () => {
    const wrapper = mount(PublicTrialNotice)
    const links = wrapper.findAll('a')

    expect(wrapper.text()).toContain('试用数据会不定期清理')
    expect(wrapper.text()).toContain('个人版完全开源免费，不包含任何收费功能')
    expect(links[0]?.attributes()).toMatchObject({
      href: 'https://www.mashirosaber.top/',
      rel: 'noopener noreferrer',
      target: '_blank',
    })
    expect(links[1]?.attributes()).toMatchObject({
      href: 'https://github.com/MashiroSaber03/Saber-Translator',
      rel: 'noopener noreferrer',
      target: '_blank',
    })
  })

  it('keeps the global trial entry and dialog isolated to the public profile', async () => {
    const localWrapper = mountAccountDock('local')
    expect(localWrapper.text()).not.toContain('试用说明')
    localWrapper.unmount()

    const publicWrapper = mountAccountDock('public')
    const trigger = publicWrapper
      .findAll('button')
      .find(button => button.text() === '试用说明')
    expect(trigger?.attributes('aria-haspopup')).toBe('dialog')

    await trigger!.trigger('click')
    await nextTick()
    const dialog = document.body.querySelector('[role="dialog"]')
    expect(dialog?.textContent).toContain('关于在线试用版')
    expect(dialog?.textContent).toContain('个人版完全开源免费')
  })
})
