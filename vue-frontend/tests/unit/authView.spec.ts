import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { flushPromises, mount } from '@vue/test-utils'
import { createMemoryHistory, createRouter } from 'vue-router'
import { describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  login: vi.fn(),
  register: vi.fn(),
}))

vi.mock('@/stores/authStore', () => ({
  useAuthStore: () => mocks,
}))

vi.mock('@/stores/runtimeStore', () => ({
  useRuntimeStore: () => ({
    capabilities: {
      profile: 'public',
      registrationRequiresInvite: false,
    },
  }),
}))

vi.mock('@/api/v2/auth', () => ({
  recoverPassword: vi.fn(),
}))

import AuthView from '@/views/AuthView.vue'

const source = readFileSync(resolve(process.cwd(), 'src/views/AuthView.vue'), 'utf8')

describe('AuthView registration policy', () => {
  it('only renders and submits an invite code when the runtime requires one', () => {
    expect(source).toContain(`mode === 'register' && registrationRequiresInvite`)
    expect(source).toContain(
      `registrationRequiresInvite.value ? inviteCode.value : undefined`,
    )
  })

  it('explains both invite-only and free registration modes', () => {
    expect(source).toContain('使用管理员提供的一次性邀请码注册。')
    expect(source).toContain('无需邀请码，设置用户名和密码即可注册。')
    expect(source).toContain("registrationRequiresInvite ? '使用邀请码注册' : '注册账户'")
  })

  it('shows the shared trial notice only in the public profile', () => {
    expect(source).toContain(
      "import PublicTrialNotice from '@/components/common/PublicTrialNotice.vue'",
    )
    expect(source).toContain("runtime.capabilities?.profile === 'public'")
    expect(source).toContain('v-if="isPublicProfile"')
    expect(source).toContain('<PublicTrialNotice')
  })

  it('clears a failed submission message when switching auth modes', async () => {
    mocks.login.mockRejectedValueOnce(new Error('用户名或密码错误'))
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [
        { path: '/login', name: 'login', component: AuthView },
        { path: '/register', name: 'register', component: AuthView },
        { path: '/recover', name: 'recover', component: AuthView },
      ],
    })
    await router.push('/login')
    await router.isReady()
    const wrapper = mount(AuthView, { global: { plugins: [router] } })

    await wrapper.get('#auth-username').setValue('alice')
    await wrapper.get('#auth-password').setValue('invalid-password')
    await wrapper.get('form').trigger('submit')
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('用户名或密码错误')

    await router.push('/recover')
    await flushPromises()
    expect(wrapper.find('[role="alert"]').exists()).toBe(false)
  })
})
