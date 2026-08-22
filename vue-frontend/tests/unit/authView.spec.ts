import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

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
})
