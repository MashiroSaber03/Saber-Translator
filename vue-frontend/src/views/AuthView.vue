<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { recoverPassword } from '@/api/v2/auth'
import { useAuthStore } from '@/stores/authStore'
import { useRuntimeStore } from '@/stores/runtimeStore'
import PublicTrialNotice from '@/components/common/PublicTrialNotice.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiInput from '@/components/ui/UiInput.vue'

const route = useRoute()
const router = useRouter()
const auth = useAuthStore()
const runtime = useRuntimeStore()
const logoUrl = '/pic/logo.png'
const username = ref('')
const password = ref('')
const inviteCode = ref('')
const recoveryCode = ref('')
const error = ref('')
const busy = ref(false)
const recoveryCodes = ref<string[]>([])
const registrationRequiresInvite = computed(
  () => runtime.capabilities?.registrationRequiresInvite !== false,
)
const isPublicProfile = computed(() => runtime.capabilities?.profile === 'public')

const mode = computed(() => String(route.name ?? 'login'))
const title = computed(() =>
  mode.value === 'register' ? '创建账户' : mode.value === 'recover' ? '重设密码' : '登录'
)
const description = computed(() =>
  mode.value === 'register'
    ? registrationRequiresInvite.value
      ? '使用管理员提供的一次性邀请码注册。'
      : '无需邀请码，设置用户名和密码即可注册。'
    : mode.value === 'recover'
      ? '输入一次性恢复码，并为账户设置新密码。'
      : '继续管理你的漫画翻译项目。'
)

watch(mode, () => {
  error.value = ''
})

async function submit(): Promise<void> {
  error.value = ''
  busy.value = true
  try {
    if (mode.value === 'register') {
      recoveryCodes.value = await auth.register(
        username.value,
        password.value,
        registrationRequiresInvite.value ? inviteCode.value : undefined,
      )
      return
    }
    if (mode.value === 'recover') {
      await recoverPassword(username.value, recoveryCode.value, password.value)
      await router.replace({ name: 'login' })
      return
    }
    await auth.login(username.value, password.value)
    const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '/'
    await router.replace(redirect)
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : '操作失败，请稍后重试'
  } finally {
    busy.value = false
  }
}

function downloadRecoveryCodes(): void {
  const blob = new Blob(
    [`Saber Translator 恢复码（每个只能使用一次）\n\n${recoveryCodes.value.join('\n')}\n`],
    { type: 'text/plain;charset=utf-8' }
  )
  const link = document.createElement('a')
  link.href = URL.createObjectURL(blob)
  link.download = 'saber-recovery-codes.txt'
  link.click()
  URL.revokeObjectURL(link.href)
}
</script>

<template>
  <main class="auth-page">
    <section class="auth-card">
      <div class="auth-brand">
        <img :src="logoUrl" alt="" class="auth-logo" />
        <span>Saber Translator</span>
      </div>
      <template v-if="recoveryCodes.length">
        <div class="auth-heading">
          <h1>保存恢复码</h1>
          <p class="auth-description">恢复码只显示这一次。请下载并保存到安全的位置。</p>
        </div>
        <pre class="recovery-list">{{ recoveryCodes.join('\n') }}</pre>
        <div class="auth-actions auth-actions--stack">
          <UiButton variant="primary" size="lg" type="button" @click="downloadRecoveryCodes">
            下载恢复码
          </UiButton>
          <UiButton type="button" @click="router.replace('/')">我已保存，进入应用</UiButton>
        </div>
      </template>
      <form v-else @submit.prevent="submit">
        <div class="auth-heading">
          <h1>{{ title }}</h1>
          <p class="auth-description">{{ description }}</p>
        </div>
        <UiField label="用户名" control-id="auth-username" required>
          <UiInput
            id="auth-username"
            v-model.trim="username"
            class="auth-input"
            autocomplete="username"
            required
            minlength="3"
            maxlength="32"
            size="lg"
          />
        </UiField>
        <UiField
          v-if="mode === 'register' && registrationRequiresInvite"
          label="邀请码"
          control-id="auth-invite"
          required
        >
          <UiInput
            id="auth-invite"
            v-model.trim="inviteCode"
            class="auth-input"
            autocomplete="off"
            required
            size="lg"
          />
        </UiField>
        <UiField v-if="mode === 'recover'" label="恢复码" control-id="auth-recovery" required>
          <UiInput
            id="auth-recovery"
            v-model.trim="recoveryCode"
            class="auth-input"
            autocomplete="off"
            required
            size="lg"
          />
        </UiField>
        <UiField
          :label="mode === 'recover' ? '新密码' : '密码'"
          control-id="auth-password"
          required
        >
          <UiInput
            id="auth-password"
            v-model="password"
            class="auth-input"
            type="password"
            :autocomplete="mode === 'login' ? 'current-password' : 'new-password'"
            required
            minlength="10"
            maxlength="128"
            size="lg"
          />
        </UiField>
        <ProductStatusBanner v-if="error" tone="danger" role="alert">
          {{
            error
          }}
        </ProductStatusBanner>
        <UiButton variant="primary" size="lg" type="submit" :loading="busy" block>
          {{
            busy
              ? '处理中…'
              : mode === 'register'
                ? '注册'
                : mode === 'recover'
                  ? '重设密码'
                  : '登录'
          }}
        </UiButton>
        <nav class="auth-links" aria-label="账号操作">
          <RouterLink v-if="mode !== 'login'" to="/login">返回登录</RouterLink>
          <RouterLink v-if="mode === 'login'" to="/register">
            {{ registrationRequiresInvite ? '使用邀请码注册' : '注册账户' }}
          </RouterLink>
          <RouterLink v-if="mode === 'login'" to="/recover">忘记密码</RouterLink>
        </nav>
      </form>
      <PublicTrialNotice
        v-if="isPublicProfile"
        class="auth-trial-notice"
        compact
      />
    </section>
  </main>
</template>

<style scoped>
.auth-page {
  --auth-page-background: #f7f7f5;
  --auth-border: #e3e3df;
  --auth-link: #52525b;
  --auth-link-decoration: #c4c4bf;
  --auth-card-shadow: rgba(24, 24, 27, 0.04);
  --color-text-default: #27272a;
  --color-text-strong: #18181b;
  --color-text-secondary: #71717a;
  --color-text-supporting: #71717a;
  --color-border-muted: #deded9;
  --color-border-brand: #18181b;
  --color-surface-card: #fff;
  --color-surface-input: #fff;
  --color-surface-quiet: #fafaf9;
  --color-focus-brand-subtle: rgba(24, 24, 27, 0.1);
  --ui-input-min-height: 48px;
  --ui-input-padding: 12px 14px;
  --ui-input-border: 1px solid #d8d8d4;
  --ui-input-radius: 10px;
  --ui-input-background: #fff;
  --ui-input-color: #18181b;
  --ui-input-focus-border: #18181b;
  --ui-input-focus-shadow: rgba(24, 24, 27, 0.1);
  --ui-button-radius: 10px;
  --ui-button-primary-background: #18181b;
  --ui-button-primary-hover-background: #2f2f32;
  --ui-button-primary-color: #fff;
  --ui-button-primary-shadow: none;
  --ui-button-primary-hover-shadow: none;
  --ui-button-primary-hover-transform: none;
  --ui-button-secondary-background: #fff;
  --ui-button-secondary-hover-background: #f4f4f2;
  --ui-button-secondary-border: 1px solid #d8d8d4;
  --ui-button-secondary-hover-border-color: #a1a19a;

  box-sizing: border-box;
  display: grid;
  min-height: 100dvh;
  place-items: center;
  padding: 48px 24px;
  background: var(--auth-page-background);
  color: var(--color-text-default);
  color-scheme: light;
}

.auth-card {
  box-sizing: border-box;
  width: min(430px, 100%);
  padding: 34px;
  border: 1px solid var(--auth-border);
  border-radius: 18px;
  background: var(--color-surface-card);
  box-shadow: 0 1px 2px var(--auth-card-shadow);
}

.auth-brand {
  display: flex;
  align-items: center;
  gap: 11px;
  margin-bottom: 34px;
  color: var(--color-text-strong);
  font-size: 0.94rem;
  font-weight: 650;
  letter-spacing: -0.01em;
}

.auth-logo {
  width: 34px;
  height: 34px;
  border: 1px solid var(--auth-border);
  border-radius: 9px;
  object-fit: cover;
}

.auth-heading {
  margin-bottom: 28px;
}

h1 {
  margin: 0;
  color: var(--color-text-strong);
  font-size: clamp(1.75rem, 4vw, 2rem);
  font-weight: 680;
  letter-spacing: -0.035em;
  line-height: 1.2;
}

.auth-description {
  margin: 9px 0 0;
  color: var(--color-text-secondary);
  font-size: 0.94rem;
  line-height: 1.6;
}

form {
  display: block;
}

.auth-input:-webkit-autofill,
.auth-input:-webkit-autofill:hover,
.auth-input:-webkit-autofill:focus {
  box-shadow: 0 0 0 1000px var(--color-surface-card) inset;
  caret-color: var(--color-text-strong);
  -webkit-text-fill-color: var(--color-text-strong);
}

.product-status-banner {
  --product-status-banner-padding: 11px 12px;
  --product-status-banner-radius: 10px;

  margin: 2px 0 18px;
}

.auth-links {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px 18px;
  margin-top: 18px;
  font-size: 0.88rem;
}

.auth-links a {
  color: var(--auth-link);
  text-decoration-color: var(--auth-link-decoration);
  text-underline-offset: 4px;
}

.auth-links a:hover {
  color: var(--color-text-strong);
  text-decoration-color: var(--color-text-strong);
}

.auth-trial-notice {
  margin: 28px 0 0;
}

.recovery-list {
  max-height: 280px;
  margin: 0 0 20px;
  padding: 16px;
  overflow: auto;
  border: 1px solid var(--color-border-muted);
  border-radius: 10px;
  background: var(--color-surface-quiet);
  color: var(--color-text-strong);
  font-size: 0.9rem;
  line-height: 1.75;
  user-select: all;
}

.auth-actions--stack {
  display: grid;
  gap: 10px;
}

@media (--breakpoint-preview-down) {
  .auth-page {
    align-items: start;
    padding: 28px 20px;
    background: var(--color-surface-card);
  }

  .auth-card {
    padding: 0;
    border: 0;
    border-radius: 0;
    box-shadow: none;
  }

  .auth-brand {
    margin-bottom: 38px;
  }
}
</style>
