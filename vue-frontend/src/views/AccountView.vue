<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { changePassword } from '@/api/v2/auth'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiInput from '@/components/ui/UiInput.vue'
import { useAuthStore } from '@/stores/authStore'

const auth = useAuthStore()
const router = useRouter()
const currentPassword = ref('')
const newPassword = ref('')
const confirmPassword = ref('')
const error = ref('')
const busy = ref(false)
const userInitial = computed(() => auth.user?.username.slice(0, 1).toUpperCase() || 'S')
const usagePercent = computed(() =>
  auth.assetQuotaBytes > 0
    ? Math.min(100, Math.max(0, (auth.assetUsageBytes / auth.assetQuotaBytes) * 100))
    : 0
)

onMounted(async () => {
  try {
    await auth.refresh()
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : '资产用量刷新失败'
  }
})

function formatBytes(value: number): string {
  const gib = 1024 ** 3
  return value >= gib ? `${(value / gib).toFixed(2)} GB` : `${(value / 1024 ** 2).toFixed(1)} MB`
}

async function submit(): Promise<void> {
  error.value = ''
  if (newPassword.value !== confirmPassword.value) {
    error.value = '两次输入的新密码不一致'
    return
  }
  busy.value = true
  try {
    await changePassword(currentPassword.value, newPassword.value)
    auth.markUnauthenticated()
    await router.replace({ name: 'login' })
  } catch (reason) {
    error.value = reason instanceof Error ? reason.message : '修改密码失败'
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <main class="account-page">
    <div class="account-shell">
      <header class="account-header">
        <div>
          <p class="page-brand">Saber Translator</p>
          <h1>账户设置</h1>
          <p class="page-description">查看资产用量，并管理账户安全。</p>
        </div>
        <RouterLink class="back-link" to="/">返回应用</RouterLink>
      </header>

      <section class="account-section account-overview">
        <div class="account-identity">
          <div class="account-avatar" aria-hidden="true">{{ userInitial }}</div>
          <div>
            <span class="section-label">当前账户</span>
            <h2>{{ auth.user?.username }}</h2>
          </div>
        </div>

        <div class="usage-summary">
          <div class="usage-heading">
            <span class="section-label">资产用量</span>
            <strong>{{ formatBytes(auth.assetUsageBytes) }} /
              {{ formatBytes(auth.assetQuotaBytes) }}</strong>
          </div>
          <div
            class="usage-track"
            role="progressbar"
            aria-label="资产额度使用进度"
            :aria-valuenow="Math.round(usagePercent)"
            aria-valuemin="0"
            aria-valuemax="100"
          >
            <span :style="{ width: `${usagePercent}%` }" />
          </div>
        </div>

        <ProductStatusBanner icon-name="lock" tone="neutral" role="note" title="密钥仅保存在浏览器">
          Provider 密钥会按需临时加载到本机服务内存，不会写入项目数据库或备份。
        </ProductStatusBanner>
      </section>

      <section class="account-section">
        <header class="section-heading">
          <div>
            <span class="section-label">账户安全</span>
            <h2>修改密码</h2>
          </div>
          <p>修改成功后，所有已登录会话都会退出。</p>
        </header>

        <form class="account-form" @submit.prevent="submit">
          <UiField
            class="account-field--wide"
            label="当前密码"
            control-id="current-password"
            required
          >
            <UiInput
              id="current-password"
              v-model="currentPassword"
              type="password"
              autocomplete="current-password"
              required
              minlength="10"
              maxlength="128"
            />
          </UiField>
          <UiField label="新密码" control-id="new-password" required>
            <UiInput
              id="new-password"
              v-model="newPassword"
              type="password"
              autocomplete="new-password"
              required
              minlength="10"
              maxlength="128"
            />
          </UiField>
          <UiField label="确认新密码" control-id="confirm-password" required>
            <UiInput
              id="confirm-password"
              v-model="confirmPassword"
              type="password"
              autocomplete="new-password"
              required
              minlength="10"
              maxlength="128"
            />
          </UiField>
          <ProductStatusBanner
            v-if="error"
            class="account-field--wide"
            tone="danger"
            role="alert"
          >
            {{ error }}
          </ProductStatusBanner>
          <div class="account-field--wide form-actions">
            <UiButton variant="primary" type="submit" :loading="busy">修改密码</UiButton>
          </div>
        </form>
      </section>
    </div>
  </main>
</template>

<style scoped>
.account-page {
  --account-border: #e3e3df;
  --account-divider: #e7e7e3;
  --account-progress-track: #ecece8;
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
  --ui-input-min-height: 44px;
  --ui-input-padding: 10px 12px;
  --ui-input-border: 1px solid #d8d8d4;
  --ui-input-radius: 9px;
  --ui-input-background: #fff;
  --ui-input-color: #18181b;
  --ui-input-focus-border: #18181b;
  --ui-input-focus-shadow: rgba(24, 24, 27, 0.1);
  --ui-button-radius: 9px;
  --ui-button-primary-background: #18181b;
  --ui-button-primary-hover-background: #2f2f32;
  --ui-button-primary-color: #fff;
  --ui-button-primary-shadow: none;
  --ui-button-primary-hover-shadow: none;
  --ui-button-primary-hover-transform: none;

  box-sizing: border-box;
  min-height: 100dvh;
  padding: 42px clamp(20px, 5vw, 72px) 80px;
  background: var(--color-surface-card);
  color: var(--color-text-default);
  color-scheme: light;
}

.account-shell {
  width: min(920px, 100%);
  margin: 0 auto;
}

.account-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
  padding-bottom: 28px;
  border-bottom: 1px solid var(--account-divider);
}

.page-brand,
.page-description,
.account-header h1,
.account-section h2,
.section-heading p {
  margin: 0;
}

.page-brand,
.section-label {
  color: var(--color-text-secondary);
  font-size: 0.78rem;
  font-weight: 650;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.account-header h1 {
  margin-top: 6px;
  color: var(--color-text-strong);
  font-size: clamp(2rem, 4vw, 2.5rem);
  font-weight: 680;
  letter-spacing: -0.04em;
  line-height: 1.1;
}

.page-description {
  margin-top: 10px;
  color: var(--color-text-secondary);
  line-height: 1.55;
}

.back-link {
  flex: 0 0 auto;
  padding: 8px 11px;
  border: 1px solid var(--color-border-muted);
  border-radius: 9px;
  color: var(--color-text-default);
  font-size: 0.88rem;
  text-decoration: none;
}

.back-link:hover {
  border-color: var(--color-text-supporting);
  background: var(--color-surface-quiet);
  color: var(--color-text-strong);
}

.account-section {
  margin-top: 20px;
  padding: 26px;
  border: 1px solid var(--account-border);
  border-radius: 16px;
  background: var(--color-surface-card);
}

.account-overview {
  display: grid;
  grid-template-columns: minmax(220px, 0.8fr) minmax(280px, 1.2fr);
  gap: 24px 36px;
  align-items: center;
}

.account-identity {
  display: flex;
  align-items: center;
  gap: 14px;
}

.account-avatar {
  display: grid;
  flex: 0 0 46px;
  width: 46px;
  height: 46px;
  place-items: center;
  border-radius: 12px;
  background: var(--color-text-strong);
  color: var(--color-surface-card);
  font-size: 1rem;
  font-weight: 700;
}

.account-identity h2 {
  margin-top: 3px;
  color: var(--color-text-strong);
  font-size: 1.25rem;
  font-weight: 650;
}

.usage-heading {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
}

.usage-heading strong {
  color: var(--color-text-default);
  font-size: 0.92rem;
  font-weight: 600;
}

.usage-track {
  height: 6px;
  margin-top: 10px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--account-progress-track);
}

.usage-track span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--color-text-strong);
  transition: width 0.2s ease;
}

.account-overview .product-status-banner {
  --product-status-banner-padding: 13px 14px;
  --product-status-banner-border: 1px solid #e3e3df;
  --product-status-banner-background: #fafaf9;
  --product-status-banner-radius: 10px;
  --product-status-banner-icon-margin: 1px 0 0;
  --product-status-banner-body-color: #52525b;

  grid-column: 1 / -1;
}

.section-heading {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 24px;
  margin-bottom: 24px;
}

.section-heading h2 {
  margin-top: 4px;
  color: var(--color-text-strong);
  font-size: 1.35rem;
  font-weight: 660;
  letter-spacing: -0.02em;
}

.section-heading p {
  color: var(--color-text-secondary);
  font-size: 0.88rem;
}

.account-form {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0 18px;
  max-width: 720px;
}

.account-field--wide {
  grid-column: 1 / -1;
}

.form-actions {
  display: flex;
  justify-content: flex-start;
  padding-top: 2px;
}

@media (--breakpoint-preview-down) {
  .account-page {
    padding: 28px 18px 64px;
  }

  .account-header {
    flex-direction: column;
    padding-bottom: 22px;
  }

  .account-section {
    padding: 20px;
    border-radius: 14px;
  }

  .account-overview,
  .account-form {
    grid-template-columns: 1fr;
  }

  .account-field--wide {
    grid-column: auto;
  }

  .section-heading {
    align-items: flex-start;
    flex-direction: column;
    gap: 8px;
  }

  .usage-heading {
    align-items: flex-start;
    flex-direction: column;
    gap: 4px;
  }

  .form-actions .ui-button {
    width: 100%;
  }
}
</style>
