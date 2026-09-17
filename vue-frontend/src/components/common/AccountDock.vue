<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/authStore'
import { useRuntimeStore } from '@/stores/runtimeStore'
import BaseModal from '@/components/common/BaseModal.vue'
import PublicTrialNotice from '@/components/common/PublicTrialNotice.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { showToast } from '@/utils/toast'

const auth = useAuthStore()
const runtime = useRuntimeStore()
const router = useRouter()
const showTrialNotice = ref(false)
const isLoggingOut = ref(false)
const isPublicProfile = computed(() => runtime.capabilities?.profile === 'public')

async function logout(): Promise<void> {
  if (isLoggingOut.value) return
  isLoggingOut.value = true
  showTrialNotice.value = false
  try {
    // Let the current page's leave guard save while the session is still valid.
    if (router.currentRoute.value.name !== 'bookshelf') {
      const failure = await router.replace({ name: 'bookshelf' })
      if (failure) return
    }
    await auth.logout()
    await router.replace('/login')
  } catch (error) {
    showToast(`退出失败：${error instanceof Error ? error.message : '请重试'}`, 'error')
  } finally {
    isLoggingOut.value = false
  }
}
</script>

<template>
  <aside v-if="auth.user" class="account-dock" aria-label="当前账号">
    <span>{{ auth.user.username }}</span>
    <RouterLink to="/account">账户</RouterLink>
    <RouterLink v-if="auth.isAdmin" to="/admin">管理</RouterLink>
    <UiButton
      v-if="isPublicProfile"
      variant="link"
      size="xs"
      type="button"
      aria-haspopup="dialog"
      @click="showTrialNotice = true"
    >
      试用说明
    </UiButton>
    <UiButton variant="link" size="xs" type="button" :loading="isLoggingOut" @click="logout">
      {{ isLoggingOut ? '正在退出…' : '退出' }}
    </UiButton>
  </aside>
  <BaseModal
    v-if="isPublicProfile"
    v-model="showTrialNotice"
    title="关于在线试用版"
    size="small"
    frame-variant="outlined"
    divider-variant="soft"
    body-padding="compact"
  >
    <PublicTrialNotice class="account-dock__trial-notice" />
  </BaseModal>
</template>

<style scoped>
.account-dock { position: sticky; z-index: var(--z-app-header); bottom: 12px; display: flex; align-items: center; justify-content: flex-end; flex-wrap: wrap; gap: 9px; width: fit-content; max-width: calc(100% - 24px); margin: 0 12px 12px auto; padding: 7px 9px; border: 1px solid var(--color-overlay-inverse-muted); border-radius: 999px; background: color-mix(in srgb, var(--color-surface-inverse) 90%, transparent); color: var(--color-text-inverse); box-shadow: 0 8px 25px var(--color-overlay-scrim-subtle); font-size: .82rem; backdrop-filter: blur(10px); }
.account-dock a, .account-dock .ui-button { color: var(--color-text-brand); }

.account-dock__trial-notice {
  --product-status-banner-border: 0;
  --product-status-banner-padding: 4px;

  box-shadow: none;
}
</style>
