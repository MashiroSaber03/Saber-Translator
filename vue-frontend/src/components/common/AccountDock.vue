<script setup lang="ts">
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/authStore'
import UiButton from '@/components/ui/UiButton.vue'

const auth = useAuthStore()
const router = useRouter()

async function logout(): Promise<void> {
  await auth.logout()
  await router.replace('/login')
}
</script>

<template>
  <aside v-if="auth.user" class="account-dock" aria-label="当前账号">
    <span>{{ auth.user.username }}</span>
    <RouterLink to="/account">账户</RouterLink>
    <RouterLink v-if="auth.isAdmin" to="/admin">管理</RouterLink>
    <UiButton variant="link" size="xs" type="button" @click="logout">退出</UiButton>
  </aside>
</template>

<style scoped>
.account-dock { position: sticky; z-index: var(--z-app-header); bottom: 12px; display: flex; align-items: center; gap: 9px; width: fit-content; margin: 0 12px 12px auto; padding: 7px 9px; border: 1px solid var(--color-overlay-inverse-muted); border-radius: 999px; background: color-mix(in srgb, var(--color-surface-inverse) 90%, transparent); color: var(--color-text-inverse); box-shadow: 0 8px 25px var(--color-overlay-scrim-subtle); font-size: .82rem; backdrop-filter: blur(10px); }
.account-dock a, .account-dock .ui-button { color: var(--color-text-brand); }
</style>
