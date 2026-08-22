import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import {
  getCurrentSession,
  login as loginRequest,
  logout as logoutRequest,
  register as registerRequest,
  type AuthSession,
  type AuthUser,
} from '@/api/v2/auth'
import { configureBrowserCredentials } from '@/services/browserCredentials'
import { useRuntimeStore } from './runtimeStore'

export const useAuthStore = defineStore('auth', () => {
  const runtime = useRuntimeStore()
  const user = ref<AuthUser | null>(null)
  const assetUsageBytes = ref(0)
  const assetQuotaBytes = ref(0)
  const restored = ref(false)
  let restorePromise: Promise<boolean> | null = null

  const authenticated = computed(() => Boolean(user.value))
  const isAdmin = computed(() => user.value?.role === 'admin')

  function applySession(session: AuthSession): void {
    user.value = session.user
    assetUsageBytes.value = session.assetUsageBytes
    assetQuotaBytes.value = session.assetQuotaBytes
    configureBrowserCredentials(
      Boolean(runtime.capabilities?.browserCredentials),
      session.user.id,
    )
  }

  async function restore(): Promise<boolean> {
    if (restored.value) return authenticated.value
    if (restorePromise) return restorePromise
    restorePromise = (async () => {
      await runtime.load()
      if (!runtime.capabilities?.requiresAuth) {
        restored.value = true
        configureBrowserCredentials(false)
        return true
      }
      try {
        applySession(await getCurrentSession())
        return true
      } catch {
        user.value = null
        configureBrowserCredentials(false)
        return false
      } finally {
        restored.value = true
      }
    })().finally(() => { restorePromise = null })
    return restorePromise
  }

  async function login(username: string, password: string): Promise<void> {
    await runtime.load()
    applySession(await loginRequest(username, password))
    restored.value = true
  }

  async function refresh(): Promise<void> {
    await runtime.load()
    if (!runtime.capabilities?.requiresAuth) return
    applySession(await getCurrentSession())
    restored.value = true
  }

  async function register(
    username: string,
    password: string,
    inviteCode?: string,
  ): Promise<string[]> {
    await runtime.load()
    const session = await registerRequest(username, password, inviteCode)
    applySession(session)
    restored.value = true
    return session.recoveryCodes ?? []
  }

  async function logout(): Promise<void> {
    try {
      await logoutRequest()
    } finally {
      user.value = null
      restored.value = true
      configureBrowserCredentials(false)
    }
  }

  function markUnauthenticated(): void {
    if (!runtime.capabilities?.requiresAuth) return
    user.value = null
    assetUsageBytes.value = 0
    assetQuotaBytes.value = 0
    restored.value = true
    configureBrowserCredentials(false)
  }

  return {
    user,
    assetUsageBytes,
    assetQuotaBytes,
    restored,
    authenticated,
    isAdmin,
    restore,
    refresh,
    login,
    register,
    logout,
    markUnauthenticated,
  }
})
