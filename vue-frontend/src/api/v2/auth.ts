import { apiClient } from '@/api/client'

export type PublicFeatureKey = 'translation' | 'insight' | 'characterStudio' | 'editMode'
export type PublicModelKey =
  | 'detector_default'
  | 'detector_ctd'
  | 'detector_yolo'
  | 'aux_ysg_yolo'
  | 'saber_yolo'
  | 'manga_ocr'
  | 'ocr_48px'
  | 'paddle_ocr'
  | 'paddleocr_vl'
  | 'lama_mpe'
  | 'litelama'

export interface PublicUserPolicy {
  features: Record<PublicFeatureKey, boolean>
  models: Record<PublicModelKey, boolean>
  settings: {
    lamaDisableResize: {
      editable: boolean
      value: boolean
    }
    parallel: {
      allowed: boolean
    }
  }
}

export type QueueDiscipline = 'owner_round_robin' | 'fifo'

export interface SchedulingPolicy {
  queueDiscipline: QueueDiscipline
  pageQuantum: number
  interactiveBurst: number
  maxDeepLearningConcurrency: number
  apiOperationConcurrency: number
  modelIdleSeconds: number
  minAvailableMemoryMiB: number
}

export interface SchedulingStatus {
  workerOnline: boolean
  currentTask: {
    jobId: string
    kind: string
    status: string
    ownerUserId: string
    ownerUsername: string
    startedAt: string | null
  } | null
  queuedJobCount: number
  queuedUserCount: number
  pausedJobCount: number
  availableMemoryMiB: number
  totalMemoryMiB: number
  waitingReason: 'worker_offline' | 'low_memory' | 'queue_blocked' | null
}

export interface SchedulingOverview {
  policy: SchedulingPolicy
  status: SchedulingStatus
}

export interface RuntimeCapabilities {
  profile: 'local' | 'public'
  requiresAuth: boolean
  browserCredentials: boolean
  registrationRequiresInvite: boolean
  publicUserPolicy: PublicUserPolicy
  scheduling: {
    maxDeepLearningConcurrency: number
  }
  features: {
    plugins: boolean
    webImport: boolean
    localProviders: boolean
  }
}

export interface AuthUser {
  id: string
  username: string
  role: 'user' | 'admin'
}

export interface AuthSession {
  user: AuthUser
  csrfToken?: string
  assetUsageBytes: number
  assetQuotaBytes: number
  recoveryCodes?: string[]
}

export interface AdminUser extends AuthUser {
  status: 'active' | 'disabled'
  assetUsageBytes: number
  assetQuotaBytes: number
  createdAt: string
  taskStatus: 'active' | 'queued' | 'paused' | 'interrupted' | 'idle'
  activeTaskCount: number
  queuedTaskCount: number
  pausedTaskCount: number
  interruptedTaskCount: number
  completedTaskCount: number
  issueTaskCount: number
  currentTaskKind: string | null
  currentTaskStartedAt: string | null
  lastTaskAt: string | null
}

export interface AdminInvite {
  id: string
  status: 'active' | 'used' | 'expired' | 'revoked'
  expiresAt: string
  usedAt: string | null
  usedBy: string | null
  createdAt: string
}

export function getCapabilities(): Promise<RuntimeCapabilities> {
  return apiClient.get('/api/v2/system/capabilities')
}

export function getCurrentSession(): Promise<AuthSession> {
  return apiClient.get('/api/v2/auth/me')
}

export function login(username: string, password: string): Promise<AuthSession> {
  return apiClient.post('/api/v2/auth/login', { username, password })
}

export function register(
  username: string,
  password: string,
  inviteCode?: string
): Promise<AuthSession> {
  return apiClient.post('/api/v2/auth/register', {
    username,
    password,
    ...(inviteCode ? { inviteCode } : {}),
  })
}

export function recoverPassword(
  username: string,
  recoveryCode: string,
  newPassword: string
): Promise<{ status: string }> {
  return apiClient.post('/api/v2/auth/recover', { username, recoveryCode, newPassword })
}

export function logout(): Promise<{ status: string }> {
  return apiClient.post('/api/v2/auth/logout')
}

export function changePassword(
  currentPassword: string,
  newPassword: string
): Promise<{ status: string }> {
  return apiClient.post('/api/v2/auth/change-password', { currentPassword, newPassword })
}

export async function listAdminUsers(): Promise<AdminUser[]> {
  return (await apiClient.get<{ users: AdminUser[] }>('/api/v2/admin/users')).users
}

export async function listAdminInvites(): Promise<AdminInvite[]> {
  return (await apiClient.get<{ invites: AdminInvite[] }>('/api/v2/admin/invites')).invites
}

export function createAdminInvite(): Promise<{ code: string; expiresAt: string }> {
  return apiClient.post('/api/v2/admin/invites')
}

export function revokeAdminInvite(inviteId: string): Promise<{ status: string }> {
  return apiClient.delete(`/api/v2/admin/invites/${encodeURIComponent(inviteId)}`)
}

export function setAdminUserStatus(
  userId: string,
  status: 'active' | 'disabled'
): Promise<{ status: string }> {
  return apiClient.patch(`/api/v2/admin/users/${encodeURIComponent(userId)}/status`, { status })
}

export function getAssetQuota(): Promise<{ assetQuotaBytes: number }> {
  return apiClient.get('/api/v2/admin/asset-quota')
}

export function setAssetQuota(assetQuotaBytes: number): Promise<{ assetQuotaBytes: number }> {
  return apiClient.patch('/api/v2/admin/asset-quota', { assetQuotaBytes })
}

export function getRegistrationPolicy(): Promise<{ registrationRequiresInvite: boolean }> {
  return apiClient.get('/api/v2/admin/registration-policy')
}

export function setRegistrationPolicy(
  registrationRequiresInvite: boolean
): Promise<{ registrationRequiresInvite: boolean }> {
  return apiClient.patch('/api/v2/admin/registration-policy', {
    registrationRequiresInvite,
  })
}

export function getPublicUserPolicy(): Promise<PublicUserPolicy> {
  return apiClient.get('/api/v2/admin/public-user-policy')
}

export function setPublicUserPolicy(policy: PublicUserPolicy): Promise<PublicUserPolicy> {
  return apiClient.patch('/api/v2/admin/public-user-policy', policy)
}

export function getSchedulingPolicy(): Promise<SchedulingOverview> {
  return apiClient.get('/api/v2/admin/scheduling-policy')
}

export function setSchedulingPolicy(policy: SchedulingPolicy): Promise<SchedulingOverview> {
  return apiClient.patch('/api/v2/admin/scheduling-policy', policy)
}

export function createUserRecoveryCode(userId: string): Promise<{ recoveryCode: string }> {
  return apiClient.post(`/api/v2/admin/users/${encodeURIComponent(userId)}/recovery-code`)
}
