import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'

export type LocalModelCatalog = components['schemas']['LocalModelCatalog']
export type LocalModelInstallRequest = components['schemas']['LocalModelInstallRequest']
export type LocalModelInstallTask = components['schemas']['LocalModelInstallTask']

export function getLocalModels(signal?: AbortSignal): Promise<LocalModelCatalog> {
  return apiClient.get('/api/v2/system/models', { signal })
}

export function installLocalModels(request: LocalModelInstallRequest): Promise<LocalModelInstallTask> {
  return apiClient.post('/api/v2/system/models/install', request)
}

export function cancelLocalModelInstall(): Promise<LocalModelInstallTask> {
  return apiClient.post('/api/v2/system/models/cancel')
}
