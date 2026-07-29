import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'

export type V2ServerInfo = components['schemas']['ServerInfo']
export type V2WorkerCommandAccepted = components['schemas']['WorkerCommandAccepted']

export function getV2ServerInfo(): Promise<V2ServerInfo> {
  return apiClient.get<V2ServerInfo>('/api/v2/system/server-info')
}

export function releaseWorkerModelCache(): Promise<V2WorkerCommandAccepted> {
  return apiClient.post<V2WorkerCommandAccepted>(
    '/api/v2/system/release-models',
  )
}
