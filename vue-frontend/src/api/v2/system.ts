import { apiClient } from '@/api/client'

export interface V2ServerInfo {
  host: string
  hostname: string
  lanUrl: string
  port: number
}

export function getV2ServerInfo(): Promise<V2ServerInfo> {
  return apiClient.get<V2ServerInfo>('/api/v2/system/server-info')
}
