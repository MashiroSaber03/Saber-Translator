export interface PluginData {
  id: string
  display_name: string
  description: string
  version: string
  author?: string
  enabled: boolean
  default_enabled: boolean
  has_config: boolean
  supported_steps: string[]
  supported_modes: string[]
  priority?: number
  failure_policy?: string
  configSchema?: Record<string, unknown>
  config?: Record<string, unknown>
}
