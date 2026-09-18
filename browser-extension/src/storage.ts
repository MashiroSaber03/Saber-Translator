import type { DomainPreference, ExtensionSettings } from './types'

export const STORAGE_KEY = 'saber-extension-settings-v1'

export const DEFAULT_PREFERENCE: DomainPreference = {
  disabled: false,
  method: 'adapter',
  mode: 'standard',
  glossaryEnabled: false,
  autoTermsEnabled: false,
}

export const DEFAULT_SETTINGS: ExtensionSettings = {
  token: '',
  serverPort: 5000,
  domains: {},
}

export async function loadSettings(): Promise<ExtensionSettings> {
  const stored = await chrome.storage.local.get(STORAGE_KEY)
  const value = stored[STORAGE_KEY] as Partial<ExtensionSettings> | undefined
  if (value === undefined) {
    return structuredClone(DEFAULT_SETTINGS)
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)
    || typeof value.token !== 'string'
    || typeof value.serverPort !== 'number'
    || !Number.isInteger(value.serverPort) || value.serverPort < 1 || value.serverPort > 65535
    || !value.domains || typeof value.domains !== 'object' || Array.isArray(value.domains)) {
    throw new Error('扩展设置格式无效')
  }
  return value as ExtensionSettings
}

let storageWrite: Promise<unknown> = Promise.resolve()

export function serializeStorageWrite<T>(operation: () => Promise<T>): Promise<T> {
  const pending = storageWrite.then(operation)
  storageWrite = pending.catch(() => undefined)
  return pending
}

export function updateSettings(change: (settings: ExtensionSettings) => void): Promise<void> {
  return serializeStorageWrite(async () => {
    const settings = await loadSettings()
    change(settings)
    await chrome.storage.local.set({ [STORAGE_KEY]: settings })
  })
}

export function preferenceFor(
  settings: ExtensionSettings,
  hostname: string,
): DomainPreference {
  const preference = settings.domains[hostname]
  if (preference === undefined) return structuredClone(DEFAULT_PREFERENCE)
  if (!preference || typeof preference !== 'object'
    || typeof preference.disabled !== 'boolean'
    || typeof preference.glossaryEnabled !== 'boolean'
    || typeof preference.autoTermsEnabled !== 'boolean'
    || !['adapter', 'dom-agent', 'similar'].includes(preference.method)
    || !['standard', 'hq'].includes(preference.mode)) {
    throw new Error('站点设置格式无效')
  }
  const result = structuredClone(preference)
  const position = result.fabPosition
  // Old pixel coordinates cannot be restored across different viewport sizes.
  if (position && (!['left', 'right'].includes(position.side)
    || !Number.isFinite(position.yRatio) || position.yRatio < 0 || position.yRatio > 1)) {
    delete result.fabPosition
  }
  return result
}
