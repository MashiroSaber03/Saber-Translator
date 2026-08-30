import type { DomainPreference, ExtensionSettings } from './types'

const STORAGE_KEY = 'saber-extension-settings-v1'

export const DEFAULT_PREFERENCE: DomainPreference = {
  disabled: false,
  method: 'adapter',
  mode: 'standard',
  glossaryEnabled: false,
  autoTermsEnabled: false,
  panelOpen: false,
}

export const DEFAULT_SETTINGS: ExtensionSettings = {
  token: '',
  serverPort: 5000,
  domains: {},
}

export async function loadSettings(): Promise<ExtensionSettings> {
  const stored = await chrome.storage.local.get(STORAGE_KEY)
  const value = stored[STORAGE_KEY]
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return structuredClone(DEFAULT_SETTINGS)
  }
  const candidate = value as Partial<ExtensionSettings>
  return {
    token: typeof candidate.token === 'string' ? candidate.token : '',
    serverPort: Number.isInteger(candidate.serverPort)
      && Number(candidate.serverPort) >= 1
      && Number(candidate.serverPort) <= 65535
      ? Number(candidate.serverPort)
      : 5000,
    domains: candidate.domains && typeof candidate.domains === 'object'
      ? candidate.domains
      : {},
  }
}

export async function saveSettings(settings: ExtensionSettings): Promise<void> {
  await chrome.storage.local.set({ [STORAGE_KEY]: settings })
}

export function preferenceFor(
  settings: ExtensionSettings,
  hostname: string,
): DomainPreference {
  return {
    ...DEFAULT_PREFERENCE,
    ...(settings.domains[hostname] ?? {}),
  }
}
