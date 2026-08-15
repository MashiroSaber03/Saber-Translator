import { getProviderOptionsForCapability } from '@/config/aiProviders'

export const WEB_IMPORT_AGENT_PROVIDERS = getProviderOptionsForCapability('webImportAgent') as ReadonlyArray<{
  value: string
  label: string
}>
