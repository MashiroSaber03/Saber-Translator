import { defineStore } from 'pinia'
import { ref } from 'vue'
import { getCapabilities, type PublicUserPolicy, type RuntimeCapabilities } from '@/api/v2/auth'
import { configureLocalProviderVisibility } from '@/config/aiProviders'

const LOCAL_CAPABILITIES: RuntimeCapabilities = {
  installedModels: [],
  profile: 'local',
  requiresAuth: false,
  browserCredentials: false,
  registrationRequiresInvite: false,
  publicUserPolicy: {
    features: {
      translation: true,
      insight: true,
      characterStudio: true,
      editMode: true,
    },
    models: {
      detector_default: true,
      detector_ctd: true,
      detector_yolo: true,
      aux_ysg_yolo: true,
      saber_yolo: true,
      manga_ocr: true,
      ocr_48px: true,
      paddle_ocr: true,
      paddleocr_vl: true,
      lama_mpe: true,
      litelama: true,
      lama_manga: true,
    },
    settings: {
      lamaDisableResize: { editable: true, value: false },
      parallel: { allowed: true },
    },
  },
  scheduling: { maxDeepLearningConcurrency: null },
  features: { plugins: true, webImport: true, localProviders: true },
}

export const useRuntimeStore = defineStore('runtime', () => {
  const capabilities = ref<RuntimeCapabilities | null>(null)
  let loading: Promise<RuntimeCapabilities> | null = null

  async function load(refresh = false): Promise<RuntimeCapabilities> {
    if (!refresh && capabilities.value) return capabilities.value
    if (loading) return loading
    loading = getCapabilities()
      .then(value => {
        configureLocalProviderVisibility(value.features.localProviders)
        if (JSON.stringify(capabilities.value) !== JSON.stringify(value)) capabilities.value = value
        return value
      })
      .finally(() => {
        loading = null
      })
    return loading
  }

  function assumeLocalForTests(): void {
    configureLocalProviderVisibility(true)
    capabilities.value = structuredClone(LOCAL_CAPABILITIES)
    capabilities.value.installedModels = Object.keys(LOCAL_CAPABILITIES.publicUserPolicy.models)
      .filter(model => model !== 'aux_ysg_yolo')
  }

  function setRegistrationRequiresInvite(value: boolean): void {
    if (!capabilities.value) return
    capabilities.value = {
      ...capabilities.value,
      registrationRequiresInvite: value,
    }
  }

  function setPublicUserPolicy(value: PublicUserPolicy): void {
    if (!capabilities.value) return
    capabilities.value = {
      ...capabilities.value,
      publicUserPolicy: value,
    }
  }

  function setMaxDeepLearningConcurrency(value: number): void {
    if (!capabilities.value) return
    capabilities.value = {
      ...capabilities.value,
      scheduling: { maxDeepLearningConcurrency: value },
    }
  }

  return {
    capabilities,
    load,
    assumeLocalForTests,
    setRegistrationRequiresInvite,
    setPublicUserPolicy,
    setMaxDeepLearningConcurrency,
  }
})
