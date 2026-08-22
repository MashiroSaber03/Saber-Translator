import { useAuthStore } from '@/stores/authStore'
import { useRuntimeStore } from '@/stores/runtimeStore'
import type { PublicFeatureKey, PublicModelKey } from '@/api/v2/auth'
import type { UiSelectOption } from '@/components/ui/selectTypes'

export function usePublicUserAccess() {
  const auth = useAuthStore()
  const runtime = useRuntimeStore()

  function restricted(): boolean {
    return runtime.capabilities?.profile === 'public' && !auth.isAdmin
  }

  function featureAllowed(feature: PublicFeatureKey): boolean {
    return !restricted() || runtime.capabilities?.publicUserPolicy.features[feature] !== false
  }

  function modelAllowed(model: PublicModelKey): boolean {
    return !restricted() || runtime.capabilities?.publicUserPolicy.models[model] !== false
  }

  function modelOptions(
    options: UiSelectOption[],
    modelByValue: Partial<Record<string, PublicModelKey>>,
  ): UiSelectOption[] {
    return options.map(option => {
      const model = modelByValue[String(option.value)]
      if (!model || modelAllowed(model)) return option
      return {
        ...option,
        label: `${option.label}（管理员已关闭）`,
        disabled: true,
      }
    })
  }

  function lamaDisableResizeEditable(): boolean {
    return !restricted()
      || runtime.capabilities?.publicUserPolicy.settings.lamaDisableResize.editable !== false
  }

  function lamaDisableResizeValue(): boolean {
    return runtime.capabilities?.publicUserPolicy.settings.lamaDisableResize.value ?? false
  }

  function parallelAllowed(): boolean {
    return !restricted()
      || runtime.capabilities?.publicUserPolicy.settings.parallel.allowed !== false
  }

  function maxDeepLearningConcurrency(): number | null {
    if (!restricted()) return null
    return runtime.capabilities?.publicUserPolicy.settings.parallel.maxDeepLearningConcurrency ?? 1
  }

  return {
    featureAllowed,
    lamaDisableResizeEditable,
    lamaDisableResizeValue,
    maxDeepLearningConcurrency,
    modelAllowed,
    modelOptions,
    parallelAllowed,
    restricted,
  }
}
