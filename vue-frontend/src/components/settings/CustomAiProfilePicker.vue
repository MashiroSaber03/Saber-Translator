<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'

import ProductActionRow from '@/components/product/ProductActionRow.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import type { UiSelectValue } from '@/components/ui/selectTypes'
import { useCustomAiProfileStore } from '@/stores/customAiProfileStore'
import type { CustomAiProfile, CustomAiProfileKind } from '@/types/customAiProfile'
import CustomAiProfileManager from './CustomAiProfileManager.vue'

const props = defineProps<{
  inputId: string
  kind: CustomAiProfileKind
  apiKey: string
  baseUrl: string
  model: string
  disabled?: boolean
}>()

const emit = defineEmits<{
  apply: [profile: CustomAiProfile]
}>()

const store = useCustomAiProfileStore()
const selectedId = ref('')
const managerOpen = ref(false)

const profiles = computed(() => store.byKind(props.kind))
const options = computed(() => [
  { value: '', label: profiles.value.length ? '选择已保存配置' : '暂无已保存配置' },
  ...profiles.value.map(profile => ({ value: profile.id, label: profile.name })),
])

function matchingProfile(): CustomAiProfile | undefined {
  return profiles.value.find(profile => (
    profile.apiKey === props.apiKey
    && profile.baseUrl === props.baseUrl.replace(/\/+$/, '')
    && profile.model === props.model
  ))
}

function apply(profile: CustomAiProfile): void {
  selectedId.value = profile.id
  emit('apply', profile)
}

function select(value: UiSelectValue): void {
  if (typeof value !== 'string') return
  selectedId.value = value
  const profile = profiles.value.find(item => item.id === value)
  if (profile) apply(profile)
}

onMounted(async () => {
  if (await store.load()) selectedId.value = matchingProfile()?.id ?? ''
})

watch(
  [
    () => props.kind,
    () => props.apiKey,
    () => props.baseUrl,
    () => props.model,
    profiles,
  ],
  () => {
    selectedId.value = matchingProfile()?.id ?? ''
  },
)
</script>

<template>
  <UiField
    class="custom-ai-profile-picker"
    variant="settings"
    label="已保存的自定义服务"
    :control-id="inputId"
    hint="选择后会应用保存的 Base URL、API Key 和模型名"
  >
    <div class="custom-ai-profile-picker__controls">
      <UiSelect
        :id="inputId"
        :model-value="selectedId"
        :options="options"
        :disabled="disabled || !store.isLoaded"
        @change="select"
      />
      <ProductActionRow justify="start" aria-label="自定义服务配置操作">
        <UiButton
          type="button"
          size="sm"
          variant="secondary"
          :disabled="disabled"
          @click="managerOpen = true"
        >
          管理配置
        </UiButton>
      </ProductActionRow>
    </div>
  </UiField>

  <CustomAiProfileManager
    v-if="managerOpen"
    v-model="managerOpen"
    :initial-kind="kind"
  />
</template>

<style scoped>
.custom-ai-profile-picker {
  margin-top: 14px;
}

.custom-ai-profile-picker__controls {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 10px;
}

@media (--breakpoint-sm-down) {
  .custom-ai-profile-picker__controls {
    grid-template-columns: 1fr;
  }
}
</style>
