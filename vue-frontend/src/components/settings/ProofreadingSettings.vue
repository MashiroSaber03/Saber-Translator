<template>
  <div class="proofreading-settings">
    <ProductFormSection>
      <template #title>AI校对设置</template>
      <UiField variant="settings" control="checkbox" hint="翻译完成后自动进行AI校对">
        <UiCheckbox
          :model-value="isProofreadingEnabled"
          label="启用AI校对"
          @update:model-value="settingsStore.setProofreadingEnabled"
        />
      </UiField>
    </ProductFormSection>

    <ProductFormSection v-show="isProofreadingEnabled">
      <template #title>
        校对轮次配置
        <UiButton
          variant="secondary"
          class="proofreading-settings__add-round-action"
          @click="addRound"
          size="sm"
        >
          <UiIcon name="plus" />
          <span>添加轮次</span>
        </UiButton>
      </template>

      <div
        v-for="(round, index) in proofreadingRounds"
        :key="round.id"
        class="proofreading-settings__round"
      >
        <div class="proofreading-settings__round-header">
          <span class="proofreading-settings__round-title">轮次 {{ index + 1 }}: {{ round.name || '未命名' }}</span>
          <UiButton
            variant="danger"
            class="proofreading-settings__round-delete-action"
            @click="removeRound(index)"
            :disabled="proofreadingRounds.length <= 1"
            size="sm"
          >
            删除
          </UiButton>
        </div>

        <div class="proofreading-settings__round-content">
          <UiField variant="settings" label="轮次名称" :control-id="roundFieldId(round.id, 'Name')">
            <UiInput
              type="text"
              :id="roundFieldId(round.id, 'Name')"
              :model-value="round.name"
              placeholder="如: 第一轮校对"
              @update:model-value="updateRoundString(index, 'name', $event)"
            />
          </UiField>

          <UiFormGrid>
            <AiProviderSelectField
              :model-value="round.provider"
              :input-id="roundFieldId(round.id, 'Provider')"
              :options="providerOptions"
              custom-profile-kind="chatVision"
              :custom-profile-api-key="round.apiKey"
              :custom-profile-base-url="round.customBaseUrl"
              :custom-profile-model="round.modelName"
              @change="handleRoundProviderChange(index, round, $event)"
              @apply-custom-profile="applyCustomProfile(index, $event)"
            />
            <AiProviderCredentialFields
              :api-key="round.apiKey"
              :api-key-input-id="roundFieldId(round.id, 'ApiKey')"
              :base-url="round.customBaseUrl"
              :base-url-input-id="roundFieldId(round.id, 'BaseUrl')"
              :show-api-key="providerRequiresApiKey(round.provider)"
              :show-base-url="false"
              :include-base-url="false"
              api-key-placeholder="请输入API Key"
              :api-key-show-label="`显示${round.name} API Key`"
              :api-key-hide-label="`隐藏${round.name} API Key`"
              @update:api-key="updateRoundString(index, 'apiKey', $event)"
            />
          </UiFormGrid>

          <AiProviderCredentialFields
            :api-key="round.apiKey"
            :api-key-input-id="roundFieldId(round.id, 'ApiKey')"
            :base-url="round.customBaseUrl"
            :base-url-input-id="roundFieldId(round.id, 'BaseUrl')"
            :show-api-key="false"
            :show-base-url="providerRequiresBaseUrl(round.provider)"
            :include-api-key="false"
            base-url-placeholder="例如: https://api.example.com/v1"
            @update:base-url="updateRoundString(index, 'customBaseUrl', $event)"
          />

          <UiField
            variant="settings"
            label="模型名称"
            :control-id="roundFieldId(round.id, 'ModelName')"
          >
            <UiModelPicker
              :input-id="roundFieldId(round.id, 'ModelName')"
              :model-value="round.modelName"
              placeholder="请输入模型名称"
              fetch-variant="primary"
              :fetching="isRoundFetching(round)"
              :fetch-disabled="isRoundFetching(round)"
              :options="getRoundModelOptions(round)"
              :model-count="getRoundModelCount(round)"
              @update:model-value="updateRoundModel(index, $event)"
              @fetch="fetchRoundModels(round)"
            />
          </UiField>

          <UiField variant="settings">
            <UiButton
              variant="secondary"
              tone="info"
              block
              @click="testRoundConnection(round)"
              :disabled="roundTestingStates[round.id]"
            >
              <span v-if="roundTestingStates[round.id]">测试中...</span>
              <template v-else>
                <span aria-hidden="true">🔗</span>
                <span>测试连接</span>
              </template>
            </UiButton>
          </UiField>

          <UiFormGrid>
            <UiField
              variant="settings"
              label="批次大小"
              :control-id="roundFieldId(round.id, 'BatchSize')"
            >
              <UiNumberField
                :input-id="roundFieldId(round.id, 'BatchSize')"
                :model-value="round.batchSize"
                :min="1"
                :step="1"
                @update:model-value="updateRoundNumber(index, 'batchSize', $event)"
              />
            </UiField>
            <UiField
              variant="settings"
              label="RPM限制"
              :control-id="roundFieldId(round.id, 'RpmLimit')"
            >
              <UiNumberField
                :input-id="roundFieldId(round.id, 'RpmLimit')"
                :model-value="round.openaiOptions.execution.rpmLimit"
                :min="0"
                :max="100000"
                :step="1"
                @update:model-value="updateRoundNumber(index, 'rpmLimit', $event)"
              />
            </UiField>
          </UiFormGrid>

          <UiFormGrid>
            <UiField
              variant="settings"
              label="业务重试"
              :control-id="roundFieldId(round.id, 'BusinessRetries')"
            >
              <UiNumberField
                :input-id="roundFieldId(round.id, 'BusinessRetries')"
                :model-value="round.openaiOptions.execution.businessRetries"
                :min="0"
                :max="100"
                :step="1"
                @update:model-value="updateRoundNumber(index, 'businessRetries', $event)"
              />
            </UiField>
            <UiField
              variant="settings"
              label="传输重试"
              :control-id="roundFieldId(round.id, 'TransportRetries')"
            >
              <UiNumberField
                :input-id="roundFieldId(round.id, 'TransportRetries')"
                :model-value="round.openaiOptions.execution.transportRetries"
                :min="0"
                :max="100"
                :step="1"
                @update:model-value="updateRoundNumber(index, 'transportRetries', $event)"
              />
            </UiField>
          </UiFormGrid>
          <UiFormGrid>
            <UiField variant="settings" control="checkbox" hint="使用 response_format: json_object">
              <UiCheckbox
                :model-value="round.openaiOptions.request.forceJsonOutput"
                label="强制JSON输出"
                @update:model-value="updateRoundBoolean(index, 'forceJsonOutput', $event)"
              />
            </UiField>
            <UiField variant="settings" control="checkbox" hint="使用流式API调用，避免超时">
              <UiCheckbox
                :model-value="round.openaiOptions.execution.useStream"
                label="流式调用"
                @update:model-value="updateRoundBoolean(index, 'useStream', $event)"
              />
            </UiField>
          </UiFormGrid>
          <UiField variant="settings">
            <OpenAIExtraBodyEditor
              :model-value="round.openaiOptions.request.extraBody"
              @update:model-value="updateRoundExtraBody(index, $event)"
            />
          </UiField>

          <UiField
            variant="settings"
            label="校对偏好"
            :control-id="roundFieldId(round.id, 'Prompt')"
            hint="只描述校对目标与语言风格，输出格式由后端管理"
          >
            <UiTextarea
              :id="roundFieldId(round.id, 'Prompt')"
              :model-value="round.prompt"
              variant="panel"
              rows="4"
              placeholder="请输入校对目标与语言风格要求"
              @update:model-value="updateRoundString(index, 'prompt', $event)"
            />
            <SavedPromptsPicker
              prompt-type="proofreading"
              @select="(content, name) => handleProofreadingPromptSelect(index, content, name)"
            />
            <ProductActionRow aria-label="校对提示词操作" justify="start">
              <UiButton variant="secondary" @click="resetRoundPrompt(index)" size="sm">
                重置为默认
              </UiButton>
            </ProductActionRow>
          </UiField>
        </div>
      </div>
    </ProductFormSection>
  </div>
</template>

<script setup lang="ts">
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import ProductFormSection from '@/components/product/ProductFormSection.vue'
import UiTextarea from '@/components/ui/UiTextarea.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiModelPicker from '@/components/ui/UiModelPicker.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import AiProviderCredentialFields from '@/components/settings/AiProviderCredentialFields.vue'
import AiProviderSelectField from '@/components/settings/AiProviderSelectField.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import type { CustomAiProfile } from '@/types/customAiProfile'
import { computed, onBeforeUnmount, ref } from 'vue'
import {
  getProviderOptionsForCapability,
  providerRequiresApiKey,
  providerRequiresApiKeyForBaseUrl,
  providerRequiresBaseUrl,
} from '@/config/aiProviders'
import { useSettingsStore } from '@/stores/settings'
import { fetchModels as fetchV2Models, testAiTranslateConnection } from '@/api/v2/diagnostics'
import { useToast } from '@/utils/toast'
import {
  DEFAULT_HQ_TRANSLATION_MAX_RETRIES,
  DEFAULT_PROOFREADING_PROMPT,
} from '@/constants'
import type { ProofreadingRound } from '@/types/settings'
import {
  newProofreadingRoundId,
  proofreadingProviderDomain,
} from '@/stores/settings/proofreadingIdentity'
import OpenAIExtraBodyEditor from '@/components/common/OpenAIExtraBodyEditor.vue'
import SavedPromptsPicker from '@/components/settings/SavedPromptsPicker.vue'
import {
  useAiModelDiscovery,
  type AiModelDiscoveryMessageTone,
} from '@/composables/useAiModelDiscovery'

const providerOptions = getProviderOptionsForCapability('hqTranslation')

const settingsStore = useSettingsStore()
const toast = useToast()

const roundTestingStates = ref<Record<string, boolean>>({})
const roundModelDiscoveries = new Map<string, ReturnType<typeof useAiModelDiscovery>>()

const proofreadingRounds = computed(() => settingsStore.settings.proofreading.rounds)
const isProofreadingEnabled = computed(() => settingsStore.settings.proofreading.enabled)

function roundFieldId(roundId: string, field: string) {
  return `proofreadingRound-${roundId}-${field}`
}

function getRoundModelOptions(round: ProofreadingRound) {
  const options = [{ label: '-- 选择模型 --', value: '' }]
  getRoundModelDiscovery(round).models.value.forEach(model => {
    options.push({ label: model.id, value: model.id })
  })
  return options
}

function notifyRoundModelDiscovery(message: string, tone: AiModelDiscoveryMessageTone): void {
  toast[tone](message)
}

function getRoundModelDiscovery(round: ProofreadingRound): ReturnType<typeof useAiModelDiscovery> {
  const roundId = round.id
  const existing = roundModelDiscoveries.get(roundId)
  if (existing) return existing

  const discovery = useAiModelDiscovery({
    source: () => {
      const currentRound = proofreadingRounds.value.find(candidate => candidate.id === roundId)
      return {
        provider: currentRound?.provider ?? '',
        apiKey: currentRound?.apiKey ?? '',
        baseUrl: currentRound?.customBaseUrl ?? '',
      }
    },
    fetcher: (provider, apiKey, baseUrl) =>
      fetchV2Models(
        provider,
        apiKey,
        baseUrl,
        proofreadingProviderDomain(roundId),
      ),
    notify: notifyRoundModelDiscovery,
    successMessage: count => {
      const index = proofreadingRounds.value.findIndex(candidate => candidate.id === roundId)
      return index >= 0
        ? `轮次 ${index + 1}: 获取到 ${count} 个模型`
        : `校对轮次已获取到 ${count} 个模型`
    },
    emptyBaseUrl: '',
  })
  roundModelDiscoveries.set(roundId, discovery)
  return discovery
}

function isRoundFetching(round: ProofreadingRound): boolean {
  return getRoundModelDiscovery(round).isFetchingModels.value
}

function getRoundModelCount(round: ProofreadingRound): number {
  return getRoundModelDiscovery(round).models.value.length
}

function updateRoundString(
  index: number,
  field: 'name' | 'apiKey' | 'customBaseUrl' | 'prompt',
  value: string | number,
): void {
  settingsStore.updateProofreadingRound(index, { [field]: String(value) })
}

function applyCustomProfile(index: number, profile: CustomAiProfile): void {
  settingsStore.updateProofreadingRound(index, {
    apiKey: profile.apiKey,
    customBaseUrl: profile.baseUrl,
    modelName: profile.model,
  })
}

function updateRoundModel(index: number, value: string | number): void {
  if (typeof value !== 'string') return
  settingsStore.updateProofreadingRound(index, { modelName: value })
}

function updateRoundNumber(
  index: number,
  field: 'batchSize' | 'rpmLimit' | 'businessRetries' | 'transportRetries',
  value: number | null,
): void {
  if (value === null) return
  settingsStore.updateProofreadingRound(index, { [field]: value })
}

function updateRoundBoolean(
  index: number,
  field: 'forceJsonOutput' | 'useStream',
  value: boolean,
): void {
  settingsStore.updateProofreadingRound(index, { [field]: value })
}

function updateRoundExtraBody(
  index: number,
  value: Record<string, unknown> | undefined,
): void {
  settingsStore.updateProofreadingRound(index, { extraBody: value })
}

function handleRoundProviderChange(index: number, round: ProofreadingRound, value: string) {
  if (!providerOptions.some(option => option.value === value)) return
  if (round.provider === value) return
  getRoundModelDiscovery(round).invalidate()
  settingsStore.updateProofreadingRound(index, {
    provider: value as ProofreadingRound['provider'],
    apiKey: '',
    modelName: '',
    customBaseUrl: '',
  })
}

async function fetchRoundModels(round: ProofreadingRound) {
  await getRoundModelDiscovery(round).fetchModels()
}

async function testRoundConnection(round: ProofreadingRound) {
  const roundId = round.id
  const roundNumber = proofreadingRounds.value.findIndex(candidate => candidate.id === roundId) + 1
  const provider = round.provider
  const apiKey = round.apiKey?.trim()
  const modelName = round.modelName?.trim()
  const baseUrl = round.customBaseUrl?.trim()

  if (
    providerRequiresApiKeyForBaseUrl(provider, baseUrl) &&
    !apiKey
  ) {
    toast.warning('请先填写 API Key')
    return
  }

  if (!modelName) {
    toast.warning('请填写模型名称')
    return
  }

  if (providerRequiresBaseUrl(provider) && !baseUrl) {
    toast.warning('自定义服务需要填写 Base URL')
    return
  }

  roundTestingStates.value[roundId] = true
  toast.info(`正在测试轮次 ${roundNumber} 的连接...`)

  try {
    const result = await testAiTranslateConnection({
      provider,
      apiKey,
      modelName,
      baseUrl,
      domain: proofreadingProviderDomain(roundId),
    })

    if (result.success) {
      toast.success(result.message || '连接成功!')
    } else {
      toast.error(result.message || '连接失败')
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : '连接测试失败'
    toast.error(errorMessage)
  } finally {
    roundTestingStates.value[roundId] = false
  }
}

function addRound() {
  const newRound: ProofreadingRound = {
    id: newProofreadingRoundId(),
    name: `第${proofreadingRounds.value.length + 1}轮校对`,
    provider: 'siliconflow',
    apiKey: '',
    modelName: '',
    customBaseUrl: '',
    openaiOptions: {
      request: {
        forceJsonOutput: false,
      },
      execution: {
        useStream: true,
        rpmLimit: 7,
        transportRetries: 1,
        businessRetries: DEFAULT_HQ_TRANSLATION_MAX_RETRIES,
      },
    },
    batchSize: 3,
    prompt: DEFAULT_PROOFREADING_PROMPT,
  }
  settingsStore.addProofreadingRound(newRound)
  toast.success('已添加新的校对轮次')
}

function removeRound(index: number) {
  if (proofreadingRounds.value.length <= 1) {
    toast.warning('至少需要保留一个校对轮次')
    return
  }
  const round = proofreadingRounds.value[index]
  if (!round) return
  roundModelDiscoveries.get(round.id)?.invalidate()
  roundModelDiscoveries.delete(round.id)
  delete roundTestingStates.value[round.id]
  settingsStore.removeProofreadingRound(index)
  toast.success('已删除校对轮次')
}

function resetRoundPrompt(index: number) {
  settingsStore.updateProofreadingRound(index, { prompt: DEFAULT_PROOFREADING_PROMPT })
  toast.success('已重置为默认提示词')
}

function handleProofreadingPromptSelect(index: number, content: string, name: string) {
  settingsStore.updateProofreadingRound(index, { prompt: content })
  toast.success(`已应用提示词: ${name}`)
}

onBeforeUnmount(() => {
  roundModelDiscoveries.forEach(discovery => discovery.invalidate())
  roundModelDiscoveries.clear()
})
</script>

<style scoped>
.proofreading-settings__round {
  border: 1px solid var(--color-border-muted);
  border-radius: 8px;
  margin-bottom: 15px;
  overflow: hidden;
}

.proofreading-settings__round-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 15px;
  background: var(--color-surface-subtle);
  border-bottom: 1px solid var(--color-border-muted);
}

.proofreading-settings__round-title {
  font-weight: 500;
}

.proofreading-settings__round-content {
  padding: 15px;
}
</style>
