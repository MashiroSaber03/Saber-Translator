<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { getLayerStyleModel, imitateLayerStyle, type LayerStyleImitation, type LayerStyleModel, type LayerStyleRecipe } from '@/api/v2/content'
import BaseModal from '@/components/common/BaseModal.vue'
import ReferenceImageInput from '@/components/common/ReferenceImageInput.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiInput from '@/components/ui/UiInput.vue'
import UiSelect from '@/components/ui/UiSelect.vue'
import { showToast } from '@/utils/toast'
import { useTextStyleFavorites } from './useTextStyleFavorites'

const props = defineProps<{ canApply: boolean; apply: (recipe: LayerStyleRecipe) => Promise<boolean> }>()
const emit = defineEmits<{ close: []; openSettings: [] }>()
const { favorites, saveRecipe, remove } = useTextStyleFavorites()
const controller = new AbortController()
const reference = ref<InstanceType<typeof ReferenceImageInput>>()
const referenceReady = ref(false)
const capturing = ref(false)
const generating = ref(false)
const applying = ref(false)
const model = ref<LayerStyleModel>()
const result = ref<LayerStyleImitation>()
const name = ref('')
const selected = ref('')
const error = ref('')
const available = computed(() => favorites.value.filter(item => item.recipe))
const savedOptions = computed(() => available.value.map(item => ({ value: item.name, label: item.name })))
const selectedRecipe = computed(() => available.value.find(item => item.name === selected.value)?.recipe)
const busy = computed(() => generating.value || applying.value || capturing.value)

onMounted(async () => {
  try { model.value = await getLayerStyleModel(controller.signal) }
  catch {
    if (!controller.signal.aborted) error.value = '无法读取 AI 视觉模型配置，请关闭后重试。'
  }
})
onBeforeUnmount(() => controller.abort())

function referenceChanged(ready: boolean): void {
  referenceReady.value = ready
  result.value = undefined
  if (model.value?.ready) error.value = ''
}

async function generate(): Promise<void> {
  if (!model.value?.ready || !referenceReady.value || !reference.value || busy.value) return
  generating.value = true
  result.value = undefined
  error.value = ''
  try {
    const image = await reference.value.read()
    if (controller.signal.aborted) return
    const next = await imitateLayerStyle(image, controller.signal)
    if (controller.signal.aborted) return
    name.value = next.name
    result.value = next
  } catch (cause) {
    if (!controller.signal.aborted) error.value = cause instanceof Error ? cause.message : '无法生成样式'
  } finally { generating.value = false }
}

function save(): void {
  if (result.value && saveRecipe(name.value, result.value.recipe)) {
    selected.value = name.value.trim()
    showToast('已保存到我的仿制样式', 'success')
  }
}

function removeSelected(): void {
  if (selectedRecipe.value && remove(selected.value, true)) selected.value = ''
}

async function applyRecipe(recipe: LayerStyleRecipe): Promise<void> {
  if (!props.canApply || busy.value) return
  applying.value = true
  error.value = ''
  try {
    if (await props.apply(recipe)) emit('close')
  } catch (cause) {
    if (!controller.signal.aborted) error.value = cause instanceof Error ? cause.message : '样式套用失败'
  } finally { applying.value = false }
}
</script>

<template>
  <BaseModal
    title="参考图仿制样式"
    size="small"
    width="min(100% - 24px, 500px)"
    :close-on-overlay="false"
    :show-close-button="!applying"
    :close-on-esc="!applying"
    @close="emit('close')"
  >
    <div class="style-imitation">
      <div v-if="savedOptions.length" class="style-imitation__row style-imitation__saved">
        <UiSelect :model-value="selected" size="sm" aria-label="我的仿制样式" placeholder="我的仿制样式" :options="savedOptions" :disabled="busy" @update:model-value="selected = String($event ?? '')" />
        <UiButton size="sm" variant="secondary" :disabled="!selectedRecipe || !canApply || busy" @click="selectedRecipe && applyRecipe(selectedRecipe)">套用收藏</UiButton>
        <UiIconButton size="sm" variant="soft" label="删除仿制样式" :disabled="!selectedRecipe || busy" @click="removeSelected"><UiIcon name="trash" size="15" /></UiIconButton>
      </div>
      <ReferenceImageInput ref="reference" :disabled="generating || applying" @change="referenceChanged" @busy="capturing = $event" />
      <div class="style-imitation__row style-imitation__submit">
        <p v-if="model?.ready" class="style-imitation__hint">开始后将参考选区发送给 {{ model.model }}。</p>
        <p v-else-if="model" class="style-imitation__hint">{{ model.message }}</p>
        <p v-else-if="!error" class="style-imitation__hint" role="status">正在读取模型配置…</p>
        <UiButton v-if="model && !model.ready" size="sm" variant="secondary" :disabled="busy" @click="emit('openSettings')">打开设置</UiButton>
        <UiButton v-else size="sm" variant="primary" :disabled="!model?.ready || !referenceReady || busy" @click="generate">
          {{ generating ? '正在仿制…' : result ? '重新仿制' : '开始仿制' }}
        </UiButton>
      </div>
      <p v-if="generating" role="status" class="style-imitation__hint">正在分析图层特效并生成样张。</p>
      <p v-if="error" class="style-imitation__error" role="alert">{{ error }}</p>
      <section v-if="result" class="style-imitation__result" aria-label="仿制样张">
        <img :src="result.preview" alt="仿制样式的实际渲染样张" width="480" height="168">
        <div class="style-imitation__row">
          <UiInput v-model="name" size="sm" maxlength="30" aria-label="预设名称" :disabled="applying" />
          <UiButton size="sm" variant="secondary" :disabled="!name.trim() || applying" @click="save">保存预设</UiButton>
          <UiButton size="sm" variant="primary" :disabled="!canApply || busy" @click="applyRecipe(result.recipe)">{{ applying ? '正在套用…' : '套用样式' }}</UiButton>
        </div>
        <p v-if="!canApply" class="style-imitation__hint">可先收藏，再选中文字图层套用。</p>
      </section>
    </div>
  </BaseModal>
</template>

<style scoped>
.style-imitation, .style-imitation__result { display: grid; gap: 12px; }
.style-imitation__row { display: grid; grid-template-columns: minmax(0, 1fr) auto auto; align-items: center; gap: 8px; }
.style-imitation__submit { grid-template-columns: minmax(0, 1fr) auto; }
.style-imitation__saved, .style-imitation__result { padding-block-start: 4px; }
.style-imitation__saved { padding-block-end: 12px; border-bottom: 1px solid var(--color-border-muted); }
.style-imitation__result img { width: 100%; height: auto; display: block; border-radius: var(--radius-md); }
.style-imitation__hint, .style-imitation__error { margin: 0; font-size: 12px; overflow-wrap: anywhere; }
.style-imitation__hint { color: var(--color-text-supporting); }
.style-imitation__error { color: var(--color-status-error); }
</style>
