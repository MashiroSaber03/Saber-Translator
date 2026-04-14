<template>
  <div class="editor-card">
    <h4>模板预设</h4>
    <p class="hint">快速应用角色卡语气/规则/MVU/UI 模板。</p>

    <div class="grid">
      <label>
        预设
        <select v-model="presetId">
          <option v-for="item in presets" :key="item.id" :value="item.id">
            {{ item.label }}
          </option>
        </select>
      </label>
      <label>
        作用范围
        <select v-model="scope">
          <option value="current">当前角色（{{ currentCharacter || '-' }}）</option>
          <option value="selected">已勾选角色（{{ selectedCharacters.length }}）</option>
        </select>
      </label>
    </div>

    <div class="desc">{{ currentPreset?.description }}</div>

    <button
      class="btn primary"
      :disabled="(scope === 'current' && !currentCharacter) || (scope === 'selected' && selectedCharacters.length === 0)"
      @click="applyPreset"
    >
      应用预设
    </button>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

interface PresetOption {
  id: string
  label: string
  description: string
}

interface PresetApplyPayload {
  preset_id: string
  scope: 'current' | 'selected'
}

const presets: PresetOption[] = [
  {
    id: 'balanced',
    label: '平衡型',
    description: '保持中性叙事与稳定角色一致性，适合默认对话场景。',
  },
  {
    id: 'dramatic',
    label: '冲突型',
    description: '强化冲突张力与剧情推进语气，提升情绪变化幅度。',
  },
  {
    id: 'daily',
    label: '日常型',
    description: '降低攻击性，增加亲和对话和轻量互动风格。',
  },
  {
    id: 'mystery',
    label: '悬疑型',
    description: '强化线索追踪与信息揭示节奏，适合推理故事。',
  },
]

const props = defineProps<{
  currentCharacter: string
  selectedCharacters: string[]
}>()

const emit = defineEmits<{
  (e: 'apply', payload: PresetApplyPayload): void
}>()

const presetId = ref('balanced')
const scope = ref<'current' | 'selected'>('current')

const currentPreset = computed(() => presets.find(item => item.id === presetId.value))

function applyPreset() {
  emit('apply', {
    preset_id: presetId.value,
    scope: scope.value,
  })
}
</script>

<style scoped>
.editor-card {
  background: var(--bg-secondary, #fff);
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 10px;
  padding: 14px;
}

.editor-card h4 {
  margin: 0 0 6px;
  font-size: 14px;
}

.hint {
  margin: 0 0 10px;
  color: var(--text-secondary, #64748b);
  font-size: 12px;
}

.grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: var(--text-secondary, #64748b);
}

select {
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 8px;
  padding: 8px;
  font-size: 13px;
  color: var(--text-primary, #0f172a);
  background: var(--bg-primary, #f8fafc);
}

.desc {
  margin: 10px 0;
  font-size: 12px;
  color: var(--text-primary, #0f172a);
}

.btn {
  border: 1px solid var(--color-primary, #6366f1);
  background: var(--color-primary, #6366f1);
  color: #fff;
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 12px;
  cursor: pointer;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

@media (max-width: 900px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
