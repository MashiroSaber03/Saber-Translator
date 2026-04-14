<template>
  <div class="character-card-studio">
    <div class="studio-sidebar">
      <div class="panel">
        <h3>🎴 角色卡工坊</h3>
        <p class="hint">从漫画分析数据生成 SillyTavern V2 PNG 角色卡。</p>

        <div class="actions">
          <button class="btn" :disabled="isLoading" @click="loadAll">刷新</button>
          <button
            class="btn primary"
            :disabled="selectedCandidateNames.length === 0 || isGenerating || isLoading"
            @click="generateDrafts"
          >
            {{ isGenerating ? '生成中...' : `生成草稿 (${selectedCandidateNames.length})` }}
          </button>
          <button class="btn" :disabled="!draft || isSaving" @click="saveDraft">
            {{ isSaving ? '保存中...' : '保存草稿' }}
          </button>
          <button class="btn" :disabled="!draft || isCompiling" @click="compileDraft">
            {{ isCompiling ? '编译中...' : '编译校验' }}
          </button>
        </div>

        <div v-if="errorMessage" class="message error">{{ errorMessage }}</div>
        <div v-if="successMessage" class="message success">{{ successMessage }}</div>
      </div>

      <div class="panel">
        <div class="panel-title">候选角色</div>
        <div v-if="isLoadingCandidates" class="placeholder">加载中...</div>
        <div v-else-if="candidates.length === 0" class="placeholder">暂无候选角色，请先生成增强时间线。</div>
        <div v-else class="candidate-list">
          <label v-for="item in candidates" :key="item.name" class="candidate-item">
            <input v-model="selectedCandidateNames" type="checkbox" :value="item.name">
            <div class="candidate-main">
              <div class="name">{{ item.name }}</div>
              <div class="meta">
                首次: {{ item.first_appearance || '-' }} 页 · 对话: {{ item.dialogue_count }}
              </div>
            </div>
          </label>
        </div>
      </div>

      <div class="panel">
        <div class="panel-title">草稿角色</div>
        <div v-if="!draft || draft.cards.length === 0" class="placeholder">尚未生成草稿</div>
        <div v-else>
          <div class="draft-toolbar">
            <button class="mini-btn" @click="selectAllDrafts">全选批量</button>
            <button class="mini-btn" @click="clearBatchSelection">清空批量</button>
          </div>
          <div class="draft-list">
            <div v-for="item in draft.cards" :key="item.character" class="draft-item">
              <button
                class="draft-btn"
                :class="{ active: selectedDraftCharacter === item.character }"
                @click="selectedDraftCharacter = item.character"
              >
                {{ item.character }}
              </button>
              <label class="batch-mark">
                <input v-model="selectedBatchCharacters" type="checkbox" :value="item.character">
                批量
              </label>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="studio-main">
      <FieldLockPanel
        :locks="fieldLocks"
        :options="lockOptions"
        @toggle="toggleFieldLock"
        @unlock-all="unlockAllFields"
      />
      <CardTemplatePresetPanel
        :current-character="selectedDraftCharacter"
        :selected-characters="batchCharacters"
        @apply="applyPreset"
      />
      <BatchEditPanel :target-characters="batchCharacters" :locks="fieldLocks" @apply="applyBatchEdit" />

      <CardFieldEditor :card="selectedCard" :locked-paths="fieldLocks" @update="updateCard" />
      <WorldbookEditor
        :book="selectedCard?.data.character_book || null"
        :locked="isLocked('data.character_book')"
        @update="updateWorldbook"
      />
      <RegexEditor
        :rules="regexProfiles"
        :locked="isLocked('data.extensions.saber_tavern.regex_profiles')"
        @update="updateRegexProfiles"
      />
      <MvuEditor
        :variables="mvuVariables"
        :locked="isLocked('data.extensions.saber_tavern.mvu.variables')"
        @update="updateMvuVariables"
      />
      <HelperUiEditor
        :manifest="uiManifest"
        :locked="isLocked('data.extensions.saber_tavern.ui_manifest')"
        @update="updateUiManifest"
      />
      <ValidationPanel :result="compileResult" />
      <PngExportPanel
        :selected-character="selectedDraftCharacter"
        :selected-characters="batchCharacters"
        :can-export="canExport"
        :exporting-single="exportingSingle"
        :exporting-batch="exportingBatch"
        @export-single="exportSingle"
        @export-batch="exportBatch"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import {
  compileCharacterCards,
  exportCharacterCardPng,
  exportCharacterCardsBatch,
  generateCharacterCardDrafts,
  getCharacterCardCandidates,
  getCharacterCardDraft,
  saveCharacterCardDraft,
} from '@/api/characterCard'
import type {
  CharacterCandidate,
  CharacterCardCompileResponse,
  CharacterCardDraftPayload,
  CharacterCardV2,
  CharacterBookSchema,
  HelperUiManifest,
  MvuVariable,
  RegexProfile,
} from '@/types/characterCard'
import CardFieldEditor from './CardFieldEditor.vue'
import WorldbookEditor from './WorldbookEditor.vue'
import RegexEditor from './RegexEditor.vue'
import MvuEditor from './MvuEditor.vue'
import HelperUiEditor from './HelperUiEditor.vue'
import ValidationPanel from './ValidationPanel.vue'
import PngExportPanel from './PngExportPanel.vue'
import FieldLockPanel from './FieldLockPanel.vue'
import BatchEditPanel from './BatchEditPanel.vue'
import CardTemplatePresetPanel from './CardTemplatePresetPanel.vue'

interface BatchEditPayload {
  characters: string[]
  tag_mode: 'append' | 'replace'
  tags: string[]
  system_prompt: string
  post_history_instructions: string
  alternate_greetings: string[]
}

interface PresetApplyPayload {
  preset_id: string
  scope: 'current' | 'selected'
}

const lockOptions = [
  { path: 'data.name', label: '角色名' },
  { path: 'data.description', label: '描述' },
  { path: 'data.personality', label: '人格' },
  { path: 'data.scenario', label: '场景' },
  { path: 'data.system_prompt', label: 'System Prompt' },
  { path: 'data.post_history_instructions', label: 'Post History' },
  { path: 'data.alternate_greetings', label: '问候组' },
  { path: 'data.tags', label: '标签' },
  { path: 'data.character_book', label: '世界书' },
  { path: 'data.extensions.saber_tavern.regex_profiles', label: '正则模板' },
  { path: 'data.extensions.saber_tavern.mvu.variables', label: 'MVU变量' },
  { path: 'data.extensions.saber_tavern.ui_manifest', label: 'UI模板' },
]

const insightStore = useInsightStore()
const bookId = computed(() => insightStore.currentBookId || '')

const isLoadingCandidates = ref(false)
const isLoadingDraft = ref(false)
const isGenerating = ref(false)
const isSaving = ref(false)
const isCompiling = ref(false)
const exportingSingle = ref(false)
const exportingBatch = ref(false)

const errorMessage = ref('')
const successMessage = ref('')

const candidates = ref<CharacterCandidate[]>([])
const selectedCandidateNames = ref<string[]>([])
const selectedBatchCharacters = ref<string[]>([])
const fieldLocks = ref<Record<string, boolean>>({})
const draft = ref<CharacterCardDraftPayload | null>(null)
const selectedDraftCharacter = ref('')
const compileResult = ref<CharacterCardCompileResponse | null>(null)

const isLoading = computed(() => isLoadingCandidates.value || isLoadingDraft.value)
const draftCharacters = computed(() => draft.value?.cards.map(item => item.character) || [])
const batchCharacters = computed(() => {
  const allowed = new Set(draftCharacters.value)
  return selectedBatchCharacters.value.filter(item => allowed.has(item))
})

const selectedDraftItem = computed(() => {
  if (!draft.value || !selectedDraftCharacter.value) return null
  return draft.value.cards.find(item => item.character === selectedDraftCharacter.value) || null
})

const selectedCard = computed<CharacterCardV2 | null>(() => selectedDraftItem.value?.card || null)

const regexProfiles = computed<RegexProfile[]>(() => {
  return selectedCard.value?.data?.extensions?.saber_tavern?.regex_profiles || []
})

const mvuVariables = computed<MvuVariable[]>(() => {
  return selectedCard.value?.data?.extensions?.saber_tavern?.mvu?.variables || []
})

const uiManifest = computed<HelperUiManifest | null>(() => {
  return selectedCard.value?.data?.extensions?.saber_tavern?.ui_manifest || null
})

const canExport = computed(() => !!compileResult.value?.valid)

function cloneDeep<T>(value: T): T {
  return JSON.parse(JSON.stringify(value))
}

function clearMessages() {
  errorMessage.value = ''
  successMessage.value = ''
}

function normalizeDraft(payload: any): CharacterCardDraftPayload {
  return {
    book_id: payload.book_id || bookId.value,
    style: payload.style || 'balanced',
    generated_at: payload.generated_at,
    saved_at: payload.saved_at,
    cards: payload.cards || [],
    missing_characters: payload.missing_characters || [],
  }
}

function ensureSelectedCharacter() {
  if (!draft.value || draft.value.cards.length === 0) {
    selectedDraftCharacter.value = ''
    return
  }
  if (!selectedDraftCharacter.value || !draft.value.cards.some(c => c.character === selectedDraftCharacter.value)) {
    selectedDraftCharacter.value = draft.value.cards[0]!.character
  }
}

function ensureBatchSelection() {
  if (!draft.value || draft.value.cards.length === 0) {
    selectedBatchCharacters.value = []
    return
  }
  const valid = new Set(draft.value.cards.map(item => item.character))
  selectedBatchCharacters.value = selectedBatchCharacters.value.filter(name => valid.has(name))
  if (selectedBatchCharacters.value.length === 0) {
    selectedBatchCharacters.value = draft.value.cards.map(item => item.character)
  }
}

function selectAllDrafts() {
  selectedBatchCharacters.value = [...draftCharacters.value]
}

function clearBatchSelection() {
  selectedBatchCharacters.value = []
}

function isLocked(path: string): boolean {
  return !!fieldLocks.value[path]
}

function toggleFieldLock(path: string, value: boolean) {
  fieldLocks.value = {
    ...fieldLocks.value,
    [path]: value,
  }
}

function unlockAllFields() {
  fieldLocks.value = {}
}

function getByPath(target: Record<string, any>, path: string): any {
  const keys = path.split('.')
  let current: any = target
  for (const key of keys) {
    if (current == null || typeof current !== 'object' || !(key in current)) {
      return undefined
    }
    current = current[key]
  }
  return current
}

function setByPath(target: Record<string, any>, path: string, value: any) {
  const keys = path.split('.')
  let current: any = target
  for (let i = 0; i < keys.length - 1; i += 1) {
    const key = keys[i]!
    if (!current[key] || typeof current[key] !== 'object') {
      current[key] = {}
    }
    current = current[key]
  }
  current[keys[keys.length - 1]!] = cloneDeep(value)
}

function deleteByPath(target: Record<string, any>, path: string) {
  const keys = path.split('.')
  let current: any = target
  for (let i = 0; i < keys.length - 1; i += 1) {
    const key = keys[i]!
    if (!current || typeof current !== 'object' || !(key in current)) {
      return
    }
    current = current[key]
  }
  const lastKey = keys[keys.length - 1]!
  if (current && typeof current === 'object' && lastKey in current) {
    delete current[lastKey]
  }
}

function enforceLocks(previous: CharacterCardV2, next: CharacterCardV2): CharacterCardV2 {
  const protectedCard = cloneDeep(next)
  for (const [path, locked] of Object.entries(fieldLocks.value)) {
    if (!locked) continue
    const oldValue = getByPath(previous as unknown as Record<string, any>, path)
    if (oldValue !== undefined) {
      setByPath(protectedCard as unknown as Record<string, any>, path, oldValue)
    } else {
      // 锁定字段在旧卡不存在时，阻止本次新增该字段
      deleteByPath(protectedCard as unknown as Record<string, any>, path)
    }
  }
  return protectedCard
}

async function loadCandidates() {
  if (!bookId.value) return
  isLoadingCandidates.value = true
  try {
    const response = await getCharacterCardCandidates(bookId.value)
    if (response.success) {
      candidates.value = response.candidates || []
      if (!draft.value || draft.value.cards.length === 0) {
        selectedCandidateNames.value = candidates.value.slice(0, 3).map(item => item.name)
      }
    } else {
      throw new Error(response.error || '加载候选失败')
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '加载候选失败'
  } finally {
    isLoadingCandidates.value = false
  }
}

async function loadDraft() {
  if (!bookId.value) return
  isLoadingDraft.value = true
  try {
    const response = await getCharacterCardDraft(bookId.value)
    if (response.success && response.has_data && response.draft) {
      draft.value = response.draft
      selectedCandidateNames.value = draft.value.cards.map(item => item.character)
      ensureSelectedCharacter()
      ensureBatchSelection()
    } else {
      draft.value = null
      selectedDraftCharacter.value = ''
      selectedBatchCharacters.value = []
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '加载草稿失败'
  } finally {
    isLoadingDraft.value = false
  }
}

async function loadAll() {
  clearMessages()
  compileResult.value = null
  await Promise.all([loadCandidates(), loadDraft()])
}

async function generateDrafts() {
  if (!bookId.value) return
  clearMessages()
  isGenerating.value = true
  try {
    const response: any = await generateCharacterCardDrafts(
      bookId.value,
      selectedCandidateNames.value,
      'balanced'
    )
    if (response.success === false) {
      throw new Error(response.error || '生成失败')
    }
    draft.value = normalizeDraft(response)
    ensureSelectedCharacter()
    ensureBatchSelection()
    compileResult.value = null
    successMessage.value = `已生成 ${draft.value.cards.length} 张角色卡草稿`
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '生成角色卡失败'
  } finally {
    isGenerating.value = false
  }
}

async function saveDraft() {
  if (!bookId.value || !draft.value) return
  clearMessages()
  isSaving.value = true
  try {
    const response = await saveCharacterCardDraft(bookId.value, draft.value)
    if (!response.success) {
      throw new Error(response.error || '保存失败')
    }
    successMessage.value = response.message || '草稿已保存'
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '保存草稿失败'
  } finally {
    isSaving.value = false
  }
}

async function compileDraft() {
  if (!bookId.value || !draft.value) return
  clearMessages()
  isCompiling.value = true
  try {
    const response = await compileCharacterCards(bookId.value, { draft: draft.value })
    compileResult.value = response
    if (!response.success) {
      throw new Error(response.error || '编译失败')
    }

    if (response.compiled_cards && draft.value) {
      draft.value.cards = draft.value.cards.map(item => ({
        ...item,
        card: response.compiled_cards?.[item.character] || item.card,
      }))
    }
    successMessage.value = response.message || (response.valid ? '编译通过' : '编译完成，存在错误')
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '编译失败'
  } finally {
    isCompiling.value = false
  }
}

function updateCard(card: CharacterCardV2) {
  if (!draft.value || !selectedDraftCharacter.value) return
  const idx = draft.value.cards.findIndex(item => item.character === selectedDraftCharacter.value)
  if (idx < 0) return

  const oldCard = draft.value.cards[idx]!.card
  const lockedSafeCard = enforceLocks(oldCard, card)
  draft.value.cards[idx] = {
    ...draft.value.cards[idx]!,
    card: lockedSafeCard,
  }
}

function ensureSaberExtension(card: CharacterCardV2) {
  if (!card.data.extensions) {
    card.data.extensions = {}
  }
  if (!card.data.extensions.saber_tavern) {
    card.data.extensions.saber_tavern = {
      regex_profiles: [],
      mvu: { version: '1.0.0', variables: [] },
      ui_manifest: {
        layout: 'split-dashboard',
        theme: 'manga-insight-light',
        panels: [],
        widgets: [],
        actions: [],
        events: [],
        bindings: [],
      },
      import_manifest: {},
    }
  }
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values.map(v => v.trim()).filter(Boolean))]
}

function updateWorldbook(book: CharacterBookSchema) {
  if (!selectedCard.value || isLocked('data.character_book')) return
  selectedCard.value.data.character_book = cloneDeep(book)
}

function updateRegexProfiles(rules: RegexProfile[]) {
  if (!selectedCard.value || isLocked('data.extensions.saber_tavern.regex_profiles')) return
  ensureSaberExtension(selectedCard.value)
  selectedCard.value.data.extensions.saber_tavern!.regex_profiles = cloneDeep(rules)
}

function updateMvuVariables(variables: MvuVariable[]) {
  if (!selectedCard.value || isLocked('data.extensions.saber_tavern.mvu.variables')) return
  ensureSaberExtension(selectedCard.value)
  selectedCard.value.data.extensions.saber_tavern!.mvu.variables = cloneDeep(variables)
}

function updateUiManifest(manifest: HelperUiManifest) {
  if (!selectedCard.value || isLocked('data.extensions.saber_tavern.ui_manifest')) return
  ensureSaberExtension(selectedCard.value)
  selectedCard.value.data.extensions.saber_tavern!.ui_manifest = cloneDeep(manifest)
}

function sanitizeFilename(name: string): string {
  const safe = name.replace(/[\\/:*?"<>|]/g, '_').trim()
  return safe || 'character_card'
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

function applyBatchEdit(payload: BatchEditPayload) {
  if (!draft.value || payload.characters.length === 0) return
  const targetSet = new Set(payload.characters)
  let changedCount = 0

  draft.value.cards = draft.value.cards.map(item => {
    if (!targetSet.has(item.character)) return item
    let nextCard = cloneDeep(item.card)

    if (payload.tags.length > 0 && !isLocked('data.tags')) {
      if (payload.tag_mode === 'replace') {
        nextCard.data.tags = uniqueStrings(payload.tags)
      } else {
        nextCard.data.tags = uniqueStrings([...(nextCard.data.tags || []), ...payload.tags])
      }
    }

    if (payload.system_prompt && !isLocked('data.system_prompt')) {
      nextCard.data.system_prompt = payload.system_prompt
    }

    if (payload.post_history_instructions && !isLocked('data.post_history_instructions')) {
      nextCard.data.post_history_instructions = payload.post_history_instructions
    }

    if (payload.alternate_greetings.length > 0 && !isLocked('data.alternate_greetings')) {
      nextCard.data.alternate_greetings = uniqueStrings([
        ...(nextCard.data.alternate_greetings || []),
        ...payload.alternate_greetings,
      ])
    }

    nextCard = enforceLocks(item.card, nextCard)
    changedCount += 1
    return {
      ...item,
      card: nextCard,
    }
  })

  if (changedCount > 0) {
    compileResult.value = null
    successMessage.value = `已批量更新 ${changedCount} 个角色草稿`
  }
}

function upsertRegexRule(card: CharacterCardV2, rule: RegexProfile) {
  ensureSaberExtension(card)
  const list = card.data.extensions.saber_tavern!.regex_profiles || []
  const index = list.findIndex(item => item.id === rule.id)
  if (index >= 0) list[index] = rule
  else list.push(rule)
  card.data.extensions.saber_tavern!.regex_profiles = list
}

function upsertMvuVariable(card: CharacterCardV2, variable: MvuVariable) {
  ensureSaberExtension(card)
  const list = card.data.extensions.saber_tavern!.mvu.variables || []
  const index = list.findIndex(item => item.name === variable.name)
  if (index >= 0) list[index] = variable
  else list.push(variable)
  card.data.extensions.saber_tavern!.mvu.variables = list
}

function applyPresetToCard(card: CharacterCardV2, presetId: string): CharacterCardV2 {
  const nextCard = cloneDeep(card)
  ensureSaberExtension(nextCard)

  if (presetId === 'balanced') {
    if (!isLocked('data.system_prompt')) {
      nextCard.data.system_prompt = '保持角色设定稳定，优先依据已知剧情和关系回应。'
    }
    if (!isLocked('data.post_history_instructions')) {
      nextCard.data.post_history_instructions = '保持叙事连续，避免跳出漫画世界观。'
    }
    if (!isLocked('data.tags')) {
      nextCard.data.tags = uniqueStrings([...(nextCard.data.tags || []), 'preset-balanced'])
    }
    if (!isLocked('data.extensions.saber_tavern.ui_manifest')) {
      nextCard.data.extensions.saber_tavern!.ui_manifest.theme = 'manga-insight-light'
    }
  }

  if (presetId === 'dramatic') {
    if (!isLocked('data.personality')) {
      nextCard.data.personality = `${nextCard.data.personality}\n情绪阈值较低，冲突时反应更强烈。`.trim()
    }
    if (!isLocked('data.system_prompt')) {
      nextCard.data.system_prompt = '突出冲突、代价与选择，保持强动机与高张力表达。'
    }
    if (!isLocked('data.tags')) {
      nextCard.data.tags = uniqueStrings([...(nextCard.data.tags || []), 'preset-dramatic'])
    }
    if (!isLocked('data.extensions.saber_tavern.regex_profiles')) {
      upsertRegexRule(nextCard, {
        id: 'dramatic_trim',
        name: '冲突语气收敛',
        enabled: true,
        scope: 'character',
        source: 'ai_output',
        pattern: '[!！]{3,}',
        replacement: '！！',
        flags: 'g',
        depth_min: 0,
        depth_max: 99,
        order: 90,
        notes: '避免感叹号失控，保持可读性。',
      })
    }
    if (!isLocked('data.extensions.saber_tavern.mvu.variables')) {
      upsertMvuVariable(nextCard, {
        name: 'conflict_level',
        type: 'number',
        scope: 'chat',
        default: 40,
        value: 40,
        validator: { min: 0, max: 100 },
        description: '冲突强度，用于调节对话张力。',
      })
    }
    if (!isLocked('data.extensions.saber_tavern.ui_manifest')) {
      nextCard.data.extensions.saber_tavern!.ui_manifest.theme = 'manga-insight-drama'
    }
  }

  if (presetId === 'daily') {
    if (!isLocked('data.personality')) {
      nextCard.data.personality = `${nextCard.data.personality}\n日常语气更柔和，优先建立信任与陪伴感。`.trim()
    }
    if (!isLocked('data.system_prompt')) {
      nextCard.data.system_prompt = '降低敌意表达，突出日常互动、细节关怀和轻松交流。'
    }
    if (!isLocked('data.tags')) {
      nextCard.data.tags = uniqueStrings([...(nextCard.data.tags || []), 'preset-daily'])
    }
    if (!isLocked('data.alternate_greetings')) {
      nextCard.data.alternate_greetings = uniqueStrings([
        ...(nextCard.data.alternate_greetings || []),
        '今天状态还不错，要不要聊点轻松的？',
      ])
    }
    if (!isLocked('data.extensions.saber_tavern.ui_manifest')) {
      nextCard.data.extensions.saber_tavern!.ui_manifest.theme = 'manga-insight-soft'
    }
  }

  if (presetId === 'mystery') {
    if (!isLocked('data.scenario')) {
      nextCard.data.scenario = `${nextCard.data.scenario}\n当前阶段以线索拼接和信息验证为优先任务。`.trim()
    }
    if (!isLocked('data.system_prompt')) {
      nextCard.data.system_prompt = '优先追踪线索与证据链，控制信息披露节奏。'
    }
    if (!isLocked('data.tags')) {
      nextCard.data.tags = uniqueStrings([...(nextCard.data.tags || []), 'preset-mystery'])
    }
    if (!isLocked('data.extensions.saber_tavern.regex_profiles')) {
      upsertRegexRule(nextCard, {
        id: 'mystery_quote_balance',
        name: '悬疑台词括号清理',
        enabled: true,
        scope: 'character',
        source: 'ai_output',
        pattern: '[（(]{2,}',
        replacement: '（',
        flags: 'g',
        depth_min: 0,
        depth_max: 99,
        order: 95,
        notes: '避免额外括号影响可读性。',
      })
    }
    if (!isLocked('data.extensions.saber_tavern.mvu.variables')) {
      upsertMvuVariable(nextCard, {
        name: 'clue_progress',
        type: 'number',
        scope: 'chat',
        default: 0,
        value: 0,
        validator: { min: 0, max: 100 },
        description: '线索推进度。',
      })
    }
    if (!isLocked('data.extensions.saber_tavern.ui_manifest')) {
      nextCard.data.extensions.saber_tavern!.ui_manifest.theme = 'manga-insight-mystery'
    }
  }

  return enforceLocks(card, nextCard)
}

function applyPreset(payload: PresetApplyPayload) {
  if (!draft.value) return
  const targets =
    payload.scope === 'current'
      ? [selectedDraftCharacter.value].filter(Boolean)
      : [...batchCharacters.value]
  if (targets.length === 0) return

  const targetSet = new Set(targets)
  let changedCount = 0
  draft.value.cards = draft.value.cards.map(item => {
    if (!targetSet.has(item.character)) return item
    changedCount += 1
    return {
      ...item,
      card: applyPresetToCard(item.card, payload.preset_id),
    }
  })

  if (changedCount > 0) {
    compileResult.value = null
    successMessage.value = `预设 ${payload.preset_id} 已应用到 ${changedCount} 个角色`
  }
}

async function exportSingle(character: string) {
  if (!bookId.value || !character) return
  clearMessages()
  exportingSingle.value = true
  try {
    const blob = await exportCharacterCardPng(bookId.value, character)
    downloadBlob(blob, `${sanitizeFilename(character)}.png`)
    successMessage.value = `已导出 ${character}.png`
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '导出失败'
  } finally {
    exportingSingle.value = false
  }
}

async function exportBatch(characters: string[]) {
  if (!bookId.value || characters.length === 0) return
  clearMessages()
  exportingBatch.value = true
  try {
    const blob = await exportCharacterCardsBatch(bookId.value, characters)
    downloadBlob(blob, `${bookId.value}_character_cards.zip`)
    successMessage.value = `已导出 ${characters.length} 张角色卡`
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '批量导出失败'
  } finally {
    exportingBatch.value = false
  }
}

watch(bookId, async newBookId => {
  if (!newBookId) return
  candidates.value = []
  draft.value = null
  selectedCandidateNames.value = []
  selectedDraftCharacter.value = ''
  selectedBatchCharacters.value = []
  compileResult.value = null
  await loadAll()
}, { immediate: true })

watch(draftCharacters, () => {
  ensureBatchSelection()
})
</script>

<style scoped>
.character-card-studio {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 14px;
  padding: 14px;
}

.studio-sidebar {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.studio-main {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.panel {
  background: var(--bg-secondary, #fff);
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 10px;
  padding: 12px;
}

.panel h3 {
  margin: 0 0 6px;
  font-size: 15px;
}

.hint {
  margin: 0 0 10px;
  color: var(--text-secondary, #64748b);
  font-size: 12px;
}

.panel-title {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 8px;
}

.placeholder {
  color: var(--text-secondary, #64748b);
  font-size: 13px;
}

.actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.btn {
  border: 1px solid var(--border-color, #e2e8f0);
  background: var(--bg-tertiary, #f1f5f9);
  border-radius: 8px;
  padding: 7px 10px;
  font-size: 12px;
  cursor: pointer;
}

.btn.primary {
  background: var(--color-primary, #6366f1);
  color: #fff;
  border-color: var(--color-primary, #6366f1);
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.message {
  margin-top: 8px;
  font-size: 12px;
  padding: 6px 8px;
  border-radius: 6px;
}

.message.error {
  color: #b91c1c;
  background: rgba(239, 68, 68, 0.12);
}

.message.success {
  color: #166534;
  background: rgba(34, 197, 94, 0.12);
}

.candidate-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 300px;
  overflow-y: auto;
}

.candidate-item {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  padding: 8px;
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 8px;
  cursor: pointer;
}

.candidate-main .name {
  font-size: 13px;
  color: var(--text-primary, #0f172a);
}

.candidate-main .meta {
  font-size: 11px;
  color: var(--text-secondary, #64748b);
}

.draft-toolbar {
  display: flex;
  gap: 6px;
  margin-bottom: 8px;
}

.mini-btn {
  border: 1px solid var(--border-color, #e2e8f0);
  background: var(--bg-primary, #f8fafc);
  color: var(--text-secondary, #475569);
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 11px;
  cursor: pointer;
}

.draft-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.draft-item {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 6px;
  align-items: center;
}

.draft-btn {
  border: 1px solid var(--border-color, #e2e8f0);
  background: var(--bg-primary, #f8fafc);
  color: var(--text-primary, #0f172a);
  border-radius: 8px;
  padding: 6px 10px;
  font-size: 12px;
  text-align: left;
  cursor: pointer;
}

.draft-btn.active {
  background: var(--color-primary, #6366f1);
  border-color: var(--color-primary, #6366f1);
  color: #fff;
}

.batch-mark {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--text-secondary, #64748b);
}

@media (max-width: 1200px) {
  .character-card-studio {
    grid-template-columns: 1fr;
  }
}
</style>
