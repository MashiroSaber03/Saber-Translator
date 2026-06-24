<template>
  <div class="character-management-panel">
    <div class="section-header">
      <div class="section-title">
        <h4>🎭 角色档案</h4>
        <p class="hint">点击角色查看和管理形态</p>
      </div>
      <UiButton variant="primary" @click="openAddCharacterDialog" size="sm">
        ➕ 新增角色
      </UiButton>
    </div>
    
    <div v-if="characters.length === 0" class="empty-state">
      <span v-if="isLoading">加载中...</span>
      <span v-else>暂无角色数据，点击"新增角色"添加</span>
    </div>
    
    <div v-else class="character-panel-layout">
      <div class="character-grid-panel">
        <UiButton
          v-for="char in characters" 
          :key="char.name" 
          variant="toolbar"
          class="character-tile"
          :class="{ selected: selectedCharacter === char.name, disabled: char.enabled === false }"
          :aria-pressed="selectedCharacter === char.name"
          @click="selectCharacter(char.name)"
        >
          <div class="tile-avatar">
            <img v-if="char.reference_image" :src="getCharacterImageUrl(char.name)" alt="">
            <div v-else class="tile-avatar-placeholder">
              <span>{{ char.name.charAt(0) }}</span>
            </div>
            <div v-if="char.forms && char.forms.length > 1" class="tile-form-badge">
              {{ char.forms.length }}
            </div>
            <div v-if="char.enabled === false" class="tile-disabled-badge">禁用</div>
          </div>
          <div class="tile-name">{{ char.name }}</div>
        </UiButton>
      </div>
      
      <CharacterDetailPanel
        :character="getSelectedCharacterData()"
        :avatar-url="selectedCharacter ? getCharacterImageUrl(selectedCharacter) : ''"
        :get-form-image-url="(formId) => getFormImageUrl(selectedCharacter!, formId)"
        @toggle-character="handleToggleCharacter"
        @edit-character="openEditCharacterDialog"
        @delete-character="handleDeleteCharacter"
        @add-form="openAddFormDialog"
        @edit-form="openEditFormDialog"
        @delete-form="handleDeleteForm"
        @upload-form-image="handleUploadFormImage"
        @delete-form-image="handleDeleteFormImage"
        @generate-orthographic="handleGenerateOrthographic"
        @toggle-form-enabled="handleToggleFormEnabled"
      />
    </div>
    
    <AddCharacterDialog
      v-if="showAddCharDialog"
      @close="showAddCharDialog = false"
      @add="handleAddCharacter"
    />
    
    <EditCharacterDialog
      v-if="showEditCharDialog && editingCharacter"
      :character="editingCharacter"
      @close="showEditCharDialog = false"
      @save="handleSaveCharacterInfo"
    />
    
    <AddFormDialog
      v-if="showAddFormDialog"
      @close="showAddFormDialog = false"
      @add="handleAddForm"
    />
    
    <EditFormDialog
      v-if="showEditFormDialog && editingForm"
      :form="editingForm"
      @close="showEditFormDialog = false"
      @save="handleSaveFormInfo"
    />
    
    <OrthographicDialog
      v-if="showOrthoDialog && selectedCharacter"
      :character-name="selectedCharacter"
      :form-id="orthoFormId"
      :form-name="orthoFormName"
      :book-id="bookId"
      ref="orthoDialogRef"
      @close="closeOrthoDialog"
      @generate="handleGenerateOrtho"
      @use-result="handleUseOrthoResult"
    />
  </div>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import { ref, computed } from 'vue'
import type { CharacterManagementComposable } from '@/composables/continuation/useCharacterManagement'
import type { ContinuationState } from '@/composables/continuation/useContinuationState'
import CharacterDetailPanel from './CharacterDetailPanel.vue'
import AddCharacterDialog from './AddCharacterDialog.vue'
import EditCharacterDialog from './EditCharacterDialog.vue'
import AddFormDialog from './AddFormDialog.vue'
import EditFormDialog from './EditFormDialog.vue'
import OrthographicDialog from './OrthographicDialog.vue'
import type { CharacterProfile, CharacterForm } from '@/api/continuation'

const props = defineProps<{
  bookId: string
  characterManagement: CharacterManagementComposable
  isLoading?: boolean
  state: ContinuationState
}>()

const charMgmt = props.characterManagement
const state = props.state

const selectedCharacter = ref<string | null>(null)

const showAddCharDialog = ref(false)
const showEditCharDialog = ref(false)
const showAddFormDialog = ref(false)
const showEditFormDialog = ref(false)
const showOrthoDialog = ref(false)

const editingCharacter = ref<CharacterProfile | null>(null)
const editingForm = ref<CharacterForm | null>(null)

const orthoFormId = ref('')
const orthoFormName = ref('')
const orthoDialogRef = ref<InstanceType<typeof OrthographicDialog> | null>(null)

const characters = computed(() => state.characters.value)

function selectCharacter(name: string) {
  selectedCharacter.value = name
}

function getSelectedCharacterData(): CharacterProfile | null {
  if (!selectedCharacter.value) return null
  return characters.value.find(c => c.name === selectedCharacter.value) || null
}

function getCharacterImageUrl(name: string): string {
  return state.getCharacterImageUrl(name)
}

function getFormImageUrl(charName: string, formId: string): string {
  const char = characters.value.find(c => c.name === charName)
  const form = char?.forms?.find(f => f.form_id === formId)
  if (!form?.reference_image) return ''
  return state.getFormImageUrl(form.reference_image)
}

function openAddCharacterDialog() {
  showAddCharDialog.value = true
}

function openEditCharacterDialog() {
  const char = getSelectedCharacterData()
  if (!char) return
  editingCharacter.value = char
  showEditCharDialog.value = true
}

async function handleAddCharacter(name: string, aliases: string[], description: string) {
  await charMgmt.addCharacter(name, aliases, description)
  showAddCharDialog.value = false
}

async function handleSaveCharacterInfo(name: string, aliases: string[]) {
  if (!selectedCharacter.value) return
  await charMgmt.updateCharacterInfo(selectedCharacter.value, name, aliases)
  showEditCharDialog.value = false
}

async function handleDeleteCharacter() {
  if (!selectedCharacter.value) return
  if (!confirm(`确定要删除角色"${selectedCharacter.value}"吗？`)) return
  
  await charMgmt.deleteCharacter(selectedCharacter.value)
  selectedCharacter.value = null
}

async function handleToggleCharacter(enabled: boolean) {
  if (!selectedCharacter.value) return
  await charMgmt.toggleCharacterEnabled(selectedCharacter.value, enabled)
}

function openAddFormDialog() {
  if (!selectedCharacter.value) return
  showAddFormDialog.value = true
}

function openEditFormDialog(form: CharacterForm) {
  if (!selectedCharacter.value) return
  editingForm.value = form
  showEditFormDialog.value = true
}

async function handleAddForm(formName: string, description: string) {
  if (!selectedCharacter.value) return
  await charMgmt.addForm(selectedCharacter.value, formName, description)
  showAddFormDialog.value = false
}

async function handleSaveFormInfo(formName: string, description: string) {
  if (!selectedCharacter.value || !editingForm.value) return
  await charMgmt.updateForm(selectedCharacter.value, editingForm.value.form_id, formName, description)
  showEditFormDialog.value = false
}

async function handleDeleteForm(form: CharacterForm) {
  if (!selectedCharacter.value) return
  if (!confirm(`确定要删除形态"${form.form_name}"吗？`)) return
  
  await charMgmt.deleteForm(selectedCharacter.value, form.form_id)
}

async function handleUploadFormImage(formId: string, file: File) {
  if (!selectedCharacter.value) return
  await charMgmt.uploadFormImage(selectedCharacter.value, formId, file)
}

async function handleDeleteFormImage(formId: string) {
  if (!selectedCharacter.value) return
  if (!confirm('确定要删除形态参考图吗？')) return
  
  await charMgmt.deleteFormImage(selectedCharacter.value, formId)
}

async function handleToggleFormEnabled(formId: string, enabled: boolean) {
  if (!selectedCharacter.value) return
  await charMgmt.toggleFormEnabled(selectedCharacter.value, formId, enabled)
}

function handleGenerateOrthographic(formId: string, formName: string) {
  orthoFormId.value = formId
  orthoFormName.value = formName
  showOrthoDialog.value = true
}

async function handleGenerateOrtho(sourceImages: File[]) {
  if (!selectedCharacter.value) return
  
  orthoDialogRef.value?.setGenerating(true)
  
  try {
    const result = await charMgmt.generateOrtho(
      selectedCharacter.value,
      orthoFormId.value,
      sourceImages
    )
    
    if (result.success && result.image_path) {
      orthoDialogRef.value?.setResult(result.image_path)
      state.showMessage('三视图生成成功', 'success')
    } else {
      state.showMessage('生成失败: ' + result.error, 'error')
      orthoDialogRef.value?.setGenerating(false)
    }
  } catch (error) {
    state.showMessage('生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
    orthoDialogRef.value?.setGenerating(false)
  }
}

async function handleUseOrthoResult(imagePath: string) {
  if (!selectedCharacter.value) return
  
  try {
    await charMgmt.setFormReference(selectedCharacter.value, orthoFormId.value, imagePath)
    state.showMessage('三视图已设置为形态参考图', 'success')
    closeOrthoDialog()
  } catch (error) {
    state.showMessage('设置失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  }
}

function closeOrthoDialog() {
  showOrthoDialog.value = false
  orthoFormId.value = ''
  orthoFormName.value = ''
}
</script>

<style scoped>
.character-management-panel {
  --character-management-panel-border-default: #c7d2fe;
  --character-management-panel-shadow-default: rgba(99, 102, 241, .2);
  --character-management-panel-surface-base: #f5f7ff;
  --character-management-panel-surface-raised: #eef2ff;
  --character-management-panel-surface-muted: #e8e8ff;
  --character-management-panel-surface-subtle: #f0f0f0;
  --character-management-panel-surface-hover: #8b5cf6;
  --character-management-panel-surface-active: rgba(239, 68, 68, .9);
  --character-management-panel-text-primary: #374151;
  --ui-button-padding: 10px 20px;
  --ui-button-radius: 8px;
  --ui-button-font-size: 14px;
  --ui-button-primary-background: var(--color-surface-brand);
  --ui-button-primary-hover-background: var(--color-surface-brand-strong);
  --ui-button-sm-padding: 6px 12px;
  --ui-button-sm-font-size: 13px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 16px;
}

.section-title h4 {
  margin: 0 0 4px;
  font-size: 16px;
}

.section-title .hint {
  margin: 0;
  font-size: 12px;
  color: var(--color-text-supporting, var(--color-text-secondary));
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: var(--color-text-supporting, var(--color-text-secondary));
  font-size: 14px;
}

.character-panel-layout {
  display: grid;
  grid-template-columns: 180px 1fr;
  gap: 20px;
  min-height: 320px;
}

.character-grid-panel {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  align-content: start;
  max-height: 400px;
  overflow-y: auto;
  padding: 4px;
}

.character-tile {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 10px 6px;
  border-radius: 12px;
  background: var(--color-surface-base);
  border: 2px solid transparent;
  color: inherit;
  cursor: pointer;
  font: inherit;
  text-align: center;
  transition: all 0.2s ease;
}

.character-tile:hover {
  background: var(--character-management-panel-surface-base);
  border-color: var(--character-management-panel-border-default);
}

.character-tile.selected {
  background: linear-gradient(135deg, var(--character-management-panel-surface-raised) 0%, var(--character-management-panel-surface-muted) 100%);
  border-color: var(--color-border-brand);
  box-shadow: 0 4px 12px var(--character-management-panel-shadow-default);
}

.character-tile.disabled {
  opacity: 0.5;
  filter: grayscale(50%);
}

.tile-avatar {
  width: 56px;
  height: 56px;
  border-radius: 10px;
  overflow: hidden;
  position: relative;
  background: var(--character-management-panel-surface-subtle);
  margin-bottom: 6px;
}

.tile-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.tile-avatar-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, var(--color-surface-brand-gradient-start) 0%, var(--color-surface-brand-gradient-end) 100%);
  color: var(--color-text-inverse);
  font-size: 20px;
  font-weight: 600;
}

.tile-form-badge {
  position: absolute;
  bottom: -4px;
  right: -4px;
  background: linear-gradient(135deg, var(--color-surface-brand) 0%, var(--character-management-panel-surface-hover) 100%);
  color: var(--color-text-inverse);
  font-size: 10px;
  font-weight: 600;
  min-width: 18px;
  height: 18px;
  border-radius: 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 2px solid var(--color-surface-base);
}

.tile-disabled-badge {
  position: absolute;
  top: 2px;
  left: 2px;
  background: var(--character-management-panel-surface-active);
  color: var(--color-text-inverse);
  font-size: 9px;
  font-weight: 500;
  padding: 1px 4px;
  border-radius: 4px;
}

.tile-name {
  font-size: 12px;
  font-weight: 500;
  color: var(--character-management-panel-text-primary);
  text-align: center;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
