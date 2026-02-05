<template>
  <div class="character-management-panel">
    <div class="section-header">
      <div class="section-title">
        <h4>🎭 角色档案</h4>
        <p class="hint">点击角色查看和管理形态</p>
      </div>
      <button class="btn small primary" @click="openAddCharacterDialog">
        ➕ 新增角色
      </button>
    </div>
    
    <div v-if="characters.length === 0" class="empty-state">
      <span v-if="isLoading">加载中...</span>
      <span v-else>暂无角色数据，点击"新增角色"添加</span>
    </div>
    
    <div v-else class="character-panel-layout">
      <!-- 左侧：角色网格 -->
      <div class="character-grid-panel">
        <div 
          v-for="char in characters" 
          :key="char.name" 
          class="character-tile"
          :class="{ selected: selectedCharacter === char.name, disabled: char.enabled === false }"
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
        </div>
      </div>
      
      <!-- 右侧：角色详情面板 -->
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
    
    <!-- 对话框 -->
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
import { ref, computed } from 'vue'
import { useCharacterManagementInject } from '@/composables/continuation/useCharacterManagement'
import { useContinuationStateInject } from '@/composables/continuation/useContinuationState'
import CharacterDetailPanel from './CharacterDetailPanel.vue'
import AddCharacterDialog from './AddCharacterDialog.vue'
import EditCharacterDialog from './EditCharacterDialog.vue'
import AddFormDialog from './AddFormDialog.vue'
import EditFormDialog from './EditFormDialog.vue'
import OrthographicDialog from './OrthographicDialog.vue'
import type { CharacterProfile, CharacterForm } from '@/api/continuation'

const props = defineProps<{
  bookId: string
  isLoading?: boolean
}>()

const charMgmt = useCharacterManagementInject()
const state = useContinuationStateInject()

// 角色选择
const selectedCharacter = ref<string | null>(null)

// 对话框状态
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

// 角色操作
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

// 形态操作
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

// 三视图生成
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
  /* 样式继承自原组件 */
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 16px;
}

.section-title h4 {
  margin: 0 0 4px 0;
  font-size: 16px;
}

.section-title .hint {
  margin: 0;
  font-size: 12px;
  color: var(--text-secondary, #666);
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn.primary {
  background: var(--primary, #6366f1);
  color: white;
}

.btn.primary:hover {
  background: var(--primary-dark, #4f46e5);
}

.btn.small {
  padding: 6px 12px;
  font-size: 13px;
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: var(--text-secondary, #666);
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
  background: #fff;
  border: 2px solid transparent;
  cursor: pointer;
  transition: all 0.2s ease;
}

.character-tile:hover {
  background: #f5f7ff;
  border-color: #c7d2fe;
}

.character-tile.selected {
  background: linear-gradient(135deg, #eef2ff 0%, #e8e8ff 100%);
  border-color: #6366f1;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
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
  background: #f0f0f0;
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
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-size: 20px;
  font-weight: 600;
}

.tile-form-badge {
  position: absolute;
  bottom: -4px;
  right: -4px;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  color: white;
  font-size: 10px;
  font-weight: 600;
  min-width: 18px;
  height: 18px;
  border-radius: 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 2px solid white;
}

.tile-disabled-badge {
  position: absolute;
  top: 2px;
  left: 2px;
  background: rgba(239, 68, 68, 0.9);
  color: white;
  font-size: 9px;
  font-weight: 500;
  padding: 1px 4px;
  border-radius: 4px;
}

.tile-name {
  font-size: 12px;
  font-weight: 500;
  color: #374151;
  text-align: center;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
