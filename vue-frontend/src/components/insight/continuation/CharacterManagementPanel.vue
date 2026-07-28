<template>
  <div class="character-management-panel">
    <ProductSectionHeader
      title="角色档案"
      description="点击角色查看和管理形态"
      icon-name="users"
    >
      <template #actions>
        <UiButton variant="primary" @click="openAddCharacterDialog" size="sm">
          <UiIcon name="plus" size="14" />
          <span>新增角色</span>
        </UiButton>
      </template>
    </ProductSectionHeader>

    <ProductStatusBanner
      v-if="characters.length === 0"
      class="character-management-panel__empty-status"
      tone="neutral"
      role="note"
      icon-name="users"
      :title="isLoading ? '正在加载角色档案' : '暂无角色档案'"
    >
      {{ isLoading ? '正在加载角色数据...' : '点击“新增角色”添加角色。' }}
    </ProductStatusBanner>

    <div v-else class="character-management-panel__layout">
      <div class="character-management-panel__grid">
        <ProductRecordCard
          v-for="char in characters"
          :key="char.name"
          as="button"
          class="character-management-panel__tile"
          :class="{
            'character-management-panel__tile--selected': selectedCharacter === char.name,
            'character-management-panel__tile--disabled': char.enabled === false
          }"
          :accent="selectedCharacter === char.name"
          :aria-label="`选择角色 ${char.name}`"
          :aria-pressed="selectedCharacter === char.name"
          @click="selectCharacter(char.name)"
        >
          <ProductAvatar
            :image-src="char.reference_image ? getCharacterImageUrl(char.name) : ''"
            :label="`角色 ${char.name} 头像`"
            :fallback-text="char.name"
            size="md"
            shape="rounded"
          />
          <div class="character-management-panel__tile-name">{{ char.name }}</div>
          <ProductChipList
            v-if="characterTileChips(char).length > 0"
            class="character-management-panel__tile-chips"
            :aria-label="`${char.name}状态`"
            :items="characterTileChips(char)"
          />
        </ProductRecordCard>
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
      :is-generating="orthoGenerating"
      :result-image-path="orthoResultImagePath"
      @close="closeOrthoDialog"
      @generate="handleGenerateOrtho"
      @use-result="handleUseOrthoResult"
    />
  </div>
</template>

<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import ProductAvatar from '@/components/product/ProductAvatar.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import type { ProductChipItem } from '@/components/product/ProductChipList.vue'
import { ref, computed } from 'vue'
import type { CharacterManagementComposable } from '@/composables/continuation/useCharacterManagement'
import type { ContinuationState } from '@/composables/continuation/useContinuationState'
import { confirmProductAction } from '@/composables/useProductConfirm'
import CharacterDetailPanel from './CharacterDetailPanel.vue'
import AddCharacterDialog from './AddCharacterDialog.vue'
import EditCharacterDialog from './EditCharacterDialog.vue'
import AddFormDialog from './AddFormDialog.vue'
import EditFormDialog from './EditFormDialog.vue'
import OrthographicDialog from './OrthographicDialog.vue'
import type { CharacterProfile, CharacterForm } from '@/api/continuation'
import * as continuationApi from '@/api/continuation'

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
const orthoGenerating = ref(false)
const orthoResultImagePath = ref<string | null>(null)

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

function characterTileChips(character: CharacterProfile): ProductChipItem[] {
  const items: ProductChipItem[] = []
  if (character.forms && character.forms.length > 1) {
    items.push({ id: 'forms', label: `${character.forms.length} 个形态`, tone: 'primary' })
  }
  if (character.enabled === false) {
    items.push({ id: 'disabled', label: '禁用', tone: 'warning' })
  }
  return items
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
  const characterName = selectedCharacter.value
  const confirmed = await confirmProductAction({
    title: '删除角色',
    message: `确定要删除角色"${characterName}"吗？`,
    confirmText: '删除',
    cancelText: '取消',
    tone: 'danger',
  })
  if (!confirmed) return

  await charMgmt.deleteCharacter(characterName)
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
  const confirmed = await confirmProductAction({
    title: '删除角色形态',
    message: `确定要删除形态"${form.form_name}"吗？`,
    confirmText: '删除',
    cancelText: '取消',
    tone: 'danger',
  })
  if (!confirmed) return

  await charMgmt.deleteForm(selectedCharacter.value, form.form_id)
}

async function handleUploadFormImage(formId: string, file: File) {
  if (!selectedCharacter.value) return
  await charMgmt.uploadFormImage(selectedCharacter.value, formId, file)
}

async function handleDeleteFormImage(formId: string) {
  if (!selectedCharacter.value) return
  const confirmed = await confirmProductAction({
    title: '删除形态参考图',
    message: '确定要删除形态参考图吗？',
    confirmText: '删除',
    cancelText: '取消',
    tone: 'danger',
  })
  if (!confirmed) return

  await charMgmt.deleteFormImage(selectedCharacter.value, formId)
}

async function handleToggleFormEnabled(formId: string, enabled: boolean) {
  if (!selectedCharacter.value) return
  await charMgmt.toggleFormEnabled(selectedCharacter.value, formId, enabled)
}

function handleGenerateOrthographic(formId: string, formName: string) {
  orthoFormId.value = formId
  orthoFormName.value = formName
  orthoGenerating.value = false
  orthoResultImagePath.value = null
  showOrthoDialog.value = true
}

async function handleGenerateOrtho(sourceImages: File[]) {
  if (!selectedCharacter.value) return

  orthoGenerating.value = true
  orthoResultImagePath.value = null

  try {
    const result = await charMgmt.generateOrtho(
      selectedCharacter.value,
      orthoFormId.value,
      sourceImages
    )

    if (result.success && result.task_id) {
      state.showMessage('三视图任务已进入任务中心，关闭浏览器也会继续运行', 'info')
      await continuationApi.waitForContinuationJob(result.task_id)
      await state.initializeData()
      const form = state.characters.value
        .find(character => character.name === selectedCharacter.value)
        ?.forms.find(item => item.form_id === orthoFormId.value)
      orthoResultImagePath.value = form?.reference_image || null
      if (!orthoResultImagePath.value) {
        throw new Error('任务完成但未找到生成结果')
      }
      state.showMessage('三视图生成成功', 'success')
    } else if (result.success && result.image_path) {
      orthoResultImagePath.value = result.image_path
      state.showMessage('三视图生成成功', 'success')
    } else {
      state.showMessage('生成失败: ' + result.error, 'error')
    }
  } catch (error) {
    state.showMessage('生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    orthoGenerating.value = false
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
  orthoGenerating.value = false
  orthoResultImagePath.value = null
}
</script>

<style scoped>
.character-management-panel {
  container-type: inline-size;
  container-name: continuation-character-management;
}

.character-management-panel__layout {
  display: grid;
  grid-template-columns: minmax(160px, 180px) minmax(0, 1fr);
  gap: 20px;
  min-height: 320px;
  min-width: 0;
}

.character-management-panel__grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  align-content: start;
  max-height: 400px;
  overflow-y: auto;
  padding: 4px;
}

@container continuation-character-management (max-width: 640px) {
  .character-management-panel__layout {
    grid-template-columns: 1fr;
  }

  .character-management-panel__grid {
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    max-height: none;
  }
}

@media (--breakpoint-md-down) {
  .character-management-panel__layout {
    grid-template-columns: 1fr;
  }

  .character-management-panel__grid {
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    max-height: none;
  }
}

.character-management-panel__tile {
  --product-record-card-background: var(--color-surface-base);
  --product-record-card-border: transparent;
  --product-record-card-radius: 12px;
  --product-record-card-padding: 10px 6px;
  --product-record-card-gap: 6px;
  --product-record-card-shadow-hover: none;

  align-items: center;
  text-align: center;
}

.character-management-panel__tile:hover {
  --product-record-card-background: var(--color-surface-muted);
  --product-record-card-border: var(--color-border-brand);
}

.character-management-panel__tile--selected {
  --product-record-card-background: var(--color-surface-muted);
  --product-record-card-border: var(--color-border-brand);
  --product-record-card-shadow: 0 4px 12px var(--color-focus-brand-soft);
}

.character-management-panel__tile--disabled {
  opacity: 0.5;
  filter: grayscale(50%);
}

.character-management-panel__tile-name {
  font-size: 12px;
  font-weight: 500;
  color: var(--color-text-default);
  text-align: center;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.character-management-panel__tile-chips {
  justify-content: center;
}
</style>
