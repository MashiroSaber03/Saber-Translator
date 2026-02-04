<script setup lang="ts">
/**
 * 漫画续写面板组件
 * 
 * 功能步骤：
 * 1. 设置 - 配置续写参数
 * 2. 脚本 - 生成和编辑全话脚本
 * 3. 页面 - 分页剧情细化
 * 4. 生成 - 图片生成和预览
 * 5. 导出 - 导出成品
 */
import { ref, computed, onMounted, watch } from 'vue'
import { useInsightStore } from '@/stores/insightStore'
import * as continuationApi from '@/api/continuation'

// ============================================================
// 类型定义
// ============================================================

interface CharacterForm {
  form_id: string
  form_name: string
  description: string
  reference_image: string
  enabled?: boolean
}

interface CharacterRef {
  name: string
  aliases: string[]
  description: string
  forms: CharacterForm[]
  reference_image: string  // 任意一张参考图（用于展示）
  enabled?: boolean
}

interface ChapterScript {
  chapter_title: string
  page_count: number
  script_text: string
  generated_at: string
}

interface PageContent {
  page_number: number
  characters: string[]
  description: string
  dialogues: Array<{ character: string; text: string }>
  image_prompt: string
  image_url: string
  previous_url: string
  status: 'pending' | 'generating' | 'generated' | 'failed'
}

// ============================================================
// 状态
// ============================================================

const insightStore = useInsightStore()

/** 当前步骤 (0-4) */
const currentStep = ref(0)

/** 步骤名称 */
const stepNames = ['设置', '脚本', '页面', '生成', '导出']

/** 加载状态 */
const isLoading = ref(false)

/** 错误信息 */
const errorMessage = ref('')

/** 成功信息 */
const successMessage = ref('')

// ===== 步骤1: 设置 =====
const pageCount = ref(15)
const styleRefPages = ref(3)
const continuationDirection = ref('')
const characters = ref<CharacterRef[]>([])
const isDataReady = ref(false)

// 三视图生成
const showOrthoDialog = ref(false)
const orthoCharacter = ref<CharacterRef | null>(null)
const orthoFormId = ref<string>('')  // 当前生成三视图的形态ID
const orthoFormName = ref<string>('')  // 当前生成三视图的形态名称
const orthoSourceImages = ref<File[]>([])
const isGeneratingOrtho = ref(false)
const orthoGenerationProgress = ref('')  // 生成进度消息
const orthoResult = ref<string | null>(null)
const imageRefreshKey = ref(Date.now())  // 用于强制刷新图片缓存
const isDraggingOrtho = ref(false)  // 拖拽上传状态

// 编辑角色
const showEditCharDialog = ref(false)
const editingCharacter = ref<CharacterRef | null>(null)
const editCharName = ref('')
const editCharAliases = ref('')  // 用逗号分隔的别名字符串

// 新增角色
const showAddCharDialog = ref(false)
const newCharName = ref('')
const newCharAliases = ref('')
const newCharDescription = ref('')
const isAddingChar = ref(false)

// 形态管理
const selectedCharacter = ref<string | null>(null)  // 选中的角色（用于详情面板）
const showAddFormDialog = ref(false)
const addFormForCharacter = ref<string | null>(null)
const newFormId = ref('')
const newFormName = ref('')
const newFormDescription = ref('')
const isAddingForm = ref(false)

const showEditFormDialog = ref(false)
const editingForm = ref<CharacterForm | null>(null)
const editFormCharacterName = ref('')
const editFormName = ref('')
const editFormDescription = ref('')
const isSavingForm = ref(false)

// ===== 步骤2: 脚本 =====
const chapterScript = ref<ChapterScript | null>(null)
const isGeneratingScript = ref(false)

// ===== 步骤3: 页面 =====
const pages = ref<PageContent[]>([])
const isGeneratingPages = ref(false)
const isGeneratingPrompts = ref(false)
const regeneratingPromptPage = ref<number | null>(null)

// ===== 步骤4: 生成 =====
const generationProgress = ref(0)
const isGeneratingImages = ref(false)
const sessionId = ref('')

// ===== 步骤5: 导出 =====
const exportFormat = ref<'images' | 'pdf'>('images')

// ============================================================
// 计算属性
// ============================================================

const canProceedToScript = computed(() => {
  return isDataReady.value && pageCount.value > 0
})

const canProceedToPages = computed(() => {
  return chapterScript.value !== null
})

const canProceedToGenerate = computed(() => {
  return pages.value.length > 0 && pages.value.every(p => p.image_prompt)
})

const canProceedToExport = computed(() => {
  return pages.value.some(p => p.status === 'generated')
})

const generatedPagesCount = computed(() => {
  return pages.value.filter(p => p.status === 'generated').length
})

// ============================================================
// 方法
// ============================================================

/** 显示消息 */
function showMessage(message: string, type: 'success' | 'error' | 'info'): void {
  if (type === 'success') {
    successMessage.value = message
    setTimeout(() => { successMessage.value = '' }, 3000)
  } else if (type === 'info') {
    // info 类型使用 success 样式但不自动清除（会被下一条消息覆盖）
    successMessage.value = message
  } else {
    errorMessage.value = message
    setTimeout(() => { errorMessage.value = '' }, 5000)
  }
}

/** 切换步骤 */
function goToStep(step: number): void {
  if (step >= 0 && step <= 4) {
    currentStep.value = step
  }
}

// ===== 步骤1: 设置相关方法 =====

/** 初始化数据 */
async function initializeData(): Promise<void> {
  if (!insightStore.currentBookId) {
    showMessage('请先选择书籍', 'error')
    return
  }
  
  isLoading.value = true
  errorMessage.value = ''
  
  try {
    // 准备续写数据（检查分析数据是否就绪）
    const prepareResult = await continuationApi.prepareContinuation(insightStore.currentBookId)
    
    if (prepareResult.success && prepareResult.ready) {
      isDataReady.value = true
      
      // 加载角色列表
      const charResult = await continuationApi.getCharacters(insightStore.currentBookId)
      if (charResult.success && charResult.characters) {
        characters.value = charResult.characters
      }
      
      // 恢复已保存的续写数据
      if (prepareResult.saved_data?.has_data) {
        const savedData = prepareResult.saved_data
        
        // 恢复配置
        if (savedData.config) {
          if (savedData.config.page_count) pageCount.value = savedData.config.page_count
          if (savedData.config.style_reference_pages) styleRefPages.value = savedData.config.style_reference_pages
          if (savedData.config.continuation_direction) continuationDirection.value = savedData.config.continuation_direction
        }
        
        // 恢复脚本
        if (savedData.script) {
          chapterScript.value = savedData.script
        }
        
        // 恢复页面详情
        if (savedData.pages && savedData.pages.length > 0) {
          pages.value = savedData.pages
        }
        
        showMessage('已恢复上次的续写进度', 'success')
      } else {
        showMessage('数据准备就绪', 'success')
      }
    } else {
      showMessage(prepareResult.message || '数据还在准备中，请稍后重试', 'error')
    }
  } catch (error) {
    showMessage('初始化失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isLoading.value = false
  }
}

// ===== 编辑角色方法 =====

/** 打开编辑角色弹窗 */
function openEditCharDialog(char: CharacterRef): void {
  editingCharacter.value = char
  editCharName.value = char.name
  editCharAliases.value = char.aliases.join(', ')
  showEditCharDialog.value = true
}

/** 关闭编辑角色弹窗 */
function closeEditCharDialog(): void {
  showEditCharDialog.value = false
  editingCharacter.value = null
}

/** 保存角色信息 */
async function saveCharacterInfo(): Promise<void> {
  if (!editingCharacter.value) return
  
  const originalName = editingCharacter.value.name
  const newName = editCharName.value.trim()
  const newAliases = editCharAliases.value
    .split(/[,，]/)  // 支持中英文逗号
    .map(a => a.trim())
    .filter(a => a.length > 0)
  
  if (!newName) {
    showMessage('角色名不能为空', 'error')
    return
  }
  
  try {
    const result = await continuationApi.updateCharacterInfo(
      insightStore.currentBookId!,
      originalName,
      { name: newName, aliases: newAliases }
    )
    
    if (result.success) {
      // 更新本地状态
      const char = characters.value.find(c => c.name === originalName)
      if (char && result.character) {
        char.name = result.character.name
        char.aliases = result.character.aliases
      }
      showMessage('角色信息已更新', 'success')
      closeEditCharDialog()
    } else {
      showMessage('保存失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('保存失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  }
}

// ===== 新增/删除角色方法 =====

/** 打开新增角色弹窗 */
function openAddCharDialog(): void {
  newCharName.value = ''
  newCharAliases.value = ''
  newCharDescription.value = ''
  showAddCharDialog.value = true
}

/** 关闭新增角色弹窗 */
function closeAddCharDialog(): void {
  showAddCharDialog.value = false
}

/** 确认新增角色 */
async function confirmAddCharacter(): Promise<void> {
  const name = newCharName.value.trim()
  if (!name) {
    showMessage('角色名不能为空', 'error')
    return
  }
  
  // 解析别名
  const aliases = newCharAliases.value
    .split(/[,，]/)
    .map(a => a.trim())
    .filter(a => a.length > 0)
  
  isAddingChar.value = true
  
  try {
    const result = await continuationApi.addCharacter(
      insightStore.currentBookId!,
      {
        name,
        aliases,
        description: newCharDescription.value.trim()
      }
    )
    
    if (result.success && result.character) {
      // 添加到本地列表
      characters.value.push(result.character)
      showMessage(`角色 "${name}" 已添加`, 'success')
      closeAddCharDialog()
    } else {
      showMessage('添加失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('添加失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isAddingChar.value = false
  }
}

/** 删除角色 */
async function deleteCharacter(char: CharacterRef): Promise<void> {
  // 确认删除
  if (!confirm(`确定要删除角色 "${char.name}" 吗？此操作不可恢复。`)) {
    return
  }
  
  try {
    const result = await continuationApi.deleteCharacter(
      insightStore.currentBookId!,
      char.name
    )
    
    if (result.success) {
      // 从本地列表移除
      const index = characters.value.findIndex(c => c.name === char.name)
      if (index !== -1) {
        characters.value.splice(index, 1)
      }
      showMessage(`角色 "${char.name}" 已删除`, 'success')
    } else {
      showMessage('删除失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('删除失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  }
}

// ===== 形态管理方法 =====

/** 选中角色（显示详情面板） */
function selectCharacter(charName: string): void {
  selectedCharacter.value = selectedCharacter.value === charName ? null : charName
}

/** 获取当前选中的角色对象 */
function getSelectedCharacterData(): CharacterRef | null {
  if (!selectedCharacter.value) return null
  return characters.value.find(c => c.name === selectedCharacter.value) || null
}

/** 打开新增形态弹窗 */
function openAddFormDialog(charName: string): void {
  addFormForCharacter.value = charName
  newFormId.value = ''
  newFormName.value = ''
  newFormDescription.value = ''
  showAddFormDialog.value = true
}

/** 关闭新增形态弹窗 */
function closeAddFormDialog(): void {
  showAddFormDialog.value = false
  addFormForCharacter.value = null
}

/** 确认添加形态 */
async function confirmAddForm(): Promise<void> {
  if (!addFormForCharacter.value || !newFormId.value.trim() || !newFormName.value.trim()) {
    showMessage('请填写形态ID和名称', 'error')
    return
  }
  
  isAddingForm.value = true
  try {
    const result = await continuationApi.addCharacterForm(
      insightStore.currentBookId!,
      addFormForCharacter.value,
      {
        form_id: newFormId.value.trim(),
        form_name: newFormName.value.trim(),
        description: newFormDescription.value.trim()
      }
    )
    
    if (result.success && result.form) {
      // 更新本地角色列表
      const char = characters.value.find(c => c.name === addFormForCharacter.value)
      if (char && char.forms) {
        char.forms.push(result.form)
      }
      showMessage(`形态 "${newFormName.value}" 已添加`, 'success')
      closeAddFormDialog()
    } else {
      showMessage('添加形态失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('添加形态失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isAddingForm.value = false
  }
}

/** 打开编辑形态弹窗 */
function openEditFormDialog(charName: string, form: CharacterForm): void {
  editFormCharacterName.value = charName
  editingForm.value = form
  editFormName.value = form.form_name
  editFormDescription.value = form.description
  showEditFormDialog.value = true
}

/** 关闭编辑形态弹窗 */
function closeEditFormDialog(): void {
  showEditFormDialog.value = false
  editingForm.value = null
  editFormCharacterName.value = ''
}

/** 保存形态信息 */
async function saveFormInfo(): Promise<void> {
  if (!editingForm.value || !editFormCharacterName.value) return
  
  isSavingForm.value = true
  try {
    const result = await continuationApi.updateCharacterForm(
      insightStore.currentBookId!,
      editFormCharacterName.value,
      editingForm.value.form_id,
      {
        form_name: editFormName.value.trim(),
        description: editFormDescription.value.trim()
      }
    )
    
    if (result.success) {
      // 更新本地数据
      const char = characters.value.find(c => c.name === editFormCharacterName.value)
      if (char && char.forms) {
        const form = char.forms.find(f => f.form_id === editingForm.value?.form_id)
        if (form) {
          form.form_name = editFormName.value.trim()
          form.description = editFormDescription.value.trim()
        }
      }
      showMessage('形态信息已更新', 'success')
      closeEditFormDialog()
    } else {
      showMessage('更新失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('更新失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isSavingForm.value = false
  }
}

/** 删除形态 */
async function deleteForm(charName: string, form: CharacterForm): Promise<void> {
  if (!confirm(`确定要删除形态 "${form.form_name}" 吗？`)) {
    return
  }
  
  try {
    const result = await continuationApi.deleteCharacterForm(
      insightStore.currentBookId!,
      charName,
      form.form_id
    )
    
    if (result.success) {
      // 从本地列表移除
      const char = characters.value.find(c => c.name === charName)
      if (char && char.forms) {
        const index = char.forms.findIndex(f => f.form_id === form.form_id)
        if (index !== -1) {
          char.forms.splice(index, 1)
        }
      }
      showMessage(`形态 "${form.form_name}" 已删除`, 'success')
    } else {
      showMessage('删除失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('删除失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  }
}

/** 上传形态参考图 */
async function uploadFormImage(charName: string, formId: string, event: Event): Promise<void> {
  const input = event.target as HTMLInputElement
  if (!input.files?.length) return
  
  const file = input.files[0]
  if (!file) return
  
  const formData = new FormData()
  formData.append('image', file)
  
  try {
    const result = await continuationApi.uploadFormImage(
      insightStore.currentBookId!,
      charName,
      formId,
      formData
    )
    
    if (result.success && result.image_path) {
      // 更新本地数据
      const char = characters.value.find(c => c.name === charName)
      if (char && char.forms) {
        const form = char.forms.find(f => f.form_id === formId)
        if (form) {
          form.reference_image = result.image_path
        }
      }
      imageRefreshKey.value = Date.now()
      showMessage('形态参考图已上传', 'success')
    } else {
      showMessage('上传失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('上传失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  }
  
  // 清空input
  input.value = ''
}

/** 删除形态参考图 */
async function deleteFormImageAction(charName: string, formId: string): Promise<void> {
  if (!confirm('确定要删除这个形态的参考图吗？')) return
  
  try {
    const result = await continuationApi.deleteFormImage(
      insightStore.currentBookId!,
      charName,
      formId
    )
    
    if (result.success) {
      // 更新本地数据
      const char = characters.value.find(c => c.name === charName)
      if (char && char.forms) {
        const form = char.forms.find(f => f.form_id === formId)
        if (form) {
          form.reference_image = ''
        }
      }
      imageRefreshKey.value = Date.now()
      showMessage('参考图已删除', 'success')
    } else {
      showMessage('删除失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('删除失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  }
}

/** 切换角色启用状态 */
async function handleToggleCharacter(charName: string, enabled: boolean): Promise<void> {
  try {
    const result = await continuationApi.toggleCharacterEnabled(
      insightStore.currentBookId!,
      charName,
      enabled
    )
    
    if (result.success) {
      // 更新本地数据
      const char = characters.value.find(c => c.name === charName)
      if (char) {
        char.enabled = enabled
      }
      showMessage(`角色已${enabled ? '启用' : '禁用'}`, 'success')
    } else {
      showMessage('操作失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('操作失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  }
}

/** 切换形态启用状态 */
async function handleToggleForm(charName: string, formId: string, enabled: boolean): Promise<void> {
  try {
    const result = await continuationApi.toggleFormEnabled(
      insightStore.currentBookId!,
      charName,
      formId,
      enabled
    )
    
    if (result.success) {
      // 更新本地数据
      const char = characters.value.find(c => c.name === charName)
      if (char && char.forms) {
        const form = char.forms.find(f => f.form_id === formId)
        if (form) {
          form.enabled = enabled
        }
      }
      showMessage(`形态已${enabled ? '启用' : '禁用'}`, 'success')
    } else {
      showMessage('操作失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('操作失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  }
}

/** 获取形态参考图URL */
function getFormImageUrl(charName: string, formId: string): string {
  const char = characters.value.find(c => c.name === charName)
  if (!char || !char.forms) return ''
  const form = char.forms.find(f => f.form_id === formId)
  if (!form || !form.reference_image) return ''
  return `/api/manga-insight/file?path=${encodeURIComponent(form.reference_image)}&t=${imageRefreshKey.value}`
}

// ===== 三视图生成方法 =====


/** 打开形态三视图生成弹窗 */
function openOrthoDialog(char: CharacterRef, formId: string, formName: string): void {
  orthoCharacter.value = char
  orthoFormId.value = formId
  orthoFormName.value = formName
  orthoSourceImages.value = []
  orthoResult.value = null
  orthoGenerationProgress.value = ''
  showOrthoDialog.value = true
}

/** 关闭三视图弹窗 */
function closeOrthoDialog(): void {
  showOrthoDialog.value = false
  orthoCharacter.value = null
  orthoFormId.value = ''
  orthoFormName.value = ''
  orthoSourceImages.value = []
  orthoResult.value = null
  orthoGenerationProgress.value = ''
}


/** 选择源图片 */
function selectOrthoImages(event: Event): void {
  const input = event.target as HTMLInputElement
  if (!input.files) return
  
  // 最多选择5张
  const files = Array.from(input.files).slice(0, 5)
  orthoSourceImages.value = files
}

/** 处理拖拽进入 */
function handleOrthoDragEnter(event: DragEvent): void {
  event.preventDefault()
  event.stopPropagation()
  isDraggingOrtho.value = true
}

/** 处理拖拽悬停 */
function handleOrthoDragOver(event: DragEvent): void {
  event.preventDefault()
  event.stopPropagation()
  isDraggingOrtho.value = true
}

/** 处理拖拽离开 */
function handleOrthoDragLeave(event: DragEvent): void {
  event.preventDefault()
  event.stopPropagation()
  isDraggingOrtho.value = false
}

/** 处理拖拽放下 */
function handleOrthoDrop(event: DragEvent): void {
  event.preventDefault()
  event.stopPropagation()
  isDraggingOrtho.value = false
  
  const files = event.dataTransfer?.files
  if (!files || files.length === 0) return
  
  // 过滤只保留图片文件，最多5张
  const imageFiles = Array.from(files)
    .filter(file => file.type.startsWith('image/'))
    .slice(0, 5)
  
  if (imageFiles.length > 0) {
    orthoSourceImages.value = imageFiles
  } else {
    showMessage('请拖拽图片文件', 'error')
  }
}

/** 生成三视图 */
async function generateOrtho(): Promise<void> {
  if (!orthoCharacter.value || !orthoFormId.value || orthoSourceImages.value.length === 0) {
    showMessage('请至少上传一张图片', 'error')
    return
  }
  
  isGeneratingOrtho.value = true
  errorMessage.value = ''
  orthoGenerationProgress.value = `正在上传 ${orthoSourceImages.value.length} 张图片...`
  
  try {
    // 模拟进度提示
    setTimeout(() => {
      if (isGeneratingOrtho.value) {
        orthoGenerationProgress.value = 'AI 正在分析角色特征...'
      }
    }, 500)
    
    setTimeout(() => {
      if (isGeneratingOrtho.value) {
        orthoGenerationProgress.value = '正在生成三视图，请耐心等待...'
      }
    }, 2000)
    
    // 使用形态级别的 API
    const result = await continuationApi.generateFormOrtho(
      insightStore.currentBookId!,
      orthoCharacter.value.name,
      orthoFormId.value,
      orthoSourceImages.value
    )
    
    if (result.success && result.image_path) {
      orthoResult.value = result.image_path
      showMessage('三视图生成成功', 'success')
    } else {
      showMessage('生成失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isGeneratingOrtho.value = false
  }
}

/** 使用生成的三视图 */
async function useOrthoResult(): Promise<void> {
  if (!orthoCharacter.value || !orthoFormId.value || !orthoResult.value) return
  
  try {
    // 调用API将三视图设置为该形态的参考图
    const result = await continuationApi.setFormReference(
      insightStore.currentBookId!,
      orthoCharacter.value.name,
      orthoFormId.value,
      orthoResult.value
    )
    
    if (result.success) {
      // 刷新图片缓存key
      imageRefreshKey.value = Date.now()
      // 设置成功后重新加载角色列表
      await initializeData()
      showMessage('三视图已设置为形态参考图', 'success')
      closeOrthoDialog()
    } else {
      showMessage('设置失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('设置失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  }
}


/** 获取角色图片URL */
function getCharacterImageUrl(characterName: string): string {
  if (!insightStore.currentBookId) return ''
  // 添加时间戳参数以强制刷新缓存
  return `/api/manga-insight/${insightStore.currentBookId}/continuation/characters/${encodeURIComponent(characterName)}/image?t=${imageRefreshKey.value}`
}

/** 创建File对象的URL */
function createObjectURL(file: File): string {
  return window.URL.createObjectURL(file)
}

/** 获取三视图结果URL */
function getOrthoResultUrl(): string {
  if (!insightStore.currentBookId || !orthoResult.value) return ''
  // 使用专门的生成图片API，通过路径参数获取
  return `/api/manga-insight/${insightStore.currentBookId}/continuation/generated-image?path=${encodeURIComponent(orthoResult.value)}`
}

/** 获取生成图片的URL */
function getGeneratedImageUrl(imagePath: string): string {
  if (!insightStore.currentBookId || !imagePath) return ''
  return `/api/manga-insight/${insightStore.currentBookId}/continuation/generated-image?path=${encodeURIComponent(imagePath)}`
}


// ===== 步骤2: 脚本相关方法 =====

/** 生成脚本 */
async function generateScript(): Promise<void> {
  if (!insightStore.currentBookId) return
  
  isGeneratingScript.value = true
  errorMessage.value = ''
  
  try {
    const result = await continuationApi.generateScript(
      insightStore.currentBookId,
      continuationDirection.value,
      pageCount.value
    )
    
    if (result.success && result.script) {
      chapterScript.value = result.script
      
      // 保存配置（脚本后端已自动保存）
      try {
        await continuationApi.saveConfig(insightStore.currentBookId, {
          page_count: pageCount.value,
          style_reference_pages: styleRefPages.value,
          continuation_direction: continuationDirection.value
        })
        console.log('配置已保存')
      } catch (saveError) {
        console.error('保存配置失败:', saveError)
      }
      
      showMessage('脚本生成成功', 'success')
    } else {
      showMessage('脚本生成失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('脚本生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isGeneratingScript.value = false
  }
}

// ===== 步骤3: 页面相关方法 =====

/** 生成页面详情（逐页生成，避免超时）+ 自动生成提示词 */
async function generatePageDetails(): Promise<void> {
  if (!insightStore.currentBookId || !chapterScript.value) return
  
  isGeneratingPages.value = true
  isGeneratingPrompts.value = true  // 同时标记正在生成提示词
  errorMessage.value = ''
  
  // 初始化空的页面列表
  const totalPages = chapterScript.value.page_count || pageCount.value
  pages.value = []
  
  try {
    // 逐页生成详情 + 提示词
    for (let i = 1; i <= totalPages; i++) {
      showMessage(`正在生成第 ${i}/${totalPages} 页详情...`, 'info')
      
      // 步骤1: 生成页面详情
      const detailResult = await continuationApi.generateSinglePageDetails(
        insightStore.currentBookId,
        chapterScript.value,
        i
      )
      
      if (!detailResult.success || !detailResult.page) {
        // 详情生成失败
        pages.value.push({
          page_number: i,
          characters: [],
          description: `生成失败: ${detailResult.error || '未知错误'}`,
          dialogues: [],
          image_prompt: '',
          image_url: '',
          previous_url: '',
          status: 'failed' as const
        })
        console.error(`第 ${i} 页详情生成失败:`, detailResult.error)
        continue
      }
      
      // 步骤2: 立即生成该页的提示词
      showMessage(`正在生成第 ${i}/${totalPages} 页提示词...`, 'info')
      
      const promptResult = await continuationApi.generateSingleImagePrompt(
        insightStore.currentBookId,
        detailResult.page,
        i
      )
      
      if (promptResult.success && promptResult.page) {
        // 成功：添加带提示词的页面
        pages.value.push(promptResult.page)
      } else {
        // 提示词生成失败，但详情是有的
        const pageWithError = { ...detailResult.page }
        pageWithError.image_prompt = `提示词生成失败: ${promptResult.error || '未知错误'}`
        pages.value.push(pageWithError)
        console.error(`第 ${i} 页提示词生成失败:`, promptResult.error)
      }
    }
    
    // 自动保存页面数据到服务器
    try {
      await continuationApi.savePages(insightStore.currentBookId, pages.value)
      console.log('页面数据已自动保存')
    } catch (saveError) {
      console.error('自动保存页面数据失败:', saveError)
    }
    
    showMessage(`页面详情和提示词生成完成 (${pages.value.length} 页)`, 'success')
  } catch (error) {
    showMessage('生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isGeneratingPages.value = false
    isGeneratingPrompts.value = false
  }
}

/** 生成图片提示词 */
async function generateImagePrompts(): Promise<void> {
  if (!insightStore.currentBookId || pages.value.length === 0) return
  
  isGeneratingPrompts.value = true
  errorMessage.value = ''
  
  try {
    const result = await continuationApi.generateImagePrompts(
      insightStore.currentBookId,
      pages.value
    )
    
    if (result.success && result.pages) {
      pages.value = result.pages
      showMessage('提示词生成成功', 'success')
    } else {
      showMessage('提示词生成失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('提示词生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isGeneratingPrompts.value = false
  }
}

/** 更新页面角色列表 */
function updateCharacters(page: PageContent, event: Event): void {
  const input = event.target as HTMLInputElement
  const value = input.value
  page.characters = value.split(',').map(s => s.trim()).filter(s => s)
  onPageDataChange()
}

/** 页面数据变更时的处理 */
function onPageDataChange(): void {
  // 可以在这里添加防抖保存等逻辑
  console.log('页面数据已修改')
}

/** 手动保存页面修改 */
async function savePageChanges(): Promise<void> {
  if (!insightStore.currentBookId || pages.value.length === 0) return
  
  try {
    await continuationApi.savePages(insightStore.currentBookId, pages.value)
    showMessage('页面数据保存成功', 'success')
  } catch (error) {
    showMessage('保存失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  }
}

/** 重新生成单页提示词 */
async function regenerateSinglePrompt(pageNumber: number): Promise<void> {
  if (!insightStore.currentBookId) return
  
  const page = pages.value.find(p => p.page_number === pageNumber)
  if (!page) return
  
  regeneratingPromptPage.value = pageNumber
  
  try {
    // 调用单页提示词生成 API
    const result = await continuationApi.generateSingleImagePrompt(
      insightStore.currentBookId,
      page,
      pageNumber
    )
    
    if (result.success && result.page) {
      // 更新该页的提示词
      page.image_prompt = result.page.image_prompt
      
      // 自动保存
      await continuationApi.savePages(insightStore.currentBookId, pages.value)
      
      showMessage(`第 ${pageNumber} 页提示词已更新`, 'success')
    } else {
      showMessage('生成失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    regeneratingPromptPage.value = null
  }
}

// ===== 步骤4: 生成相关方法 =====

/** 批量生成图片 */
async function batchGenerateImages(): Promise<void> {
  if (!insightStore.currentBookId || pages.value.length === 0) return
  
  isGeneratingImages.value = true
  generationProgress.value = 0
  errorMessage.value = ''
  
  try {
    // 获取初始画风参考图（原漫画最后N页）
    let styleRefs: string[] = []
    const styleResult = await continuationApi.getStyleReferences(
      insightStore.currentBookId,
      styleRefPages.value  // 使用用户设置的参考图数量
    )
    if (styleResult.success && styleResult.images) {
      styleRefs = [...styleResult.images]  // 复制一份
    }
    
    console.log(`初始参考图 (${styleRefPages.value}张):`, styleRefs)
    
    // 逐页生成图片（避免批量超时）
    const totalPages = pages.value.length
    for (let i = 0; i < totalPages; i++) {
      const page = pages.value[i]
      if (!page) continue  // 安全检查
      
      // 跳过已有图片的页面（但将其加入参考图）
      if (page.image_url && page.status !== 'failed') {
        // 将已生成的图片加入参考图滑动窗口
        styleRefs.push(page.image_url)
        // 保持参考图数量不超过用户设置的数量
        if (styleRefs.length > styleRefPages.value) {
          styleRefs = styleRefs.slice(-styleRefPages.value)
        }
        generationProgress.value = Math.round(((i + 1) / totalPages) * 100)
        continue
      }
      
      showMessage(`正在生成第 ${page.page_number}/${totalPages} 页图片...`, 'info')
      page.status = 'generating'
      
      console.log(`第 ${page.page_number} 页使用参考图 (${styleRefs.length}张):`, styleRefs.slice(-3))  // 只打印最后3个
      
      try {
        const result = await continuationApi.generatePageImage(
          insightStore.currentBookId,
          page.page_number,
          page,
          styleRefs,
          sessionId.value,
          styleRefPages.value  // 传递用户设置的参考图数量
        )
        
        if (result.success && result.image_path) {
          page.image_url = result.image_path
          page.status = 'generated'
          
          // 【滑动窗口】将新生成的图片加入参考图
          styleRefs.push(result.image_path)
          // 保持参考图数量不超过用户设置的数量
          if (styleRefs.length > styleRefPages.value) {
            styleRefs = styleRefs.slice(-styleRefPages.value)
          }
          
          // 更新 sessionId（如果后端返回了）
          if (result.session_id) {
            sessionId.value = result.session_id
          }
        } else {
          page.status = 'failed'
          console.error(`第 ${page.page_number} 页图片生成失败:`, result.error)
        }
      } catch (pageError) {
        page.status = 'failed'
        console.error(`第 ${page.page_number} 页图片生成异常:`, pageError)
      }
      
      generationProgress.value = Math.round(((i + 1) / totalPages) * 100)
    }
    
    // 自动保存页面数据（包含生成的图片路径）
    try {
      await continuationApi.savePages(insightStore.currentBookId, pages.value)
      console.log('页面数据（含图片）已自动保存')
    } catch (saveError) {
      console.error('自动保存页面数据失败:', saveError)
    }
    
    const successCount = pages.value.filter(p => p.status === 'generated').length
    showMessage(`图片生成完成 (${successCount}/${totalPages} 页成功)`, 'success')
    
  } catch (error) {
    showMessage('图片生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isGeneratingImages.value = false
    generationProgress.value = 100
  }
}

/** 重新生成单页图片 */
async function regeneratePageImage(pageNumber: number): Promise<void> {
  if (!insightStore.currentBookId) return
  
  const page = pages.value.find(p => p.page_number === pageNumber)
  if (!page) return
  
  page.status = 'generating'
  
  try {
    // 构建滑动窗口参考图（复用批量生成的逻辑）
    let styleRefs: string[] = []
    
    // 1. 先获取原漫画最后N页作为基础参考图
    const styleResult = await continuationApi.getStyleReferences(
      insightStore.currentBookId,
      styleRefPages.value
    )
    if (styleResult.success && styleResult.images) {
      styleRefs = [...styleResult.images]
    }
    
    // 2. 加入该页之前已生成的页面图片（模拟滑动窗口）
    // 例如重新生成第5页时，应该用第2、3、4页的图片作为参考（假设窗口大小为3）
    const pageIndex = pages.value.findIndex(p => p.page_number === pageNumber)
    if (pageIndex > 0) {
      // 遍历该页之前的所有页面
      for (let i = 0; i < pageIndex; i++) {
        const prevPage = pages.value[i]
        if (prevPage && prevPage.image_url && prevPage.status === 'generated') {
          styleRefs.push(prevPage.image_url)
          // 保持参考图数量不超过用户设置的数量
          if (styleRefs.length > styleRefPages.value) {
            styleRefs = styleRefs.slice(-styleRefPages.value)
          }
        }
      }
    }
    
    console.log(`重新生成第 ${pageNumber} 页，使用参考图:`, styleRefs)
    
    const result = await continuationApi.regeneratePageImage(
      insightStore.currentBookId,
      pageNumber,
      page,
      styleRefs,  // 传递滑动窗口参考图
      sessionId.value,
      styleRefPages.value  // 传递滑动窗口大小
    )

    
    if (result.success && result.image_path) {
      page.previous_url = page.image_url
      page.image_url = result.image_path
      page.status = 'generated'
      
      // 自动保存页面数据
      try {
        await continuationApi.savePages(insightStore.currentBookId, pages.value)
      } catch (saveError) {
        console.error('自动保存页面数据失败:', saveError)
      }
      
      showMessage(`第 ${pageNumber} 页重新生成成功`, 'success')
    } else {
      page.status = 'failed'
      showMessage('重新生成失败: ' + result.error, 'error')
    }
  } catch (error) {
    page.status = 'failed'
    showMessage('重新生成失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  }
}

/** 使用上一版本 */
async function usePreviousVersion(pageNumber: number): Promise<void> {
  const page = pages.value.find(p => p.page_number === pageNumber)
  if (page && page.previous_url) {
    const temp = page.image_url
    page.image_url = page.previous_url
    page.previous_url = temp
    
    // 自动保存页面数据
    if (insightStore.currentBookId) {
      try {
        await continuationApi.savePages(insightStore.currentBookId, pages.value)
      } catch (saveError) {
        console.error('自动保存页面数据失败:', saveError)
      }
    }
    
    showMessage('已切换到上一版本', 'success')
  }
}

// ===== 步骤5: 导出相关方法 =====

/** 导出为图片 ZIP */
async function exportAsImages(): Promise<void> {
  if (!insightStore.currentBookId || pages.value.length === 0) {
    showMessage('没有可导出的页面', 'error')
    return
  }
  
  isLoading.value = true
  
  try {
    // 后端会直接从 pages.json 加载图片路径
    const blob = await continuationApi.exportAsImages(insightStore.currentBookId)
    
    // 创建下载链接
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `continuation_${Date.now()}.zip`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
    
    showMessage('导出成功', 'success')
  } catch (error) {
    showMessage('导出失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isLoading.value = false
  }
}

/** 导出为 PDF */
async function exportAsPdf(): Promise<void> {
  if (!insightStore.currentBookId || pages.value.length === 0) {
    showMessage('没有可导出的页面', 'error')
    return
  }
  
  isLoading.value = true
  
  try {
    // 后端会直接从 pages.json 加载图片路径
    const blob = await continuationApi.exportAsPdf(insightStore.currentBookId)
    
    // 创建下载链接
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `continuation_${Date.now()}.pdf`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
    
    showMessage('导出成功', 'success')
  } catch (error) {
    showMessage('导出失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isLoading.value = false
  }
}

/** 清除续写数据，重新开始 */
async function clearAndRestart(): Promise<void> {
  if (!insightStore.currentBookId) return
  
  if (!confirm('确定要清除所有续写数据并重新开始吗？此操作不可恢复。')) {
    return
  }
  
  isLoading.value = true
  
  try {
    const result = await continuationApi.clearContinuationData(insightStore.currentBookId)
    
    if (result.success) {
      // 重置所有状态
      chapterScript.value = null
      pages.value = []
      sessionId.value = ''
      currentStep.value = 0
      
      showMessage('续写数据已清除，可以重新开始', 'success')
    } else {
      showMessage('清除失败: ' + result.error, 'error')
    }
  } catch (error) {
    showMessage('清除失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isLoading.value = false
  }
}

// ============================================================
// 生命周期
// ============================================================

onMounted(() => {
  if (insightStore.currentBookId) {
    initializeData()
  }
})

watch(() => insightStore.currentBookId, (newBookId) => {
  if (newBookId) {
    // 重置状态
    currentStep.value = 0
    isDataReady.value = false
    characters.value = []
    chapterScript.value = null
    pages.value = []
    sessionId.value = ''
    initializeData()
  }
})
</script>

<template>
  <div class="continuation-panel">
    <!-- 消息提示 -->
    <div v-if="errorMessage" class="message error">{{ errorMessage }}</div>
    <div v-if="successMessage" class="message success">{{ successMessage }}</div>
    
    <!-- 步骤指示器 -->
    <div class="step-indicator">
      <div 
        v-for="(name, index) in stepNames" 
        :key="index"
        class="step"
        :class="{ 
          active: currentStep === index, 
          completed: currentStep > index,
          clickable: index <= currentStep || (index === 1 && canProceedToScript)
        }"
        @click="goToStep(index)"
      >
        <span class="step-number">{{ index + 1 }}</span>
        <span class="step-name">{{ name }}</span>
      </div>
    </div>
    
    <!-- 步骤内容 -->
    <div class="step-content">
      <!-- 步骤1: 设置 -->
      <div v-show="currentStep === 0" class="step-panel">
        <h3>📝 续写设置</h3>
        
        <div class="form-group">
          <label>续写页数</label>
          <input v-model.number="pageCount" type="number" min="5" max="50">
          <p class="hint">建议 10-20 页</p>
        </div>
        
        <div class="form-group">
          <label>画风参考页数</label>
          <input v-model.number="styleRefPages" type="number" min="1" max="10">
          <p class="hint">用于维持画风一致性</p>
        </div>
        
        <div class="form-group">
          <label>续写方向（可选）</label>
          <textarea 
            v-model="continuationDirection" 
            rows="4" 
            placeholder="例如：延续主线剧情，探索新的冒险..."
          ></textarea>
          <p class="hint">留空将自动根据剧情发展生成</p>
        </div>
        
        <!-- 角色管理区域 - 左右分栏布局 -->
        <div class="characters-section">
          <div class="section-header">
            <div class="section-title">
              <h4>🎭 角色档案</h4>
              <p class="hint">点击角色查看和管理形态</p>
            </div>
            <button class="btn small primary" @click="openAddCharDialog">
              ➕ 新增角色
            </button>
          </div>
          
          <div v-if="characters.length === 0" class="empty-state">
            <span v-if="isLoading">加载中...</span>
            <span v-else>暂无角色数据，点击"新增角色"添加</span>
          </div>
          
          <!-- 左右分栏容器 -->
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
            <div class="character-detail-panel" :class="{ 'has-selection': selectedCharacter }">
              <template v-if="getSelectedCharacterData()">
                <div class="detail-header">
                  <div class="detail-main-info">
                    <div class="detail-avatar">
                      <img v-if="getSelectedCharacterData()?.reference_image" :src="getCharacterImageUrl(getSelectedCharacterData()!.name)" alt="">
                      <div v-else class="detail-avatar-placeholder">{{ getSelectedCharacterData()?.name.charAt(0) }}</div>
                    </div>
                    <div class="detail-info">
                      <h4>{{ getSelectedCharacterData()?.name }}</h4>
                      <p v-if="getSelectedCharacterData()?.aliases?.length" class="detail-aliases">
                        别名：{{ getSelectedCharacterData()?.aliases.join('、') }}
                      </p>
                    </div>
                  </div>
                  <div class="detail-actions">
                    <label class="toggle-switch" title="启用/禁用角色">
                      <input 
                        type="checkbox" 
                        :checked="getSelectedCharacterData()?.enabled !== false"
                        @change="handleToggleCharacter(getSelectedCharacterData()!.name, ($event.target as HTMLInputElement).checked)"
                      >
                      <span class="toggle-slider"></span>
                    </label>
                    <button class="icon-btn-lg" @click="openEditCharDialog(getSelectedCharacterData()!)" title="编辑角色">✏️</button>
                    <button class="icon-btn-lg danger" @click="deleteCharacter(getSelectedCharacterData()!)" title="删除角色">🗑️</button>
                  </div>
                </div>
                
                <div class="detail-forms-section">
                  <div class="forms-header">
                    <h5>角色形态 ({{ getSelectedCharacterData()?.forms?.length || 1 }})</h5>
                    <button class="btn small secondary" @click="openAddFormDialog(getSelectedCharacterData()!.name)">
                      ➕ 添加形态
                    </button>
                  </div>
                  
                  <div class="forms-grid">
                    <div 
                      v-for="form in (getSelectedCharacterData()?.forms || [])" 
                      :key="form.form_id" 
                      class="form-tile"
                      :class="{ disabled: form.enabled === false }"
                    >
                      <div class="form-tile-image">
                        <img v-if="form.reference_image" :src="getFormImageUrl(getSelectedCharacterData()!.name, form.form_id)" alt="">
                        <div v-else class="form-tile-placeholder">
                          <span>📷</span>
                          <p>未上传</p>
                        </div>
                        <label class="form-upload-overlay">
                          <span>{{ form.reference_image ? '更换' : '上传' }}</span>
                          <input type="file" accept="image/*" hidden @change="uploadFormImage(getSelectedCharacterData()!.name, form.form_id, $event)">
                        </label>
                      </div>
                      <div class="form-tile-info">
                        <div class="form-tile-name">
                          {{ form.form_name }}
                          <span v-if="form.enabled === false" class="disabled-tag">禁用</span>
                        </div>
                        <p v-if="form.description" class="form-tile-desc">{{ form.description }}</p>
                      </div>
                      <div class="form-tile-actions">
                        <label class="toggle-switch-mini" title="启用/禁用形态">
                          <input 
                            type="checkbox" 
                            :checked="form.enabled !== false"
                            @change="handleToggleForm(getSelectedCharacterData()!.name, form.form_id, ($event.target as HTMLInputElement).checked)"
                          >
                          <span class="toggle-slider"></span>
                        </label>
                        <button class="mini-btn primary" @click="openOrthoDialog(getSelectedCharacterData()!, form.form_id, form.form_name)" title="生成三视图">🎨</button>
                        <button v-if="form.reference_image" class="mini-btn" @click="deleteFormImageAction(getSelectedCharacterData()!.name, form.form_id)">🗑️</button>
                        <button class="mini-btn" @click="openEditFormDialog(getSelectedCharacterData()!.name, form)">编辑</button>
                        <button class="mini-btn danger" @click="deleteForm(getSelectedCharacterData()!.name, form)">删除</button>
                      </div>
                    </div>
                  </div>
                </div>
              </template>
              
              <div v-else class="detail-empty">
                <div class="detail-empty-icon">👈</div>
                <p>选择左侧角色查看详情</p>
              </div>
            </div>
          </div>
        </div>
        
        <div class="actions">
          <button 
            v-if="chapterScript || pages.length > 0"
            class="btn danger" 
            :disabled="isLoading"
            @click="clearAndRestart"
          >
            🗑️ 清除数据重新开始
          </button>
          <button 
            class="btn primary" 
            :disabled="!canProceedToScript || isLoading"
            @click="goToStep(1)"
          >
            下一步：生成脚本 →
          </button>
        </div>
      </div>
      
      <!-- 步骤2: 脚本 -->
      <div v-show="currentStep === 1" class="step-panel">
        <h3>📜 全话脚本</h3>
        
        <div v-if="!chapterScript" class="generate-prompt">
          <p>根据已分析的漫画内容和您的设置，生成续写脚本。</p>
          <button 
            class="btn primary" 
            :disabled="isGeneratingScript"
            @click="generateScript"
          >
            {{ isGeneratingScript ? '生成中...' : '🎬 生成脚本' }}
          </button>
        </div>
        
        <div v-else class="script-editor">
          <div class="script-header">
            <h4>{{ chapterScript.chapter_title }}</h4>
            <span class="meta">共 {{ chapterScript.page_count }} 页</span>
          </div>
          <textarea 
            v-model="chapterScript.script_text" 
            class="script-textarea"
            rows="20"
          ></textarea>
          <div class="script-actions">
            <button class="btn secondary" @click="generateScript" :disabled="isGeneratingScript">
              🔄 重新生成
            </button>
          </div>
        </div>
        
        <div class="actions">
          <button class="btn secondary" @click="goToStep(0)">← 上一步</button>
          <button 
            class="btn primary" 
            :disabled="!canProceedToPages"
            @click="goToStep(2)"
          >
            下一步：页面细化 →
          </button>
        </div>
      </div>
      
      <!-- 步骤3: 页面 -->
      <div v-show="currentStep === 2" class="step-panel">
        <h3>📄 页面剧情</h3>
        
        <div v-if="pages.length === 0" class="generate-prompt">
          <p>将脚本拆分为每页详细内容和生图提示词。</p>
          <button 
            class="btn primary" 
            :disabled="isGeneratingPages"
            @click="generatePageDetails"
          >
            {{ isGeneratingPages ? '生成中...' : '📑 生成页面详情' }}
          </button>
        </div>
        
        <div v-else class="pages-list">
          <div v-for="page in pages" :key="page.page_number" class="page-card">
            <div class="page-header">
              <span class="page-number">第 {{ page.page_number }} 页</span>
            </div>
            
            <div class="page-content">
              <!-- 人物 -->
              <div class="page-field">
                <label>人物：</label>
                <input 
                  type="text" 
                  :value="page.characters.join(', ')" 
                  class="page-input"
                  @change="updateCharacters(page, $event)"
                />
              </div>
              
              <!-- 描述 -->
              <div class="page-field">
                <label>描述：</label>
                <textarea 
                  v-model="page.description" 
                  class="page-textarea"
                  rows="3"
                  @change="onPageDataChange"
                ></textarea>
              </div>
              
              <!-- 对话 -->
              <div v-if="page.dialogues.length" class="page-field">
                <label>对话：</label>
                <div class="page-dialogues-list">
                  <div v-for="(d, i) in page.dialogues" :key="i" class="dialogue-item">
                    <input 
                      type="text" 
                      v-model="d.character" 
                      class="dialogue-speaker"
                      placeholder="角色"
                      @change="onPageDataChange"
                    />
                    <span class="dialogue-sep">:</span>
                    <input 
                      type="text" 
                      v-model="d.text" 
                      class="dialogue-text"
                      placeholder="对话内容"
                      @change="onPageDataChange"
                    />
                  </div>
                </div>
              </div>
            </div>
            
            <!-- 生图提示词 -->
            <div class="page-prompt">
              <div class="prompt-header">
                <label>生图提示词：</label>
                <button 
                  class="btn-small"
                  :disabled="regeneratingPromptPage === page.page_number"
                  @click="regenerateSinglePrompt(page.page_number)"
                >
                  {{ regeneratingPromptPage === page.page_number ? '生成中...' : '🔄 重新生成' }}
                </button>
              </div>
              <textarea 
                v-model="page.image_prompt" 
                class="page-textarea prompt-textarea"
                rows="4"
                @change="onPageDataChange"
              ></textarea>
            </div>
          </div>
          
          <div class="pages-actions">
            <button 
              class="btn secondary" 
              :disabled="isGeneratingPrompts"
              @click="generateImagePrompts"
            >
              {{ isGeneratingPrompts ? '生成中...' : '🎨 重新生成提示词' }}
            </button>
            <button 
              class="btn secondary" 
              @click="savePageChanges"
            >
              💾 保存修改
            </button>
          </div>
        </div>
        
        <div class="actions">
          <button class="btn secondary" @click="goToStep(1)">← 上一步</button>
          <button 
            class="btn primary" 
            :disabled="!canProceedToGenerate"
            @click="goToStep(3)"
          >
            下一步：生成图片 →
          </button>
        </div>
      </div>
      
      <!-- 步骤4: 生成 -->
      <div v-show="currentStep === 3" class="step-panel">
        <h3>🎨 图片生成</h3>
        
        <div class="generation-controls">
          <button 
            class="btn primary large" 
            :disabled="isGeneratingImages"
            @click="batchGenerateImages"
          >
            {{ isGeneratingImages ? '生成中...' : '🚀 开始批量生成' }}
          </button>
          <div class="progress-info">
            <span>已完成：{{ generatedPagesCount }} / {{ pages.length }} 页</span>
          </div>
        </div>
        
        <div v-if="isGeneratingImages" class="progress-bar">
          <div class="progress-fill" :style="{ width: generationProgress + '%' }"></div>
        </div>
        
        <div class="generated-pages">
          <div v-for="page in pages" :key="page.page_number" class="generated-card">
            <div class="card-header">
              <span class="page-num">第 {{ page.page_number }} 页</span>
              <span :class="['status', page.status]">
                {{ 
                  page.status === 'pending' ? '等待中' :
                  page.status === 'generating' ? '生成中' :
                  page.status === 'generated' ? '已完成' : '失败'
                }}
              </span>
            </div>
            <div class="card-body">
              <div class="image-area">
                <img v-if="page.image_url" :src="getGeneratedImageUrl(page.image_url)" alt="生成图片">
                <div v-else class="placeholder">
                  {{ page.status === 'generating' ? '⏳ 生成中...' : '等待生成' }}
                </div>
              </div>
              <div class="prompt-area">
                <label>提示词：</label>
                <textarea v-model="page.image_prompt" rows="4" :disabled="page.status === 'generating'"></textarea>
              </div>
            </div>
            <div class="card-actions">
              <button 
                class="btn small" 
                :disabled="page.status === 'generating'"
                @click="regeneratePageImage(page.page_number)"
              >
                🔄 重新生成
              </button>
              <button 
                v-if="page.previous_url" 
                class="btn small secondary"
                @click="usePreviousVersion(page.page_number)"
              >
                ◀ 上一版本
              </button>
            </div>
          </div>
        </div>
        
        <div class="actions">
          <button class="btn secondary" @click="goToStep(2)">← 上一步</button>
          <button 
            class="btn primary" 
            :disabled="!canProceedToExport"
            @click="goToStep(4)"
          >
            下一步：导出 →
          </button>
        </div>
      </div>
      
      <!-- 步骤5: 导出 -->
      <div v-show="currentStep === 4" class="step-panel">
        <h3>📦 导出成品</h3>
        
        <div class="export-options">
          <div class="export-summary">
            <p>共生成 <strong>{{ generatedPagesCount }}</strong> 页图片，可导出为以下格式：</p>
          </div>
          
          <div class="export-formats">
            <div class="format-card" @click="exportFormat = 'images'" :class="{ selected: exportFormat === 'images' }">
              <span class="format-icon">🖼️</span>
              <span class="format-name">图片 ZIP</span>
              <span class="format-desc">所有页面打包下载</span>
            </div>
            <div class="format-card" @click="exportFormat = 'pdf'" :class="{ selected: exportFormat === 'pdf' }">
              <span class="format-icon">📄</span>
              <span class="format-name">PDF 文档</span>
              <span class="format-desc">方便阅读和分享</span>
            </div>
          </div>
          
          <button 
            class="btn primary large" 
            :disabled="isLoading"
            @click="exportFormat === 'images' ? exportAsImages() : exportAsPdf()"
          >
            {{ isLoading ? '导出中...' : '📥 下载' }}
          </button>
        </div>
        
        <div class="actions">
          <button class="btn secondary" @click="goToStep(3)">← 返回生成</button>
        </div>
      </div>
    </div>
    
    <!-- 三视图生成弹窗 -->
    <div v-if="showOrthoDialog" class="modal-overlay" @click.self="closeOrthoDialog">
      <div class="modal-dialog ortho-dialog">
        <div class="modal-header">
          <h3>🎨 生成三视图 - {{ orthoCharacter?.name }} <span v-if="orthoFormName && orthoFormName !== '默认'">({{ orthoFormName }})</span></h3>
          <button class="close-btn" @click="closeOrthoDialog">×</button>
        </div>
        
        <div class="modal-body">
          <div class="ortho-upload-section">
            <label 
              class="upload-area"
              :class="{ 'drag-over': isDraggingOrtho }"
              @dragenter="handleOrthoDragEnter"
              @dragover="handleOrthoDragOver"
              @dragleave="handleOrthoDragLeave"
              @drop="handleOrthoDrop"
            >
              <input 
                type="file" 
                accept="image/*" 
                multiple 
                hidden 
                @change="selectOrthoImages"
              >
              <div class="upload-placeholder">
                <span class="upload-icon">{{ isDraggingOrtho ? '📥' : '📁' }}</span>
                <p v-if="isDraggingOrtho">释放以上传图片</p>
                <p v-else>点击选择或拖拽角色图片（1-5张）</p>
                <p class="hint">可上传多张图片帮助AI理解角色特征</p>
              </div>
            </label>
            
            <div v-if="orthoSourceImages.length > 0" class="source-images">
              <div v-for="(file, index) in orthoSourceImages" :key="index" class="source-image">
                <img :src="createObjectURL(file)" :alt="`源图${index + 1}`">
                <span class="image-index">{{ index + 1 }}</span>
              </div>
            </div>
          </div>
          
          <div v-if="isGeneratingOrtho" class="generating-state">
            <div class="spinner"></div>
            <p class="progress-message">{{ orthoGenerationProgress }}</p>
            <p class="progress-tip">⏱️ AI 生成通常需要 30-60 秒</p>
          </div>
          
          <div v-else-if="orthoResult" class="ortho-result">
            <h4>生成结果：</h4>
            <div class="result-preview">
              <img :src="getOrthoResultUrl()" alt="三视图">
            </div>
          </div>
        </div>
        
        <div class="modal-footer">
          <button class="btn secondary" @click="closeOrthoDialog">取消</button>
          <button 
            v-if="!orthoResult"
            class="btn primary"
            :disabled="orthoSourceImages.length === 0 || isGeneratingOrtho"
            @click="generateOrtho"
          >
            {{ isGeneratingOrtho ? '生成中...' : '🎨 生成三视图' }}
          </button>
          <div v-else class="result-actions">
            <button class="btn secondary" @click="generateOrtho">重新生成</button>
            <button class="btn primary" @click="useOrthoResult">✓ 使用三视图</button>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 编辑角色弹窗 -->
    <div v-if="showEditCharDialog" class="modal-overlay" @click.self="closeEditCharDialog">
      <div class="modal-dialog edit-char-dialog">
        <div class="modal-header">
          <h3>✏️ 编辑角色</h3>
          <button class="close-btn" @click="closeEditCharDialog">×</button>
        </div>
        
        <div class="modal-body">
          <div class="form-group">
            <label>角色名称</label>
            <input 
              v-model="editCharName" 
              type="text" 
              class="form-input"
              placeholder="输入角色主名称"
            >
          </div>
          
          <div class="form-group">
            <label>别名（用逗号分隔）</label>
            <input 
              v-model="editCharAliases" 
              type="text" 
              class="form-input"
              placeholder="例如: 桐乃, 新垣彩世"
            >
            <p class="form-hint">AI生成脚本时可能使用这些名字引用角色</p>
          </div>
        </div>
        
        <div class="modal-footer">
          <button class="btn secondary" @click="closeEditCharDialog">取消</button>
          <button class="btn primary" @click="saveCharacterInfo">💾 保存</button>
        </div>
      </div>
    </div>
    
    <!-- 新增角色弹窗 -->
    <div v-if="showAddCharDialog" class="modal-overlay" @click.self="closeAddCharDialog">
      <div class="modal-dialog add-char-dialog">
        <div class="modal-header">
          <h3>➕ 新增角色</h3>
          <button class="close-btn" @click="closeAddCharDialog">×</button>
        </div>
        
        <div class="modal-body">
          <div class="form-group">
            <label>角色名称 <span class="required">*</span></label>
            <input 
              v-model="newCharName" 
              type="text" 
              class="form-input"
              placeholder="输入角色名称"
            >
          </div>
          
          <div class="form-group">
            <label>别名（用逗号分隔，可选）</label>
            <input 
              v-model="newCharAliases" 
              type="text" 
              class="form-input"
              placeholder="例如: 小明, 阿明"
            >
          </div>
          
          <div class="form-group">
            <label>角色描述（可选）</label>
            <textarea 
              v-model="newCharDescription"
              rows="3"
              class="form-input"
              placeholder="简单描述角色的外观特征..."
            ></textarea>
          </div>
        </div>
        
        <div class="modal-footer">
          <button class="btn secondary" @click="closeAddCharDialog">取消</button>
          <button 
            class="btn primary" 
            :disabled="!newCharName.trim() || isAddingChar"
            @click="confirmAddCharacter"
          >
            {{ isAddingChar ? '添加中...' : '✓ 确认添加' }}
          </button>
        </div>
      </div>
    </div>
    
    <!-- 新增形态弹窗 -->
    <div v-if="showAddFormDialog" class="modal-overlay" @click.self="closeAddFormDialog">
      <div class="modal-dialog add-form-dialog">
        <div class="modal-header">
          <h3>➕ 新增形态</h3>
          <button class="close-btn" @click="closeAddFormDialog">×</button>
        </div>
        
        <div class="modal-body">
          <div class="form-group">
            <label>形态ID <span class="required">*</span></label>
            <input 
              v-model="newFormId" 
              type="text" 
              class="form-input"
              placeholder="例如: battle, dark, casual"
            >
            <p class="form-hint">英文标识符，用于系统识别</p>
          </div>
          
          <div class="form-group">
            <label>形态名称 <span class="required">*</span></label>
            <input 
              v-model="newFormName" 
              type="text" 
              class="form-input"
              placeholder="例如: 战斗服、黑化形态"
            >
          </div>
          
          <div class="form-group">
            <label>形态描述（可选）</label>
            <textarea 
              v-model="newFormDescription"
              rows="2"
              class="form-input"
              placeholder="简单描述该形态的特征..."
            ></textarea>
          </div>
        </div>
        
        <div class="modal-footer">
          <button class="btn secondary" @click="closeAddFormDialog">取消</button>
          <button 
            class="btn primary" 
            :disabled="!newFormId.trim() || !newFormName.trim() || isAddingForm"
            @click="confirmAddForm"
          >
            {{ isAddingForm ? '添加中...' : '✓ 确认添加' }}
          </button>
        </div>
      </div>
    </div>
    
    <!-- 编辑形态弹窗 -->
    <div v-if="showEditFormDialog" class="modal-overlay" @click.self="closeEditFormDialog">
      <div class="modal-dialog edit-form-dialog">
        <div class="modal-header">
          <h3>✏️ 编辑形态</h3>
          <button class="close-btn" @click="closeEditFormDialog">×</button>
        </div>
        
        <div class="modal-body">
          <div class="form-group">
            <label>形态名称</label>
            <input 
              v-model="editFormName" 
              type="text" 
              class="form-input"
              placeholder="形态显示名"
            >
          </div>
          
          <div class="form-group">
            <label>形态描述</label>
            <textarea 
              v-model="editFormDescription"
              rows="2"
              class="form-input"
              placeholder="形态描述..."
            ></textarea>
          </div>
        </div>
        
        <div class="modal-footer">
          <button class="btn secondary" @click="closeEditFormDialog">取消</button>
          <button 
            class="btn primary" 
            :disabled="isSavingForm"
            @click="saveFormInfo"
          >
            {{ isSavingForm ? '保存中...' : '💾 保存' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.continuation-panel {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

/* 消息提示 */
.message {
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 16px;
  font-size: 14px;
}

.message.error {
  background: #fef2f2;
  color: #dc2626;
  border: 1px solid #fecaca;
}

.message.success {
  background: #f0fdf4;
  color: #16a34a;
  border: 1px solid #bbf7d0;
}

/* 步骤指示器 */
.step-indicator {
  display: flex;
  justify-content: center;
  gap: 8px;
  margin-bottom: 24px;
  padding: 16px;
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 12px;
}

.step {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 20px;
  background: var(--bg-primary, #fff);
  border: 2px solid var(--border-color, #e0e0e0);
  transition: all 0.3s;
}

.step.clickable {
  cursor: pointer;
}

.step.clickable:hover {
  border-color: var(--primary, #6366f1);
}

.step.active {
  background: var(--primary, #6366f1);
  border-color: var(--primary, #6366f1);
  color: white;
}

.step.completed {
  background: #22c55e;
  border-color: #22c55e;
  color: white;
}

.step-number {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: rgba(255,255,255,0.2);
  font-weight: bold;
  font-size: 13px;
}

.step:not(.active):not(.completed) .step-number {
  background: var(--bg-secondary, #f5f5f5);
}

.step-name {
  font-size: 14px;
  font-weight: 500;
}

/* 步骤内容 */
.step-content {
  background: var(--bg-primary, #fff);
  border-radius: 12px;
  border: 1px solid var(--border-color, #e0e0e0);
}

.step-panel {
  padding: 24px;
}

.step-panel h3 {
  margin: 0 0 20px 0;
  font-size: 18px;
  font-weight: 600;
}

/* 表单样式 */
.form-group {
  margin-bottom: 16px;
}

.form-group label {
  display: block;
  margin-bottom: 6px;
  font-weight: 500;
  font-size: 14px;
}

.form-group input,
.form-group textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 8px;
  font-size: 14px;
}

.form-group input:focus,
.form-group textarea:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
}

.hint {
  margin-top: 4px;
  font-size: 12px;
  color: var(--text-secondary, #666);
}

/* 角色区域 */
.characters-section {
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid var(--border-color, #e0e0e0);
}

.characters-section h4 {
  margin: 0 0 8px 0;
  font-size: 16px;
}

/* 方格布局 */
.character-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 16px;
  margin-top: 16px;
}

.character-card {
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 12px;
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
}

.character-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.character-preview {
  width: 100%;
  height: 180px;
  background: #e0e0e0;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.character-preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.no-image-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #999;
  height: 100%;
}

.no-image-placeholder span {
  font-size: 48px;
  margin-bottom: 8px;
}

.no-image-placeholder p {
  margin: 0;
  font-size: 14px;
}

.character-details {
  padding: 12px;
}

.character-header {
  margin-bottom: 12px;
}

.character-name {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
  display: block;
}

.character-aliases {
  font-size: 12px;
  color: var(--text-secondary, #666);
  display: block;
  margin-top: 4px;
}

.character-name-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.edit-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 14px;
  padding: 2px 4px;
  border-radius: 4px;
  opacity: 0.6;
  transition: opacity 0.2s, background 0.2s;
}

.edit-btn:hover {
  opacity: 1;
  background: rgba(0, 0, 0, 0.05);
}

/* 区域标题头部 */
.section-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 16px;
}

.section-title h4 {
  margin: 0 0 4px 0;
}

.section-title .hint {
  margin: 0;
}

/* 图标按钮 */
.icon-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 14px;
  padding: 2px 4px;
  border-radius: 4px;
  opacity: 0.6;
  transition: opacity 0.2s, background 0.2s;
}

.icon-btn:hover {
  opacity: 1;
  background: rgba(0, 0, 0, 0.05);
}

.icon-btn.danger:hover {
  background: rgba(220, 53, 69, 0.1);
}

.header-actions {
  display: flex;
  gap: 4px;
}

/* 小号按钮 */
.btn.small {
  padding: 6px 12px;
  font-size: 13px;
}

/* 新增角色弹窗 */
.add-char-dialog {
  max-width: 450px;
}

.required {
  color: #dc3545;
}

.edit-char-dialog {
  max-width: 400px;
}

.form-group {
  margin-bottom: 16px;
}

.form-group label {
  display: block;
  font-weight: 600;
  margin-bottom: 6px;
  font-size: 14px;
}

.form-input {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid #ddd;
  border-radius: 8px;
  font-size: 14px;
  transition: border-color 0.2s;
}

.form-input:focus {
  outline: none;
  border-color: var(--primary-color, #5b6eae);
}

.form-hint {
  font-size: 12px;
  color: var(--text-secondary, #888);
  margin-top: 6px;
}

.character-actions {
  display: flex;
  gap: 8px;
}

.action-btn {
  flex: 1;
  padding: 8px 12px;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
  text-align: center;
}

.action-btn.upload {
  background: var(--primary, #6366f1);
  color: white;
}

.action-btn.upload:hover {
  background: var(--primary-dark, #4f46e5);
}

.action-btn.generate {
  background: transparent;
  color: var(--primary, #6366f1);
  border: 1px solid var(--primary, #6366f1);
}

.action-btn.generate:hover {
  background: var(--primary, #6366f1);
  color: white;
}

/* 按钮 */
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

.btn.primary:hover:not(:disabled) {
  background: var(--primary-dark, #4f46e5);
}

.btn.secondary {
  background: var(--bg-secondary, #f3f4f6);
  color: var(--text-primary, #333);
  border: 1px solid var(--border-color, #e0e0e0);
}

.btn.secondary:hover:not(:disabled) {
  background: var(--bg-hover, #e5e7eb);
}

.btn.danger {
  background: #dc2626;
  color: white;
}

.btn.danger:hover:not(:disabled) {
  background: #b91c1c;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn.large {
  padding: 14px 28px;
  font-size: 16px;
}

.btn.small {
  padding: 6px 12px;
  font-size: 13px;
}

.actions {
  display: flex;
  justify-content: space-between;
  margin-top: 24px;
  padding-top: 20px;
  border-top: 1px solid var(--border-color, #e0e0e0);
}

/* 脚本编辑器 */
.script-editor {
  margin-bottom: 20px;
}

.script-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.script-header h4 {
  margin: 0;
}

.script-textarea {
  width: 100%;
  padding: 16px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 8px;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.6;
  resize: vertical;
}

.script-actions {
  margin-top: 12px;
}

/* 页面卡片 */
.pages-list {
  display: grid;
  gap: 16px;
}

.page-card {
  padding: 16px;
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 8px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 12px;
}

.page-number {
  font-weight: 600;
}

.page-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* 页面字段 */
.page-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.page-field label {
  font-weight: 500;
  font-size: 13px;
  color: var(--text-secondary, #666);
}

.page-input {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 6px;
  font-family: inherit;
  font-size: 14px;
  background: var(--bg-primary, #fff);
}

.page-input:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
}

.page-textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 6px;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.5;
  resize: vertical;
  background: var(--bg-primary, #fff);
  min-height: 60px;
}

.page-textarea:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
}

.prompt-textarea {
  min-height: 80px;
  background: #fffef0;
}

.prompt-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.prompt-header label {
  margin-bottom: 0;
}

.btn-small {
  padding: 4px 10px;
  font-size: 12px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 4px;
  background: var(--bg-primary, #fff);
  color: var(--text-primary, #333);
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-small:hover:not(:disabled) {
  background: var(--bg-hover, #f0f0f0);
  border-color: var(--primary, #6366f1);
  color: var(--primary, #6366f1);
}

.btn-small:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* 对话列表 */
.page-dialogues-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.dialogue-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.dialogue-speaker {
  width: 100px;
  padding: 6px 10px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 4px;
  font-size: 13px;
  font-weight: 500;
  color: var(--primary, #6366f1);
  background: var(--bg-primary, #fff);
}

.dialogue-sep {
  color: var(--text-secondary, #666);
  font-weight: 500;
}

.dialogue-text {
  flex: 1;
  padding: 6px 10px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 4px;
  font-size: 13px;
  background: var(--bg-primary, #fff);
}

.dialogue-speaker:focus,
.dialogue-text:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
}

.page-prompt {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid var(--border-color, #e0e0e0);
}

.page-prompt label {
  display: block;
  margin-bottom: 8px;
  font-weight: 500;
  font-size: 13px;
  color: var(--text-secondary, #666);
}

.pages-actions {
  display: flex;
  gap: 12px;
  margin-top: 16px;
}

/* 生成页面 */
.generation-controls {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 20px;
}

.progress-info {
  font-size: 14px;
  color: var(--text-secondary, #666);
}

.progress-bar {
  height: 8px;
  background: var(--bg-secondary, #e0e0e0);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 20px;
}

.progress-fill {
  height: 100%;
  background: var(--primary, #6366f1);
  transition: width 0.3s;
}

.generated-pages {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 16px;
}

.generated-card {
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 8px;
  overflow: hidden;
}

.card-header {
  display: flex;
  justify-content: space-between;
  padding: 12px 16px;
  background: var(--bg-tertiary, #e5e5e5);
}

.page-num {
  font-weight: 600;
}

.status {
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 10px;
}

.status.pending {
  background: #fef3cd;
  color: #856404;
}

.status.generating {
  background: #cce5ff;
  color: #004085;
}

.status.generated {
  background: #d4edda;
  color: #155724;
}

.status.failed {
  background: #f8d7da;
  color: #721c24;
}

.card-body {
  padding: 16px;
}

.image-area {
  aspect-ratio: 2/3;
  background: #ddd;
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-area img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.placeholder {
  color: #999;
  font-size: 14px;
}

.prompt-area label {
  display: block;
  margin-bottom: 4px;
  font-size: 12px;
  font-weight: 500;
}

.prompt-area textarea {
  width: 100%;
  padding: 8px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 6px;
  font-size: 12px;
  resize: vertical;
}

.card-actions {
  display: flex;
  gap: 8px;
  padding: 12px 16px;
  border-top: 1px solid var(--border-color, #e0e0e0);
}

/* 导出页面 */
.export-options {
  text-align: center;
  padding: 40px 20px;
}

.export-summary {
  margin-bottom: 24px;
  font-size: 16px;
}

.export-formats {
  display: flex;
  justify-content: center;
  gap: 24px;
  margin-bottom: 32px;
}

.format-card {
  padding: 24px 32px;
  background: var(--bg-secondary, #f5f5f5);
  border: 2px solid var(--border-color, #e0e0e0);
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s;
  text-align: center;
}

.format-card:hover {
  border-color: var(--primary, #6366f1);
}

.format-card.selected {
  border-color: var(--primary, #6366f1);
  background: rgba(99, 102, 241, 0.1);
}

.format-icon {
  display: block;
  font-size: 40px;
  margin-bottom: 12px;
}

.format-name {
  display: block;
  font-weight: 600;
  font-size: 16px;
  margin-bottom: 4px;
}

.format-desc {
  display: block;
  font-size: 13px;
  color: var(--text-secondary, #666);
}

/* 通用 */
.empty-state {
  padding: 40px;
  text-align: center;
  color: var(--text-secondary, #666);
}

.generate-prompt {
  text-align: center;
  padding: 40px 20px;
}

.generate-prompt p {
  margin-bottom: 20px;
  color: var(--text-secondary, #666);
}

/* 三视图弹窗样式 */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  animation: fadeIn 0.2s ease-out;
}

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

@keyframes slideUp {
  from { 
    opacity: 0;
    transform: translateY(20px);
  }
  to { 
    opacity: 1;
    transform: translateY(0);
  }
}

.modal-dialog {
  background: white;
  border-radius: 16px;
  max-width: 90%;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
  animation: slideUp 0.3s ease-out;
}

.ortho-dialog {
  width: 600px;
}

/* 新增角色/形态弹窗 - 更宽更美观 */
.add-char-dialog,
.edit-char-dialog,
.add-form-dialog,
.edit-form-dialog {
  width: 480px;
  min-width: 420px;
}

.modal-header {
  padding: 24px 28px 20px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
  border-radius: 16px 16px 0 0;
}

.modal-header h3 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  color: #1a1a2e;
}

.close-btn {
  background: none;
  border: none;
  font-size: 24px;
  color: #999;
  cursor: pointer;
  padding: 0;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  transition: all 0.2s;
}

.close-btn:hover {
  background: rgba(99, 102, 241, 0.1);
  color: #6366f1;
}

.modal-body {
  padding: 28px;
  overflow-y: auto;
  flex: 1;
}

/* 表单组样式优化 */
.modal-body .form-group {
  margin-bottom: 24px;
}

.modal-body .form-group:last-child {
  margin-bottom: 0;
}

.modal-body .form-group label {
  display: block;
  font-size: 14px;
  font-weight: 500;
  color: #374151;
  margin-bottom: 8px;
}

.modal-body .form-group label .required {
  color: #ef4444;
  margin-left: 2px;
}

.modal-body .form-input {
  width: 100%;
  padding: 12px 16px;
  border: 2px solid #e5e7eb;
  border-radius: 10px;
  font-size: 15px;
  transition: all 0.2s;
  background: #fafafa;
  box-sizing: border-box;
}

.modal-body .form-input:focus {
  outline: none;
  border-color: #6366f1;
  background: #fff;
  box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
}

.modal-body .form-input::placeholder {
  color: #9ca3af;
}

.modal-body textarea.form-input {
  resize: vertical;
  min-height: 80px;
  line-height: 1.5;
}

.modal-body .form-hint {
  font-size: 12px;
  color: #9ca3af;
  margin-top: 6px;
  margin-bottom: 0;
}

.modal-footer {
  padding: 20px 28px;
  border-top: 1px solid rgba(0, 0, 0, 0.08);
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  background: #fafbfc;
  border-radius: 0 0 16px 16px;
}

.modal-footer .btn {
  padding: 12px 24px;
  font-size: 14px;
  font-weight: 500;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.2s;
}

.modal-footer .btn.secondary {
  background: #fff;
  border: 2px solid #e5e7eb;
  color: #374151;
}

.modal-footer .btn.secondary:hover {
  background: #f3f4f6;
  border-color: #d1d5db;
}

.modal-footer .btn.primary {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  border: none;
  color: white;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.modal-footer .btn.primary:hover:not(:disabled) {
  background: linear-gradient(135deg, #5558e3 0%, #7c4fe8 100%);
  box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4);
  transform: translateY(-1px);
}

.modal-footer .btn.primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.ortho-upload-section {
  margin-bottom: 24px;
}

.upload-area {
  display: block;
  cursor: pointer;
}

.upload-placeholder {
  border: 2px dashed #ccc;
  border-radius: 12px;
  padding: 40px;
  text-align: center;
  transition: all 0.2s;
}

.upload-area:hover .upload-placeholder {
  border-color: var(--primary, #6366f1);
  background: #f8f9ff;
}

.upload-area.drag-over .upload-placeholder {
  border-color: var(--primary, #6366f1);
  border-style: solid;
  background: rgba(99, 102, 241, 0.1);
  transform: scale(1.02);
}

.upload-icon {
  font-size: 48px;
  display: block;
  margin-bottom: 12px;
}

.upload-placeholder p {
  margin: 8px 0;
  font-size: 14px;
}

.source-images {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 12px;
  margin-top: 16px;
}

.source-image {
  position: relative;
  aspect-ratio: 1;
  border-radius: 8px;
  overflow: hidden;
}

.source-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-index {
  position: absolute;
  top: 4px;
  right: 4px;
  background: rgba(0, 0, 0, 0.6);
  color: white;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
}

.generating-state {
  text-align: center;
  padding: 40px;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid var(--primary, #6366f1);
  border-radius: 50%;
  margin: 0 auto 16px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.progress-message {
  font-size: 16px;
  font-weight: 500;
  color: var(--primary, #6366f1);
  margin-bottom: 8px;
}

.progress-tip {
  font-size: 13px;
  color: #666;
  opacity: 0.8;
}

.ortho-result h4 {
  margin: 0 0 16px 0;
  font-size: 16px;
}

.result-preview {
  border-radius: 8px;
  overflow: hidden;
  background: #f5f5f5;
}

.result-preview img {
  width: 100%;
  display: block;
}

.result-actions {
  display: flex;
  gap: 12px;
}

/* ===== 新的左右分栏布局样式 ===== */
.character-panel-layout {
  display: grid;
  grid-template-columns: 180px 1fr;
  gap: 20px;
  min-height: 320px;
}

/* 左侧角色网格 */
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

/* 禁用徽章 */
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

.disabled-tag {
  display: inline-block;
  background: #ef4444;
  color: white;
  font-size: 10px;
  padding: 1px 4px;
  border-radius: 4px;
  margin-left: 4px;
}

/* 角色/形态禁用状态 */
.character-tile.disabled,
.form-tile.disabled {
  opacity: 0.5;
  filter: grayscale(50%);
}

/* 开关样式 */
.toggle-switch {
  position: relative;
  display: inline-block;
  width: 40px;
  height: 22px;
  cursor: pointer;
}

.toggle-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-switch .toggle-slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #cbd5e1;
  transition: 0.3s;
  border-radius: 22px;
}

.toggle-switch .toggle-slider:before {
  position: absolute;
  content: "";
  height: 16px;
  width: 16px;
  left: 3px;
  bottom: 3px;
  background-color: white;
  transition: 0.3s;
  border-radius: 50%;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}

.toggle-switch input:checked + .toggle-slider {
  background: linear-gradient(135deg, #10b981, #059669);
}

.toggle-switch input:checked + .toggle-slider:before {
  transform: translateX(18px);
}

/* 迷你开关 */
.toggle-switch-mini {
  position: relative;
  display: inline-block;
  width: 28px;
  height: 16px;
  cursor: pointer;
  flex-shrink: 0;
}

.toggle-switch-mini input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-switch-mini .toggle-slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #cbd5e1;
  transition: 0.3s;
  border-radius: 16px;
}

.toggle-switch-mini .toggle-slider:before {
  position: absolute;
  content: "";
  height: 12px;
  width: 12px;
  left: 2px;
  bottom: 2px;
  background-color: white;
  transition: 0.3s;
  border-radius: 50%;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
}

.toggle-switch-mini input:checked + .toggle-slider {
  background: linear-gradient(135deg, #10b981, #059669);
}

.toggle-switch-mini input:checked + .toggle-slider:before {
  transform: translateX(12px);
}

/* 右侧详情面板 */
.character-detail-panel {
  background: linear-gradient(135deg, #fafbff 0%, #f5f7ff 100%);
  border-radius: 16px;
  border: 1px solid #e0e4ff;
  padding: 20px;
  display: flex;
  flex-direction: column;
  min-height: 280px;
}

.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding-bottom: 16px;
  border-bottom: 1px solid #e5e7eb;
  margin-bottom: 16px;
}

.detail-main-info {
  display: flex;
  gap: 14px;
  align-items: center;
}

.detail-avatar {
  width: 64px;
  height: 64px;
  border-radius: 12px;
  overflow: hidden;
  background: #f0f0f0;
  flex-shrink: 0;
}

.detail-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.detail-avatar-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-size: 24px;
  font-weight: 600;
}

.detail-info h4 {
  margin: 0 0 4px 0;
  font-size: 18px;
  font-weight: 600;
  color: #1a1a2e;
}

.detail-aliases {
  margin: 0;
  font-size: 13px;
  color: #6b7280;
}

.detail-actions {
  display: flex;
  gap: 8px;
}

.icon-btn-lg {
  width: 40px;
  height: 40px;
  border: none;
  background: #fff;
  border-radius: 10px;
  cursor: pointer;
  font-size: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
}

.icon-btn-lg:hover {
  background: #f0f2ff;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
}

.icon-btn-lg.danger:hover {
  background: #fef2f2;
}

/* 形态区域 */
.detail-forms-section {
  flex: 1;
}

.forms-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.forms-header h5 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: #374151;
}

.forms-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
  gap: 12px;
}

.form-tile {
  background: #fff;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid #e5e7eb;
  transition: all 0.2s;
}

.form-tile:hover {
  border-color: #c7d2fe;
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.1);
}

.form-tile-image {
  aspect-ratio: 1;
  position: relative;
  background: #f5f5f5;
  overflow: hidden;
}

.form-tile-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.form-tile-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #9ca3af;
}

.form-tile-placeholder span {
  font-size: 28px;
  margin-bottom: 4px;
}

.form-tile-placeholder p {
  margin: 0;
  font-size: 11px;
}

.form-upload-overlay {
  position: absolute;
  inset: 0;
  background: rgba(99, 102, 241, 0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.2s;
  cursor: pointer;
}

.form-upload-overlay span {
  color: white;
  font-size: 13px;
  font-weight: 500;
}

.form-tile-image:hover .form-upload-overlay {
  opacity: 1;
}

.form-tile-info {
  padding: 10px;
}

.form-tile-name {
  font-size: 13px;
  font-weight: 500;
  color: #1f2937;
  display: flex;
  align-items: center;
  gap: 6px;
}

.default-tag {
  font-size: 10px;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  color: white;
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 500;
}

.form-tile-desc {
  margin: 4px 0 0 0;
  font-size: 11px;
  color: #6b7280;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.form-tile-actions {
  padding: 8px 10px;
  border-top: 1px solid #f3f4f6;
  display: flex;
  gap: 6px;
}

.mini-btn {
  flex: 1;
  padding: 5px 8px;
  font-size: 11px;
  border: 1px solid #e5e7eb;
  background: #fff;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.15s;
  color: #374151;
}

.mini-btn:hover {
  background: #f3f4f6;
}

.mini-btn.danger {
  color: #dc2626;
  border-color: #fecaca;
}

.mini-btn.danger:hover {
  background: #fef2f2;
}

/* 空状态 */
.detail-empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #9ca3af;
}

.detail-empty-icon {
  font-size: 48px;
  margin-bottom: 12px;
  opacity: 0.6;
}

.detail-empty p {
  margin: 0;
  font-size: 14px;
}
</style>
