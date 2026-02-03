/**
 * 漫画续写 API
 * 包含续写配置、脚本生成、图片生成、导出等功能
 */

import { apiClient } from './client'

// ==================== 类型定义 ====================

/**
 * 角色形态
 */
export interface CharacterForm {
    form_id: string           // 形态ID（如 "default", "battle", "dark"）
    form_name: string         // 形态显示名（如 "常服", "战斗服"）
    description: string       // 形态描述
    reference_image: string   // 参考图路径
    enabled?: boolean         // 是否启用此形态
}

/**
 * 角色档案（支持多形态）
 */
export interface CharacterProfile {
    name: string              // 角色名
    aliases: string[]         // 别名列表
    description: string       // 角色基础描述
    forms: CharacterForm[]    // 形态列表
    reference_image: string   // 向后兼容：默认形态的参考图
    enabled?: boolean         // 是否启用此角色
}

/**
 * 角色参考图（向后兼容别名）
 */
export type CharacterRef = CharacterProfile

export interface ChapterScript {
    chapter_title: string
    page_count: number
    script_text: string
    generated_at: string
}

/**
 * 页面角色形态选择
 */
export interface CharacterFormSelection {
    character: string         // 角色名
    form_id: string           // 选择的形态ID
}

export interface PageContent {
    page_number: number
    scene: string  // 已废弃，后端会提供空字符串作为默认值
    characters: string[]                   // 出场人物（向后兼容）
    character_forms?: CharacterFormSelection[]  // 角色形态选择（可选，新增）
    description: string
    dialogues: Array<{ character: string; text: string }>
    mood: string
    image_prompt: string
    image_url: string
    previous_url: string
    status: 'pending' | 'generating' | 'generated' | 'failed'
}

// ==================== 响应类型 ====================

interface SavedContinuationData {
    script: ChapterScript | null
    pages: PageContent[]
    config: {
        page_count?: number
        style_reference_pages?: number
        continuation_direction?: string
    } | null
    has_data: boolean
}

interface PrepareResponse {
    success: boolean
    ready?: boolean
    message?: string
    error?: string
    saved_data?: SavedContinuationData
}

interface CharactersResponse {
    success: boolean
    characters?: CharacterRef[]
    error?: string
}

interface UploadImageResponse {
    success: boolean
    image_path?: string
    error?: string
}

interface ScriptResponse {
    success: boolean
    script?: ChapterScript
    error?: string
}

interface PagesResponse {
    success: boolean
    pages?: PageContent[]
    error?: string
}

interface ImageGenerateResponse {
    success: boolean
    image_path?: string
    pages?: PageContent[]
    session_id?: string
    error?: string
}

// ==================== API 函数 ====================

/**
 * 准备续写数据（检查分析数据是否就绪）
 */
export async function prepareContinuation(bookId: string): Promise<PrepareResponse> {
    return apiClient.get(`/api/manga-insight/${bookId}/continuation/prepare`)
}

/**
 * 获取角色列表
 */
export async function getCharacters(bookId: string): Promise<CharactersResponse> {
    return apiClient.get(`/api/manga-insight/${bookId}/continuation/characters`)
}

/**
 * 新增角色
 */
export async function addCharacter(
    bookId: string,
    data: { name: string; aliases?: string[]; description?: string }
): Promise<{ success: boolean; character?: CharacterRef; error?: string }> {
    return apiClient.post(
        `/api/manga-insight/${bookId}/continuation/characters`,
        data
    )
}

/**
 * 删除角色
 */
export async function deleteCharacter(
    bookId: string,
    characterName: string
): Promise<{ success: boolean; message?: string; error?: string }> {
    return apiClient.delete(
        `/api/manga-insight/${bookId}/continuation/characters/${encodeURIComponent(characterName)}`
    )
}


/**
 * 更新角色信息（名称和别名）
 */
export async function updateCharacterInfo(
    bookId: string,
    characterName: string,
    data: { name?: string; aliases?: string[] }
): Promise<{ success: boolean; character?: CharacterRef; error?: string }> {
    return apiClient.put(
        `/api/manga-insight/${bookId}/continuation/characters/${encodeURIComponent(characterName)}`,
        data
    )
}

// ==================== 形态管理 API ====================

/**
 * 为角色添加新形态
 */
export async function addCharacterForm(
    bookId: string,
    characterName: string,
    data: { form_id: string; form_name: string; description?: string }
): Promise<{ success: boolean; form?: CharacterForm; error?: string }> {
    return apiClient.post(
        `/api/manga-insight/${bookId}/continuation/characters/${encodeURIComponent(characterName)}/forms`,
        data
    )
}

/**
 * 更新角色形态信息
 */
export async function updateCharacterForm(
    bookId: string,
    characterName: string,
    formId: string,
    data: { form_name?: string; description?: string }
): Promise<{ success: boolean; error?: string }> {
    return apiClient.put(
        `/api/manga-insight/${bookId}/continuation/characters/${encodeURIComponent(characterName)}/forms/${encodeURIComponent(formId)}`,
        data
    )
}

/**
 * 删除角色形态
 */
export async function deleteCharacterForm(
    bookId: string,
    characterName: string,
    formId: string
): Promise<{ success: boolean; message?: string; error?: string }> {
    return apiClient.delete(
        `/api/manga-insight/${bookId}/continuation/characters/${encodeURIComponent(characterName)}/forms/${encodeURIComponent(formId)}`
    )
}

/**
 * 切换角色启用状态
 */
export async function toggleCharacterEnabled(
    bookId: string,
    characterName: string,
    enabled: boolean
): Promise<{ success: boolean; enabled?: boolean; error?: string }> {
    return apiClient.post(
        `/api/manga-insight/${bookId}/continuation/characters/${encodeURIComponent(characterName)}/toggle`,
        { enabled }
    )
}

/**
 * 切换角色形态启用状态
 */
export async function toggleFormEnabled(
    bookId: string,
    characterName: string,
    formId: string,
    enabled: boolean
): Promise<{ success: boolean; enabled?: boolean; error?: string }> {
    return apiClient.post(
        `/api/manga-insight/${bookId}/continuation/characters/${encodeURIComponent(characterName)}/forms/${encodeURIComponent(formId)}/toggle`,
        { enabled }
    )
}

/**
 * 为指定形态上传参考图
 */
export async function uploadFormImage(
    bookId: string,
    characterName: string,
    formId: string,
    formData: FormData
): Promise<UploadImageResponse> {
    return apiClient.upload(
        `/api/manga-insight/${bookId}/continuation/characters/${encodeURIComponent(characterName)}/forms/${encodeURIComponent(formId)}/image`,
        formData
    )
}

/**
 * 删除指定形态的参考图
 */
export async function deleteFormImage(
    bookId: string,
    characterName: string,
    formId: string
): Promise<{ success: boolean; error?: string }> {
    return apiClient.delete(
        `/api/manga-insight/${bookId}/continuation/characters/${encodeURIComponent(characterName)}/forms/${encodeURIComponent(formId)}/image`
    )
}

/**
 * 生成形态正交图（三视图）
 */
export async function generateFormOrtho(
    bookId: string,
    characterName: string,
    formId: string,
    sourceImages: File[]
): Promise<UploadImageResponse> {
    const formData = new FormData()
    sourceImages.forEach((file) => {
        formData.append(`images`, file)
    })

    return apiClient.post(
        `/api/manga-insight/${bookId}/continuation/characters/${encodeURIComponent(characterName)}/forms/${encodeURIComponent(formId)}/orthographic`,
        formData,
        {
            headers: { 'Content-Type': 'multipart/form-data' },
            timeout: 0  // AI生图可能很耗时
        }
    )
}

/**
 * 设置形态参考图（使用生成的三视图）
 */
export async function setFormReference(
    bookId: string,
    characterName: string,
    formId: string,
    imagePath: string
): Promise<{ success: boolean; error?: string }> {
    return apiClient.post(
        `/api/manga-insight/${bookId}/continuation/characters/${encodeURIComponent(characterName)}/forms/${encodeURIComponent(formId)}/set-reference`,
        { image_path: imagePath }
    )
}

/**
 * 生成脚本
 */
export async function generateScript(
    bookId: string,
    direction: string,
    pageCount: number
): Promise<ScriptResponse> {
    return apiClient.post(`/api/manga-insight/${bookId}/continuation/script`, {
        direction,
        page_count: pageCount
    }, {
        timeout: 0  // 移除超时限制，LLM 生成可能很耗时
    })
}

/**
 * 保存页面详情（持久化到服务器）
 */
export async function savePages(
    bookId: string,
    pages: PageContent[]
): Promise<{ success: boolean; error?: string }> {
    return apiClient.post(`/api/manga-insight/${bookId}/continuation/save-pages`, {
        pages
    })
}

/**
 * 保存续写配置
 */
export async function saveConfig(
    bookId: string,
    config: {
        page_count: number
        style_reference_pages: number
        continuation_direction: string
    }
): Promise<{ success: boolean; error?: string }> {
    return apiClient.post(`/api/manga-insight/${bookId}/continuation/save-config`, config)
}

/**
 * 清除续写数据（重新开始）
 */
export async function clearContinuationData(
    bookId: string
): Promise<{ success: boolean; message?: string; error?: string }> {
    return apiClient.delete(`/api/manga-insight/${bookId}/continuation/clear`)
}

/**
 * 生成单页详情（推荐使用，避免超时）
 */
export async function generateSinglePageDetails(
    bookId: string,
    script: ChapterScript,
    pageNumber: number
): Promise<{ success: boolean; page?: PageContent; error?: string }> {
    return apiClient.post(`/api/manga-insight/${bookId}/continuation/pages/${pageNumber}`, {
        script
    }, {
        timeout: 0  // 移除超时限制
    })
}

/**
 * 生成图片提示词（批量）
 */
export async function generateImagePrompts(
    bookId: string,
    pages: PageContent[]
): Promise<PagesResponse> {
    return apiClient.post(`/api/manga-insight/${bookId}/continuation/prompts`, {
        pages
    }, {
        timeout: 0  // 移除超时限制，需要为每页生成提示词
    })
}

/**
 * 生成单页提示词（推荐使用，避免超时）
 */
export async function generateSingleImagePrompt(
    bookId: string,
    page: PageContent,
    pageNumber: number
): Promise<{ success: boolean; page?: PageContent; error?: string }> {
    return apiClient.post(`/api/manga-insight/${bookId}/continuation/prompts/${pageNumber}`, {
        page
    }, {
        timeout: 0  // 移除超时限制
    })
}

/**
 * 获取画风参考图路径
 */
export async function getStyleReferences(
    bookId: string,
    count: number = 3
): Promise<{ success: boolean; images?: string[]; error?: string }> {
    return apiClient.get(`/api/manga-insight/${bookId}/continuation/style-references?count=${count}`)
}

/**
 * 生成单页图片
 */
export async function generatePageImage(
    bookId: string,
    pageNumber: number,
    page: PageContent,
    styleRefs: string[],
    sessionId?: string,
    styleRefCount: number = 3  // 画风参考图数量
): Promise<ImageGenerateResponse> {
    return apiClient.post(`/api/manga-insight/${bookId}/continuation/generate/${pageNumber}`, {
        page,
        style_reference_images: styleRefs,
        session_id: sessionId,
        style_ref_count: styleRefCount
    }, {
        timeout: 0  // 移除超时限制，图片生成可能很耗时
    })
}

/**
 * 重新生成页面图片
 */
export async function regeneratePageImage(
    bookId: string,
    pageNumber: number,
    page: PageContent,
    styleRefs: string[],
    sessionId?: string,
    styleRefCount: number = 3  // 画风参考图数量（滑动窗口大小）
): Promise<ImageGenerateResponse> {
    return apiClient.post(`/api/manga-insight/${bookId}/continuation/regenerate/${pageNumber}`, {
        page,
        style_reference_images: styleRefs,
        session_id: sessionId,
        style_ref_count: styleRefCount
    }, {
        timeout: 0  // 移除超时限制
    })
}

/**
 * 导出为图片 ZIP
 * 后端会自动从 pages.json 加载图片路径
 */
export async function exportAsImages(bookId: string): Promise<Blob> {
    const response = await fetch(`/api/manga-insight/${bookId}/continuation/export/images`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.error || '导出失败')
    }

    return response.blob()
}

/**
 * 导出为 PDF
 * 后端会自动从 pages.json 加载图片路径
 */
export async function exportAsPdf(bookId: string): Promise<Blob> {
    const response = await fetch(`/api/manga-insight/${bookId}/continuation/export/pdf`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    })

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.error || '导出失败')
    }

    return response.blob()
}
