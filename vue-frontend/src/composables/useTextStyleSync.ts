/**
 * 文字样式同步与应用 composable
 * 
 * 从 TranslateView 提取的文字样式管理逻辑，负责：
 * - 图片 ↔ 侧边栏的双向文字样式同步
 * - 「应用到全部」批量设置功能
 * - 自动字号开关变更处理
 * - 文字样式变更后触发重渲染
 */

import { ref, watch, computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { showToast } from '@/utils/toast'
import { getEffectiveDirection } from '@/types/bubble'
import { useTranslation } from '@/composables/useTranslationPipeline'
import { getPureBase64FromImageSource } from '@/utils/imageBase64'

/**
 * 应用设置选项接口
 */
export interface ApplySettingsOptions {
    fontSize: boolean
    fontFamily: boolean
    layoutDirection: boolean
    lineSpacing: boolean
    textColor: boolean
    fillColor: boolean
    strokeEnabled: boolean
    strokeColor: boolean
    strokeWidth: boolean
}

/**
 * 文字样式同步与应用 composable
 */
export function useTextStyleSync() {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    const bubbleStore = useBubbleStore()
    const translation = useTranslation()

    // 当前图片（计算属性）
    const currentImage = computed(() => imageStore.currentImage)

    // ============================================================
    // 同步标志与辅助函数
    // ============================================================

    /** 【防止循环触发】同步标志 */
    const isSyncingTextStyle = ref(false)

    /**
     * 辅助函数：从图片同步文字设置到侧边栏
     * @param image 源图片数据
     * 【修复】只同步图片中有明确定义的设置，undefined 值回退到当前 settingsStore 设置
     */
    function syncImageToSidebar(image: typeof imageStore.currentImage) {
        if (!image) return

        const currentStyle = settingsStore.settings.textStyle

        settingsStore.updateTextStyle({
            fontSize: image.fontSize ?? currentStyle.fontSize,
            autoFontSize: image.autoFontSize ?? currentStyle.autoFontSize,
            fontFamily: image.fontFamily ?? currentStyle.fontFamily,
            layoutDirection: image.layoutDirection ?? currentStyle.layoutDirection,
            lineSpacing: image.lineSpacing ?? currentStyle.lineSpacing,
            textColor: image.textColor ?? currentStyle.textColor,
            fillColor: image.fillColor ?? currentStyle.fillColor,
            strokeEnabled: image.strokeEnabled ?? currentStyle.strokeEnabled,
            strokeColor: image.strokeColor ?? currentStyle.strokeColor,
            strokeWidth: image.strokeWidth ?? currentStyle.strokeWidth,
            inpaintMethod: image.inpaintMethod ?? currentStyle.inpaintMethod,
            useAutoTextColor: image.useAutoTextColor ?? currentStyle.useAutoTextColor
        })
    }

    /**
     * 辅助函数：从侧边栏同步文字设置到图片
     * @param style 文字样式设置
     */
    function syncSidebarToImage(style: typeof settingsStore.settings.textStyle) {
        const currentImg = imageStore.currentImage
        if (!currentImg) return

        imageStore.updateCurrentImage({
            fontSize: style.fontSize,
            autoFontSize: style.autoFontSize,
            fontFamily: style.fontFamily,
            layoutDirection: style.layoutDirection,
            lineSpacing: style.lineSpacing,
            textColor: style.textColor,
            fillColor: style.fillColor,
            strokeEnabled: style.strokeEnabled,
            strokeColor: style.strokeColor,
            strokeWidth: style.strokeWidth,
            inpaintMethod: style.inpaintMethod,
            useAutoTextColor: style.useAutoTextColor
        })
    }

    // ============================================================
    // 双向同步 watchers
    // ============================================================

    // 【关键修复】监听当前图片变化，同步图片的文字设置到侧边栏
    // 这样切换图片时，侧边栏会显示该图片保存的设置（包括 autoFontSize、layoutDirection 等）
    watch(
        () => imageStore.currentImage,
        (newImage) => {
            if (newImage && !isSyncingTextStyle.value) {
                isSyncingTextStyle.value = true
                try {
                    syncImageToSidebar(newImage)
                    console.log(`[useTextStyleSync] 已同步图片 ${imageStore.currentImageIndex} 的文字设置到侧边栏`)
                } finally {
                    isSyncingTextStyle.value = false
                }
            }
        },
        { immediate: false } // 不需要立即执行，因为初始化时没有图片
    )

    // 【关键修复】监听侧边栏文字设置变化，同步回当前图片
    // 这样用户在侧边栏修改设置时，会保存到当前图片的 ImageData
    watch(
        () => settingsStore.settings.textStyle,
        (newStyle) => {
            if (imageStore.currentImage && !isSyncingTextStyle.value) {
                isSyncingTextStyle.value = true
                try {
                    syncSidebarToImage(newStyle)
                    console.log(`[useTextStyleSync] 已将侧边栏设置同步到图片 ${imageStore.currentImageIndex}`)
                } finally {
                    isSyncingTextStyle.value = false
                }
            }
        },
        { deep: true } // 深度监听，因为 textStyle 是对象
    )

    // ============================================================
    // 处理文字样式设置变更
    // ============================================================

    /**
     * 处理文字样式设置变更
     * 与原版 handleGlobalSettingChange 对应：更新所有气泡的对应参数，然后重新渲染
     * @param settingKey - 变更的设置项
     * @param newValue - 新值
     */
    async function handleTextStyleChanged(settingKey: string, newValue: unknown) {
        const image = currentImage.value
        if (!image || !image.translatedDataURL || !image.bubbleStates || image.bubbleStates.length === 0) {
            // 没有已翻译的图片或气泡，不需要重新渲染
            return
        }

        // 注意：原版有 _isChangingFromSwitchImage 标记来避免切换图片时重渲染
        // Vue 版暂时不实现此检查，因为切换图片时不会触发设置变更事件

        // 需要重新渲染的设置项（与原版 renderSettings 一致）
        const renderSettings = ['fontSize', 'fontFamily', 'layoutDirection', 'textColor',
            'strokeEnabled', 'strokeColor', 'strokeWidth', 'fillColor', 'lineSpacing']

        if (!renderSettings.includes(settingKey)) {
            return
        }

        console.log(`全局设置变更 (${settingKey}=${newValue})，准备重渲染...`)

        // 更新所有气泡的对应属性（与原版 propertyMap 一致）
        const propertyMap: Record<string, string> = {
            'fontSize': 'fontSize',
            'fontFamily': 'fontFamily',
            'layoutDirection': 'textDirection',  // UI 是 layoutDirection，状态是 textDirection
            'textColor': 'textColor',
            'strokeEnabled': 'strokeEnabled',
            'strokeColor': 'strokeColor',
            'strokeWidth': 'strokeWidth',
            'fillColor': 'fillColor',
            'lineSpacing': 'lineSpacing'
        }

        const stateProperty = propertyMap[settingKey]
        if (stateProperty && image.bubbleStates) {
            // 【简化设计】处理 layoutDirection 变更
            if (settingKey === 'layoutDirection') {
                if (newValue === 'auto') {
                    // 切换到"自动"：从备份的 autoTextDirection 恢复到 textDirection
                    console.log("排版方向设置为 'auto'，从 autoTextDirection 恢复每个气泡的排版方向")
                    const updatedBubbles = image.bubbleStates.map(bs => ({
                        ...bs,
                        // 直接用备份的检测结果，不再是 'auto'
                        textDirection: (bs.autoTextDirection === 'vertical' || bs.autoTextDirection === 'horizontal')
                            ? bs.autoTextDirection
                            : 'vertical'
                    }))
                    imageStore.updateCurrentImage({ bubbleStates: updatedBubbles })
                    bubbleStore.setBubbles(updatedBubbles)
                } else {
                    // 切换到强制横排/竖排：直接赋值
                    console.log(`排版方向设置为 '${newValue}'，应用到所有气泡`)
                    const updatedBubbles = image.bubbleStates.map(bs => ({
                        ...bs,
                        textDirection: newValue as 'vertical' | 'horizontal'
                    }))
                    imageStore.updateCurrentImage({ bubbleStates: updatedBubbles })
                    bubbleStore.setBubbles(updatedBubbles)
                }
            } else {
                // 其他设置项：正常更新
                const updatedBubbles = image.bubbleStates.map(bs => ({
                    ...bs,
                    [stateProperty]: newValue
                }))

                // 更新图片的 bubbleStates
                imageStore.updateCurrentImage({ bubbleStates: updatedBubbles })

                // 同步更新 bubbleStore
                bubbleStore.setBubbles(updatedBubbles)
            }
        }

        // 触发重新渲染（调用 reRenderImage API）
        // 后端需要的参数格式：clean_image, bubble_texts, bubble_coords, bubble_states
        try {
            // 获取最新的 bubbleStates（可能刚刚被更新）
            const latestImage = imageStore.currentImage
            const bubbleStates = latestImage?.bubbleStates || image.bubbleStates || []

            // 检查是否有有效的气泡坐标
            if (bubbleStates.length === 0 || !bubbleStates[0]?.coords) {
                console.log('没有有效的气泡坐标，跳过重渲染')
                return
            }

            // 构建 API 参数（与原版 edit_mode.js reRenderFullImage 一致）
            const layoutDir = settingsStore.settings.textStyle.layoutDirection

            // 构建气泡状态数组（与原版 bubbleStatesForApi 格式一致）
            const bubbleStatesForApi = bubbleStates.map((bs) => ({
                translatedText: bs.translatedText || '',
                coords: bs.coords,
                fontSize: bs.fontSize || settingsStore.settings.textStyle.fontSize,
                fontFamily: bs.fontFamily || settingsStore.settings.textStyle.fontFamily,
                textDirection: getEffectiveDirection(bs),
                lineSpacing: typeof bs.lineSpacing === 'number' ? bs.lineSpacing : settingsStore.settings.textStyle.lineSpacing,
                textColor: bs.textColor || settingsStore.settings.textStyle.textColor,
                rotationAngle: bs.rotationAngle || 0,
                position: bs.position || { x: 0, y: 0 },
                strokeEnabled: bs.strokeEnabled ?? settingsStore.settings.textStyle.strokeEnabled,
                strokeColor: bs.strokeColor || settingsStore.settings.textStyle.strokeColor,
                strokeWidth: bs.strokeWidth ?? settingsStore.settings.textStyle.strokeWidth,
            }))

            const bubbleTexts = bubbleStatesForApi.map(s => s.translatedText)
            const bubbleCoords = bubbleStatesForApi.map(s => s.coords)

            // 【修复P1】提取 clean_image 的 base64 部分，原版兜底策略：clean → original
            let cleanImageBase64 = ''
            if (image.cleanImageData) {
                const cleanData = image.cleanImageData
                if (cleanData.startsWith('/api/')) {
                    cleanImageBase64 = (await getPureBase64FromImageSource(cleanData)) || ''
                } else {
                    cleanImageBase64 = cleanData.includes('base64,')
                        ? (cleanData.split('base64,')[1] || '')
                        : cleanData
                }
            } else if (image.originalDataURL) {
                // 兜底：使用原图作为背景
                const base64 = await getPureBase64FromImageSource(image.originalDataURL)
                cleanImageBase64 = base64 || ''
                console.log('handleTextStyleChanged: 使用原图作为背景（兜底）')
            }

            if (!cleanImageBase64) {
                console.log('没有可用的背景图，跳过重渲染')
                return
            }

            const { apiClient } = await import('@/api/client')
            const response = await apiClient.post<{ rendered_image?: string; error?: string }>(
                '/api/re_render_image',
                {
                    clean_image: cleanImageBase64,
                    bubble_texts: bubbleTexts,
                    bubble_coords: bubbleCoords,
                    fontSize: settingsStore.settings.textStyle.fontSize,
                    fontFamily: settingsStore.settings.textStyle.fontFamily,
                    textDirection: layoutDir === 'auto' ? 'vertical' : layoutDir,
                    lineSpacing: settingsStore.settings.textStyle.lineSpacing,
                    textColor: settingsStore.settings.textStyle.textColor,
                    bubble_states: bubbleStatesForApi,
                    use_individual_styles: true,
                    use_inpainting: false,
                    use_lama: false,
                    fillColor: null,
                    is_font_style_change: true,
                    strokeEnabled: settingsStore.settings.textStyle.strokeEnabled,
                    strokeColor: settingsStore.settings.textStyle.strokeColor,
                    strokeWidth: settingsStore.settings.textStyle.strokeWidth,
                }
            )

            if (response.rendered_image) {
                imageStore.updateCurrentImage({
                    translatedDataURL: `data:image/png;base64,${response.rendered_image}`,
                    hasUnsavedChanges: true
                })
                console.log('设置变更后重新渲染成功')
            } else if (response.error) {
                console.error('重新渲染失败:', response.error)
            }
        } catch (error) {
            console.error('设置变更后重新渲染失败:', error)
        }
    }

    // ============================================================
    // 处理自动字号开关变更
    // ============================================================

    /**
     * 处理自动字号开关变更
     * 【复刻原版 events.js handleAutoFontSizeChange】
     * 核心逻辑：
     * - 开启自动字号：调用 reRenderFullImage(..., useAutoFontSize=true) 重新计算字号并渲染
     * - 关闭自动字号：将所有气泡设为输入框中的固定字号，然后渲染
     * @param isAutoFontSize - 自动字号是否启用
     */
    async function handleAutoFontSizeChanged(isAutoFontSize: boolean) {
        const image = currentImage.value
        if (!image || !image.translatedDataURL) {
            // 没有已翻译的图片，仅影响下次翻译（与原版一致）
            console.log(`自动字号设置变更: ${isAutoFontSize} (仅影响下次翻译)`)
            return
        }

        const bubbleStates = image.bubbleStates
        if (!bubbleStates || !Array.isArray(bubbleStates) || bubbleStates.length === 0) {
            console.log('当前图片没有 bubbleStates，跳过重渲染')
            return
        }

        console.log(`自动字号设置变更: ${isAutoFontSize}，将重新渲染...`)

        if (isAutoFontSize) {
            // 【复刻原版】开启自动字号：重新计算每个气泡的字号
            // 原版调用 editMode.reRenderFullImage(false, false, true)
            // 第三个参数 true 表示 useAutoFontSize，对应后端 autoFontSize 参数
            console.log('自动字号已开启，重新计算字号并渲染...')

            try {
                const { apiClient } = await import('@/api/client')

                // 提取 clean_image 的 base64 部分
                let cleanImageBase64 = ''
                if (image.cleanImageData) {
                    const cleanData = image.cleanImageData
                    if (cleanData.startsWith('/api/')) {
                        cleanImageBase64 = (await getPureBase64FromImageSource(cleanData)) || ''
                    } else {
                        cleanImageBase64 = cleanData.includes('base64,')
                            ? (cleanData.split('base64,')[1] || '')
                            : cleanData
                    }
                } else if (image.originalDataURL) {
                    const base64 = await getPureBase64FromImageSource(image.originalDataURL)
                    cleanImageBase64 = base64 || ''
                }

                if (!cleanImageBase64) {
                    console.log('没有可用的背景图，跳过重渲染')
                    return
                }

                const bubbleStatesForApi = bubbleStates.map((bs) => ({
                    translatedText: bs.translatedText || '',
                    coords: bs.coords,
                    fontSize: bs.fontSize || settingsStore.settings.textStyle.fontSize,  // 传递当前字号，后端会根据 autoFontSize=true 重新计算
                    fontFamily: bs.fontFamily || settingsStore.settings.textStyle.fontFamily,
                    textDirection: getEffectiveDirection(bs),
                    textColor: bs.textColor || settingsStore.settings.textStyle.textColor,
                    rotationAngle: bs.rotationAngle || 0,
                    position: bs.position || { x: 0, y: 0 },
                    strokeEnabled: bs.strokeEnabled ?? settingsStore.settings.textStyle.strokeEnabled,
                    strokeColor: bs.strokeColor || settingsStore.settings.textStyle.strokeColor,
                    strokeWidth: bs.strokeWidth ?? settingsStore.settings.textStyle.strokeWidth,
                }))

                const bubbleTexts = bubbleStatesForApi.map(s => s.translatedText)
                const bubbleCoords = bubbleStatesForApi.map(s => s.coords)

                const response = await apiClient.post<{ rendered_image?: string; error?: string; bubble_states?: Array<{ fontSize?: number }> }>(
                    '/api/re_render_image',
                    {
                        clean_image: cleanImageBase64,
                        bubble_texts: bubbleTexts,
                        bubble_coords: bubbleCoords,
                        fontSize: settingsStore.settings.textStyle.fontSize,  // 后端需要数字类型
                        fontFamily: settingsStore.settings.textStyle.fontFamily,
                        textDirection: settingsStore.settings.textStyle.layoutDirection === 'auto' ? 'vertical' : settingsStore.settings.textStyle.layoutDirection,
                        textColor: settingsStore.settings.textStyle.textColor,
                        bubble_states: bubbleStatesForApi,
                        use_individual_styles: true,
                        use_inpainting: false,
                        use_lama: false,
                        fillColor: null,
                        is_font_style_change: true,
                        autoFontSize: true,  // 【修复】使用正确的参数名 autoFontSize（与原版 edit_mode.js 行 407 一致）
                        strokeEnabled: settingsStore.settings.textStyle.strokeEnabled,
                        strokeColor: settingsStore.settings.textStyle.strokeColor,
                        strokeWidth: settingsStore.settings.textStyle.strokeWidth,
                    }
                )

                if (response.rendered_image) {
                    // 【复刻原版】如果后端返回了更新后的 bubble_states，需要回写字号
                    if (response.bubble_states && Array.isArray(response.bubble_states)) {
                        const updatedBubbles = bubbleStates.map((bs, idx) => ({
                            ...bs,
                            fontSize: response.bubble_states![idx]?.fontSize ?? bs.fontSize
                        }))
                        imageStore.updateCurrentImage({
                            translatedDataURL: `data:image/png;base64,${response.rendered_image}`,
                            bubbleStates: updatedBubbles,
                            hasUnsavedChanges: true
                        })
                        bubbleStore.setBubbles(updatedBubbles)
                    } else {
                        imageStore.updateCurrentImage({
                            translatedDataURL: `data:image/png;base64,${response.rendered_image}`,
                            hasUnsavedChanges: true
                        })
                    }
                    console.log('自动字号渲染成功')
                } else if (response.error) {
                    console.error('自动字号渲染失败:', response.error)
                    showToast('重新渲染失败: ' + response.error, 'error')
                }
            } catch (error) {
                console.error('自动字号渲染出错:', error)
            }
        } else {
            // 【复刻原版】关闭自动字号：将所有气泡设为输入框中的固定字号
            const fixedFontSize = settingsStore.settings.textStyle.fontSize
            console.log(`自动字号已关闭，使用固定字号 ${fixedFontSize} 渲染...`)

            // 更新所有气泡的字号
            const updatedBubbles = bubbleStates.map(bs => ({
                ...bs,
                fontSize: fixedFontSize
            }))

            // 更新状态
            imageStore.updateCurrentImage({ bubbleStates: updatedBubbles })
            bubbleStore.setBubbles(updatedBubbles)

            // 触发重渲染（复用 handleTextStyleChanged 的逻辑）
            await handleTextStyleChanged('fontSize', fixedFontSize)
        }
    }

    // ============================================================
    // 处理应用设置到全部
    // ============================================================

    /**
     * 处理应用设置到全部
     * 
     * 【增强版】支持应用自动设置：
     * - 自动排版方向：将每个气泡的 autoTextDirection 应用到 textDirection
     * - 自动文字颜色：将每个气泡的 autoFgColor/autoBgColor 应用到 textColor/fillColor
     * - 自动字号：调用后端重新计算每个气泡的最佳字号
     * 
     * @param options - 选择要应用的设置项
     */
    async function handleApplyToAll(options: ApplySettingsOptions) {
        // 检查是否至少选择了一个选项
        const hasSelectedOption = Object.values(options).some(v => v)
        if (!hasSelectedOption) {
            showToast('请至少选择一个要应用的设置项', 'warning')
            return
        }

        // 检查是否有图片
        if (imageStore.images.length === 0) {
            showToast('没有可应用的图片', 'warning')
            return
        }

        if (imageStore.images.length <= 1) {
            showToast('只有一张图片，无需应用到全部', 'info')
            return
        }

        // 【优化】预先收集有气泡的图片索引，避免多次遍历
        const translatedImageIndices: number[] = []
        for (let i = 0; i < imageStore.images.length; i++) {
            const img = imageStore.images[i]
            if (img?.bubbleStates && img.bubbleStates.length > 0) {
                translatedImageIndices.push(i)
            }
        }

        if (translatedImageIndices.length === 0) {
            showToast('没有已翻译的图片', 'warning')
            return
        }

        try {
            const { textStyle } = settingsStore.settings

            // 检查自动模式设置
            const isAutoLayout = textStyle.layoutDirection === 'auto'
            const isAutoTextColor = textStyle.useAutoTextColor === true
            const isAutoFontSize = textStyle.autoFontSize === true

            // 固定值设置（全部从侧边栏读取）
            const fixedSettings = {
                fontSize: textStyle.fontSize,
                fontFamily: textStyle.fontFamily,
                textDirection: isAutoLayout ? 'vertical' : textStyle.layoutDirection,
                lineSpacing: textStyle.lineSpacing,
                textColor: textStyle.textColor,
                fillColor: textStyle.fillColor,
                strokeEnabled: textStyle.strokeEnabled,
                strokeColor: textStyle.strokeColor,
                strokeWidth: textStyle.strokeWidth,
            }

            /**
             * 辅助函数：RGB数组转十六进制颜色
             */
            const rgbToHex = (rgb: [number, number, number] | null | undefined): string | null => {
                if (!rgb || rgb.length !== 3) return null
                const [r, g, b] = rgb
                return '#' + [r, g, b].map(x => {
                    const hex = Math.max(0, Math.min(255, Math.round(x))).toString(16)
                    return hex.length === 1 ? '0' + hex : hex
                }).join('')
            }

            /**
             * 辅助函数：应用设置到单个气泡
             */
            const applySettingsToBubble = (bubble: typeof bubbleStore.bubbles[0]) => {
                const updatedBubble = { ...bubble }

                // 字号：自动模式下不更新，由后端重新计算；非自动模式使用固定值
                if (options.fontSize && !isAutoFontSize) {
                    updatedBubble.fontSize = fixedSettings.fontSize
                }

                // 字体
                if (options.fontFamily) {
                    updatedBubble.fontFamily = fixedSettings.fontFamily
                }

                // 排版方向
                if (options.layoutDirection) {
                    if (isAutoLayout) {
                        const autoDir = bubble.autoTextDirection
                        updatedBubble.textDirection = (autoDir === 'vertical' || autoDir === 'horizontal')
                            ? autoDir
                            : 'vertical'
                    } else {
                        updatedBubble.textDirection = fixedSettings.textDirection as 'vertical' | 'horizontal'
                    }
                }

                if (options.lineSpacing) {
                    updatedBubble.lineSpacing = fixedSettings.lineSpacing
                }

                // 文字颜色
                if (options.textColor) {
                    if (isAutoTextColor && bubble.autoFgColor) {
                        updatedBubble.textColor = rgbToHex(bubble.autoFgColor) ?? fixedSettings.textColor
                    } else {
                        updatedBubble.textColor = fixedSettings.textColor
                    }
                }

                // 填充颜色
                if (options.fillColor) {
                    if (isAutoTextColor && bubble.autoBgColor) {
                        updatedBubble.fillColor = rgbToHex(bubble.autoBgColor) ?? fixedSettings.fillColor
                    } else {
                        updatedBubble.fillColor = fixedSettings.fillColor
                    }
                }

                // 描边设置
                if (options.strokeEnabled) updatedBubble.strokeEnabled = fixedSettings.strokeEnabled
                if (options.strokeColor) updatedBubble.strokeColor = fixedSettings.strokeColor
                if (options.strokeWidth) updatedBubble.strokeWidth = fixedSettings.strokeWidth

                return updatedBubble
            }

            // 【优化】收集需要重渲染的图片索引（有翻译结果的图片）
            const imagesToReRender: number[] = []

            // 【优化】合并遍历：更新气泡状态 + 收集重渲染列表
            for (const i of translatedImageIndices) {
                const image = imageStore.images[i]
                if (!image?.bubbleStates) continue

                // 更新每个气泡的设置
                const updatedBubbleStates = image.bubbleStates.map(applySettingsToBubble)

                // 构建图片级别的设置更新
                const imageUpdates: any = { bubbleStates: updatedBubbleStates }

                if (options.fontSize) {
                    imageUpdates.autoFontSize = isAutoFontSize
                    if (!isAutoFontSize) imageUpdates.fontSize = fixedSettings.fontSize
                }

                if (options.fontFamily) imageUpdates.fontFamily = fixedSettings.fontFamily
                if (options.layoutDirection) imageUpdates.layoutDirection = textStyle.layoutDirection
                if (options.lineSpacing) imageUpdates.lineSpacing = fixedSettings.lineSpacing

                // 颜色相关设置
                if (options.textColor || options.fillColor) {
                    imageUpdates.useAutoTextColor = isAutoTextColor
                    if (!isAutoTextColor) {
                        if (options.textColor) imageUpdates.textColor = fixedSettings.textColor
                        if (options.fillColor) imageUpdates.fillColor = fixedSettings.fillColor
                    }
                }

                if (options.strokeEnabled) imageUpdates.strokeEnabled = fixedSettings.strokeEnabled
                if (options.strokeColor) imageUpdates.strokeColor = fixedSettings.strokeColor
                if (options.strokeWidth) imageUpdates.strokeWidth = fixedSettings.strokeWidth

                imageStore.updateImageByIndex(i, imageUpdates)

                // 同时收集需要重渲染的图片（有翻译结果的）
                if (image.translatedDataURL) {
                    imagesToReRender.push(i)
                }
            }

            // 同时更新当前气泡 store 中的气泡（如果有）
            if (bubbleStore.bubbles.length > 0) {
                bubbleStore.setBubbles(bubbleStore.bubbles.map(applySettingsToBubble))
            }

            // 构建应用的设置项描述
            const appliedItems: string[] = []
            if (options.fontSize) appliedItems.push(isAutoFontSize ? '自动字号' : '字号')
            if (options.fontFamily) appliedItems.push('字体')
            if (options.layoutDirection) appliedItems.push(isAutoLayout ? '自动排版方向' : '排版方向')
            if (options.lineSpacing) appliedItems.push('行间距')
            if (options.textColor) appliedItems.push(isAutoTextColor ? '自动文字颜色' : '文字颜色')
            if (options.fillColor) appliedItems.push(isAutoTextColor ? '自动填充颜色' : '填充颜色')
            if (options.strokeEnabled) appliedItems.push('描边开关')
            if (options.strokeColor) appliedItems.push('描边颜色')
            if (options.strokeWidth) appliedItems.push('描边宽度')

            // 重新渲染已翻译的图片
            if (imagesToReRender.length > 0) {
                translation.progress.value = {
                    isInProgress: true,
                    current: 0,
                    total: imagesToReRender.length,
                    completed: 0,
                    failed: 0,
                    label: `应用设置中：0 / ${imagesToReRender.length}`,
                    percentage: 0
                }

                // 使用独立的渲染步骤模块
                const { executeRender } = await import('@/composables/translation/core/steps')
                let completedCount = 0

                for (const imageIndex of imagesToReRender) {
                    const img = imageStore.images[imageIndex]
                    if (!img?.bubbleStates) continue

                    try {
                        // 背景兜底策略：clean → original
                        let cleanImageBase64 = ''
                        if (img.cleanImageData) {
                            if (img.cleanImageData.startsWith('/api/')) {
                                cleanImageBase64 = (await getPureBase64FromImageSource(img.cleanImageData)) || ''
                            } else {
                                cleanImageBase64 = img.cleanImageData.includes('base64,')
                                    ? (img.cleanImageData.split('base64,')[1] || '')
                                    : img.cleanImageData
                            }
                        } else if (img.originalDataURL) {
                            const base64 = await getPureBase64FromImageSource(img.originalDataURL)
                            cleanImageBase64 = base64 || ''
                            console.log(`handleApplyToAll: 图片 ${imageIndex} 使用原图作为背景（兜底）`)
                        }

                        if (!cleanImageBase64) {
                            console.log(`handleApplyToAll: 图片 ${imageIndex} 没有可用的背景图，跳过`)
                            translation.progress.value.failed++
                            continue
                        }

                        // 准备渲染数据
                        const bubbleCoords = img.bubbleStates.map(bs => bs.coords)
                        const bubbleAngles = img.bubbleStates.map(bs => bs.rotationAngle || 0)
                        const autoDirections = img.bubbleStates.map(bs => bs.autoTextDirection || 'vertical')
                        const originalTexts = img.bubbleStates.map(bs => bs.originalText || '')
                        const translatedTexts = img.bubbleStates.map(bs => bs.translatedText || '')
                        const textboxTexts = img.bubbleStates.map(bs => bs.textboxText || '')

                        // 构建颜色数据
                        const colors = img.bubbleStates.map(bs => ({
                            textColor: bs.textColor || '#000000',
                            bgColor: bs.fillColor || '#FFFFFF',
                            autoFgColor: bs.autoFgColor || null,
                            autoBgColor: bs.autoBgColor || null
                        }))

                        // 构建savedTextStyles（从当前图片的设置）
                        const savedTextStyles = {
                            fontSize: img.fontSize,
                            autoFontSize: options.fontSize && isAutoFontSize,
                            fontFamily: img.fontFamily,
                            lineSpacing: img.lineSpacing ?? textStyle.lineSpacing,
                            textDirection: img.layoutDirection,
                            autoTextDirection: textStyle.layoutDirection === 'auto',
                            textColor: img.textColor,
                            fillColor: img.fillColor,
                            strokeEnabled: img.strokeEnabled,
                            strokeColor: img.strokeColor,
                            strokeWidth: img.strokeWidth,
                            inpaintMethod: img.inpaintMethod,
                            useAutoTextColor: img.useAutoTextColor
                        }

                        // 调用独立的渲染步骤模块
                        const result = await executeRender({
                            imageIndex: imageIndex,
                            cleanImage: cleanImageBase64,
                            bubbleCoords: bubbleCoords as any,
                            bubbleAngles: bubbleAngles,
                            autoDirections: autoDirections,
                            originalTexts: originalTexts,
                            translatedTexts: translatedTexts,
                            textboxTexts: textboxTexts,
                            colors: colors,
                            savedTextStyles: savedTextStyles as any,
                            currentMode: 'standard'
                        })

                        if (result.finalImage) {
                            imageStore.updateImageByIndex(imageIndex, {
                                translatedDataURL: `data:image/png;base64,${result.finalImage}`,
                                bubbleStates: result.bubbleStates || img.bubbleStates,
                                hasUnsavedChanges: true
                            })

                            completedCount++
                            translation.progress.value.current = completedCount
                            translation.progress.value.label = `应用设置中：${completedCount} / ${imagesToReRender.length}`
                            translation.progress.value.percentage = Math.round((completedCount / imagesToReRender.length) * 100)
                        } else {
                            translation.progress.value.failed++
                        }
                    } catch (err) {
                        console.error(`重渲染图片 ${imageIndex} 失败:`, err)
                        translation.progress.value.failed++
                    }
                }

                translation.progress.value.isInProgress = false
            }

            // 同步当前图片设置到侧边栏
            const currentImg = imageStore.currentImage
            if (currentImg) {
                isSyncingTextStyle.value = true
                try {
                    syncImageToSidebar(currentImg)
                    console.log('[useTextStyleSync] 应用到全部完成，已同步当前图片设置到侧边栏')
                } finally {
                    isSyncingTextStyle.value = false
                }
            }

            showToast(`已将 ${appliedItems.join('、')} 应用到 ${translatedImageIndices.length} 张图片`, 'success')
            console.log(`[useTextStyleSync] 应用设置到全部完成，更新了 ${translatedImageIndices.length} 张图片，重渲染了 ${imagesToReRender.length} 张`)

        } catch (error) {
            console.error('应用设置到全部失败:', error)
            showToast('应用设置失败', 'error')
        }
    }

    // ============================================================
    // 返回
    // ============================================================

    return {
        // 同步标志（供外部检测）
        isSyncingTextStyle,

        // 同步函数
        syncImageToSidebar,
        syncSidebarToImage,

        // 文字样式操作
        handleTextStyleChanged,
        handleAutoFontSizeChanged,
        handleApplyToAll,
    }
}
