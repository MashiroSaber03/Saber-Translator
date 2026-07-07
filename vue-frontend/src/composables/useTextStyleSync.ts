import { ref, watch, computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { useBubbleStore } from '@/stores/bubbleStore'
import { showToast } from '@/utils/toast'
import { useTranslation } from '@/composables/useTranslationPipeline'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'
import { buildEditRenderInput } from '@/composables/edit/editRenderRequest'
import { executeRender } from '@/composables/translation/core/steps'
import { extractBase64Payload } from '@/utils/dataUrl'
import type { BubbleState } from '@/types/bubble'
import type { ImageDataUpdates } from '@/types/image'
import type { SavedTextStyles } from '@/composables/translation/core/types'

export interface ApplySettingsOptions {
    fontSize: boolean
    fontFamily: boolean
    layoutDirection: boolean
    textColor: boolean
    fillColor: boolean
    strokeEnabled: boolean
    strokeColor: boolean
    strokeWidth: boolean
    lineSpacing: boolean
    textAlign: boolean
}

function rgbToHex(rgb: [number, number, number] | null | undefined): string | null {
    if (!rgb || rgb.length !== 3) return null
    const [r, g, b] = rgb
    return '#' + [r, g, b].map(x => {
        const hex = Math.max(0, Math.min(255, Math.round(x))).toString(16)
        return hex.length === 1 ? '0' + hex : hex
    }).join('')
}

export function useTextStyleSync() {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    const bubbleStore = useBubbleStore()
    const translation = useTranslation()

    const currentImage = computed(() => imageStore.currentImage)

    const isSyncingTextStyle = ref(false)

    function syncImageToSidebar(image: typeof imageStore.currentImage) {
        if (!image) return

        const currentStyle = settingsStore.settings.textStyle

        settingsStore.updateTextStyle({
            fontSize: image.fontSize ?? currentStyle.fontSize,
            autoFontSize: image.autoFontSize ?? currentStyle.autoFontSize,
            fontFamily: image.fontFamily ?? currentStyle.fontFamily,
            layoutDirection: image.layoutDirection ?? currentStyle.layoutDirection,
            textColor: image.textColor ?? currentStyle.textColor,
            fillColor: image.fillColor ?? currentStyle.fillColor,
            strokeEnabled: image.strokeEnabled ?? currentStyle.strokeEnabled,
            strokeColor: image.strokeColor ?? currentStyle.strokeColor,
            strokeWidth: image.strokeWidth ?? currentStyle.strokeWidth,
            lineSpacing: image.lineSpacing ?? currentStyle.lineSpacing,
            textAlign: image.textAlign ?? currentStyle.textAlign,
            inpaintMethod: image.inpaintMethod ?? currentStyle.inpaintMethod,
            useAutoTextColor: image.useAutoTextColor ?? currentStyle.useAutoTextColor
        })
    }

    function syncSidebarToImage(style: typeof settingsStore.settings.textStyle) {
        const currentImg = imageStore.currentImage
        if (!currentImg) return

        imageStore.updateCurrentImage({
            fontSize: style.fontSize,
            autoFontSize: style.autoFontSize,
            fontFamily: style.fontFamily,
            layoutDirection: style.layoutDirection,
            textColor: style.textColor,
            fillColor: style.fillColor,
            strokeEnabled: style.strokeEnabled,
            strokeColor: style.strokeColor,
            strokeWidth: style.strokeWidth,
            lineSpacing: style.lineSpacing,
            textAlign: style.textAlign,
            inpaintMethod: style.inpaintMethod,
            useAutoTextColor: style.useAutoTextColor
        })
    }

    function getRenderableBackgroundBase64(image: typeof imageStore.currentImage): string {
        if (!image) return ''
        return extractBase64Payload(image.cleanImageData) || extractBase64Payload(image.originalDataURL)
    }

    function isCurrentImageId(expectedImageId: string): boolean {
        return imageStore.currentImage?.id === expectedImageId
    }

    async function renderWithCurrentBubbleStates(
        bubbleStates: BubbleState[],
        cleanImageBase64: string,
        renderStylePolicy: {
            fontSize: 'preserve' | 'initialize_auto'
            color: 'preserve' | 'initialize_auto'
        } = {
            fontSize: 'preserve',
            color: 'preserve'
        }
    ) {
        return await executeRender(buildEditRenderInput({
            imageIndex: imageStore.currentImageIndex,
            cleanImage: cleanImageBase64,
            bubbleStates,
            settings: settingsStore.settings,
            renderStylePolicy,
        }))
    }

    // 切换图片时，侧边栏展示该图片保存的文字设置。
    watch(
        () => imageStore.currentImage,
        (newImage) => {
            if (newImage && !isSyncingTextStyle.value) {
                isSyncingTextStyle.value = true
                try {
                    syncImageToSidebar(newImage)
                } finally {
                    isSyncingTextStyle.value = false
                }
            }
        },
        { immediate: false } // 不需要立即执行，因为初始化时没有图片
    )

    // 侧边栏文字设置变化时，同步保存到当前图片。
    watch(
        () => settingsStore.settings.textStyle,
        (newStyle) => {
            if (imageStore.currentImage && !isSyncingTextStyle.value) {
                isSyncingTextStyle.value = true
                try {
                    syncSidebarToImage(newStyle)
                } finally {
                    isSyncingTextStyle.value = false
                }
            }
        },
        { deep: true } // 深度监听，因为 textStyle 是对象
    )

    async function handleTextStyleChanged(settingKey: string, newValue: unknown) {
        const image = currentImage.value
        if (!image || !image.translatedDataURL || !image.bubbleStates || image.bubbleStates.length === 0) {
            // 没有已翻译的图片或气泡，不需要重新渲染
            return
        }
        const expectedImageId = image.id

        // 渲染返回时仍需确认当前页，避免慢响应写入切换后的图片。

        // 需要重新渲染的文字样式设置项。
        const renderSettings = ['fontSize', 'fontFamily', 'layoutDirection', 'textColor',
            'strokeEnabled', 'strokeColor', 'strokeWidth', 'fillColor',
            'lineSpacing', 'textAlign']

        if (!renderSettings.includes(settingKey)) {
            return
        }

        // 将设置键映射到气泡状态字段。
        const propertyMap: Record<string, string> = {
            'fontSize': 'fontSize',
            'fontFamily': 'fontFamily',
            'layoutDirection': 'textDirection',  // UI 是 layoutDirection，状态是 textDirection
            'textColor': 'textColor',
            'strokeEnabled': 'strokeEnabled',
            'strokeColor': 'strokeColor',
            'strokeWidth': 'strokeWidth',
            'fillColor': 'fillColor',
            'lineSpacing': 'lineSpacing',
            'textAlign': 'textAlign'
        }

        const stateProperty = propertyMap[settingKey]
        if (stateProperty && image.bubbleStates) {
            // layoutDirection 为 auto 时恢复每个气泡的检测方向。
            if (settingKey === 'layoutDirection') {
                if (newValue === 'auto') {
                    // 切换到"自动"：从备份的 autoTextDirection 恢复到 textDirection
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

        // 触发共享 render step 重渲染，普通样式变更一律保留当前气泡具体值。
        try {
            // 获取最新的 bubbleStates（可能刚刚被更新）
            const latestImage = imageStore.currentImage
            const bubbleStates = latestImage?.bubbleStates || image.bubbleStates || []

            // 检查是否有有效的气泡坐标
            if (bubbleStates.length === 0 || !bubbleStates[0]?.coords) {
                return
            }

            // 提取 clean_image 的 base64 部分，背景兜底策略：clean → original。
            const cleanImageBase64 = getRenderableBackgroundBase64(image)

            if (!cleanImageBase64) {
                return
            }

            const result = await renderWithCurrentBubbleStates(bubbleStates, cleanImageBase64)

            if (!isCurrentImageId(expectedImageId)) {
                return
            }

            if (result.finalImage) {
                imageStore.updateCurrentImage({
                    translatedDataURL: `data:image/png;base64,${result.finalImage}`,
                    bubbleStates: result.bubbleStates,
                    hasUnsavedChanges: true
                })
            }
        } catch {
            showToast('设置变更后重新渲染失败', 'error')
        }
    }

    async function handleAutoFontSizeChanged(isAutoFontSize: boolean) {
        const image = currentImage.value
        if (!image || !image.translatedDataURL) {
            // 没有已翻译的图片，仅影响下次翻译。
            return
        }
        const expectedImageId = image.id

        const bubbleStates = image.bubbleStates
        if (!bubbleStates || !Array.isArray(bubbleStates) || bubbleStates.length === 0) {
            return
        }

        if (isAutoFontSize) {
            // 开启自动字号时显式触发一次字号初始化。

            try {
                const cleanImageBase64 = getRenderableBackgroundBase64(image)

                if (!cleanImageBase64) {
                    return
                }

                const result = await renderWithCurrentBubbleStates(
                    bubbleStates,
                    cleanImageBase64,
                    {
                        fontSize: 'initialize_auto',
                        color: 'preserve'
                    }
                )

                if (!isCurrentImageId(expectedImageId)) {
                    return
                }

                if (result.finalImage) {
                    if (result.bubbleStates && Array.isArray(result.bubbleStates)) {
                        const updatedBubbles = bubbleStates.map((bs, idx) => ({
                            ...bs,
                            fontSize: result.bubbleStates[idx]?.fontSize ?? bs.fontSize
                        }))
                        imageStore.updateCurrentImage({
                            translatedDataURL: `data:image/png;base64,${result.finalImage}`,
                            bubbleStates: updatedBubbles,
                            hasUnsavedChanges: true
                        })
                        bubbleStore.setBubbles(updatedBubbles)
                    } else {
                        imageStore.updateCurrentImage({
                            translatedDataURL: `data:image/png;base64,${result.finalImage}`,
                            hasUnsavedChanges: true
                        })
                    }
                }
            } catch {
                showToast('自动字号渲染失败', 'error')
            }
        } else {
            // 关闭自动字号时将所有气泡设为输入框中的固定字号。
            const fixedFontSize = settingsStore.settings.textStyle.fontSize

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

    async function handleAutoTextColorChanged(isAutoTextColor: boolean) {
        const image = currentImage.value
        if (!image || !image.translatedDataURL) {
            return
        }
        const expectedImageId = image.id

        const bubbleStates = image.bubbleStates
        if (!bubbleStates || !Array.isArray(bubbleStates) || bubbleStates.length === 0) {
            return
        }

        if (!isAutoTextColor) {
            return
        }

        const updatedBubbles = bubbleStates.map((bubble) => ({
            ...bubble,
            textColor: rgbToHex(bubble.autoFgColor) ?? bubble.textColor ?? settingsStore.settings.textStyle.textColor,
            fillColor: rgbToHex(bubble.autoBgColor) ?? bubble.fillColor ?? settingsStore.settings.textStyle.fillColor
        }))

        imageStore.updateCurrentImage({
            bubbleStates: updatedBubbles,
            hasUnsavedChanges: true
        })
        bubbleStore.setBubbles(updatedBubbles)

        try {
            const cleanImageBase64 = getRenderableBackgroundBase64(image)

            if (!cleanImageBase64) {
                return
            }

            const result = await renderWithCurrentBubbleStates(
                updatedBubbles,
                cleanImageBase64,
                {
                    fontSize: 'preserve',
                    color: 'preserve'
                }
            )

            if (!isCurrentImageId(expectedImageId)) {
                return
            }

            if (result.finalImage) {
                imageStore.updateCurrentImage({
                    translatedDataURL: `data:image/png;base64,${result.finalImage}`,
                    bubbleStates: result.bubbleStates,
                    hasUnsavedChanges: true
                })
                bubbleStore.setBubbles(result.bubbleStates)
            }
        } catch {
            showToast('自动文字颜色渲染失败', 'error')
        }
    }

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

        // 先收集有气泡的图片索引，后续只处理可应用的图片。
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
                textColor: textStyle.textColor,
                fillColor: textStyle.fillColor,
                strokeEnabled: textStyle.strokeEnabled,
                strokeColor: textStyle.strokeColor,
                strokeWidth: textStyle.strokeWidth,
                lineSpacing: textStyle.lineSpacing,
                textAlign: textStyle.textAlign,
            }

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

                // 排版设置
                if (options.lineSpacing) updatedBubble.lineSpacing = fixedSettings.lineSpacing
                if (options.textAlign) updatedBubble.textAlign = fixedSettings.textAlign

                return updatedBubble
            }

            // 收集需要重渲染的图片索引（有翻译结果的图片）
            const imagesToReRender: number[] = []

            // 合并遍历：更新气泡状态 + 收集重渲染列表
            for (const i of translatedImageIndices) {
                const image = imageStore.images[i]
                if (!image?.bubbleStates) continue

                // 更新每个气泡的设置
                const updatedBubbleStates = image.bubbleStates.map(applySettingsToBubble)

                // 构建图片级别的设置更新
                const imageUpdates: ImageDataUpdates = { bubbleStates: updatedBubbleStates }

                if (options.fontSize) {
                    imageUpdates.autoFontSize = isAutoFontSize
                    if (!isAutoFontSize) imageUpdates.fontSize = fixedSettings.fontSize
                }

                if (options.fontFamily) imageUpdates.fontFamily = fixedSettings.fontFamily
                if (options.layoutDirection) imageUpdates.layoutDirection = textStyle.layoutDirection

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
                if (options.lineSpacing) imageUpdates.lineSpacing = fixedSettings.lineSpacing
                if (options.textAlign) imageUpdates.textAlign = fixedSettings.textAlign

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
            if (options.textColor) appliedItems.push(isAutoTextColor ? '自动文字颜色' : '文字颜色')
            if (options.fillColor) appliedItems.push(isAutoTextColor ? '自动填充颜色' : '填充颜色')
            if (options.strokeEnabled) appliedItems.push('描边开关')
            if (options.strokeColor) appliedItems.push('描边颜色')
            if (options.strokeWidth) appliedItems.push('描边宽度')
            if (options.lineSpacing) appliedItems.push('行间距')
            if (options.textAlign) appliedItems.push('对齐方式')

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

                let completedCount = 0

                for (const imageIndex of imagesToReRender) {
                    const img = imageStore.images[imageIndex]
                    if (!img?.bubbleStates) continue

                    try {
                        // 背景兜底策略：clean → original
                        const cleanImageBase64 = getRenderableBackgroundBase64(img)

                        if (!cleanImageBase64) {
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
                            textColor: bs.textColor || TEXT_STYLE_DEFAULTS.textColor,
                            bgColor: bs.fillColor || TEXT_STYLE_DEFAULTS.fillColor,
                            autoFgColor: bs.autoFgColor || null,
                            autoBgColor: bs.autoBgColor || null
                        }))

                        // 构建savedTextStyles（从当前图片的设置）
                        const savedTextStyles: SavedTextStyles = {
                            fontSize: img.fontSize,
                            autoFontSize: options.fontSize && isAutoFontSize,
                            fontFamily: img.fontFamily,
                            textDirection: img.layoutDirection,
                            autoTextDirection: textStyle.layoutDirection === 'auto',
                            textColor: img.textColor,
                            fillColor: img.fillColor,
                            strokeEnabled: img.strokeEnabled,
                            strokeColor: img.strokeColor,
                            strokeWidth: img.strokeWidth,
                            lineSpacing: img.lineSpacing ?? textStyle.lineSpacing,
                            textAlign: img.textAlign ?? textStyle.textAlign,
                            inpaintMethod: img.inpaintMethod,
                            useAutoTextColor: img.useAutoTextColor
                        }

                        const result = await executeRender({
                            imageIndex: imageIndex,
                            cleanImage: cleanImageBase64,
                            bubbleCoords: bubbleCoords,
                            bubbleAngles: bubbleAngles,
                            autoDirections: autoDirections,
                            originalTexts: originalTexts,
                            translatedTexts: translatedTexts,
                            textboxTexts: textboxTexts,
                            colors: colors,
                            savedTextStyles: savedTextStyles,
                            currentMode: 'standard',
                            settingsSnapshot: settingsStore.settings,
                            renderStylePolicy: {
                                fontSize: options.fontSize && isAutoFontSize ? 'initialize_auto' : 'preserve',
                                color: 'preserve'
                            }
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
                    } catch {
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
                } finally {
                    isSyncingTextStyle.value = false
                }
            }

            showToast(`已将 ${appliedItems.join('、')} 应用到 ${translatedImageIndices.length} 张图片`, 'success')

        } catch {
            showToast('应用设置失败', 'error')
        }
    }

    return {
        // 同步标志（供外部检测）
        isSyncingTextStyle,

        // 同步函数
        syncImageToSidebar,
        syncSidebarToImage,

        // 文字样式操作
        handleTextStyleChanged,
        handleAutoFontSizeChanged,
        handleAutoTextColorChanged,
        handleApplyToAll,
    }
}
