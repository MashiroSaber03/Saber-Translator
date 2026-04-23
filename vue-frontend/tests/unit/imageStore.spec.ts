/**
 * imageStore 单元测试
 * 测试关键功能：添加图片、更新尺寸、切换图片等
 */

import { setActivePinia, createPinia } from 'pinia'
import { describe, it, expect, beforeEach } from 'vitest'
import { useImageStore } from '@/stores/imageStore'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'

describe('imageStore', () => {
    beforeEach(() => {
        setActivePinia(createPinia())
    })

    describe('图片管理', () => {
        it('应该能添加图片', () => {
            const store = useImageStore()
            const mockDataURL = 'data:image/png;base64,mockdata'

            const image = store.addImage('test.png', mockDataURL)

            expect(image.fileName).toBe('test.png')
            expect(image.originalDataURL).toBe(mockDataURL)
            expect(store.imageCount).toBe(1)
            expect(store.currentImageIndex).toBe(0)
        })

        it('应该能更新图片尺寸', () => {
            const store = useImageStore()
            store.addImage('test.png', 'data:image/png;base64,mockdata')

            store.updateCurrentImageDimensions(1920, 1080)

            expect(store.currentImage?.width).toBe(1920)
            expect(store.currentImage?.height).toBe(1080)
        })

        it('应该在切换图片时更新 currentImage', () => {
            const store = useImageStore()
            store.addImage('test1.png', 'data:image/png;base64,mockdata1')
            store.addImage('test2.png', 'data:image/png;base64,mockdata2')

            expect(store.currentImage?.fileName).toBe('test1.png')

            store.goToNext()

            expect(store.currentImage?.fileName).toBe('test2.png')
            expect(store.currentImageIndex).toBe(1)
        })
    })

    describe('边界情况', () => {
        it('空列表时 currentImage 应该为 null', () => {
            const store = useImageStore()

            expect(store.currentImage).toBeNull()
            expect(store.imageCount).toBe(0)
        })

        it('删除当前图片后应该正确调整索引', () => {
            const store = useImageStore()
            store.addImage('test1.png', 'data:image/png;base64,mockdata1')
            store.addImage('test2.png', 'data:image/png;base64,mockdata2')

            store.deleteCurrentImage()

            expect(store.imageCount).toBe(1)
            expect(store.currentImage?.fileName).toBe('test2.png')
            expect(store.currentImageIndex).toBe(0)
        })

        it('加载旧会话图片时应补齐统一的文字样式默认值', () => {
            const store = useImageStore()

            store.setImages([
                {
                    id: 'legacy-image',
                    fileName: 'legacy.png',
                    width: 0,
                    height: 0,
                    originalDataURL: 'data:image/png;base64,legacy',
                    translatedDataURL: null,
                    cleanImageData: null,
                    bubbleStates: null,
                    translationStatus: 'pending',
                    translationFailed: false,
                    hasUnsavedChanges: false,
                } as any
            ])

            expect(store.currentImage?.fontSize).toBe(TEXT_STYLE_DEFAULTS.fontSize)
            expect(store.currentImage?.autoFontSize).toBe(TEXT_STYLE_DEFAULTS.autoFontSize)
            expect(store.currentImage?.fontFamily).toBe(TEXT_STYLE_DEFAULTS.fontFamily)
            expect(store.currentImage?.layoutDirection).toBe(TEXT_STYLE_DEFAULTS.layoutDirection)
            expect(store.currentImage?.textColor).toBe(TEXT_STYLE_DEFAULTS.textColor)
            expect(store.currentImage?.fillColor).toBe(TEXT_STYLE_DEFAULTS.fillColor)
            expect(store.currentImage?.inpaintMethod).toBe(TEXT_STYLE_DEFAULTS.inpaintMethod)
            expect(store.currentImage?.strokeEnabled).toBe(TEXT_STYLE_DEFAULTS.strokeEnabled)
            expect(store.currentImage?.strokeColor).toBe(TEXT_STYLE_DEFAULTS.strokeColor)
            expect(store.currentImage?.strokeWidth).toBe(TEXT_STYLE_DEFAULTS.strokeWidth)
            expect(store.currentImage?.lineSpacing).toBe(TEXT_STYLE_DEFAULTS.lineSpacing)
            expect(store.currentImage?.textAlign).toBe(TEXT_STYLE_DEFAULTS.textAlign)
            expect(store.currentImage?.useAutoTextColor).toBe(TEXT_STYLE_DEFAULTS.useAutoTextColor)
        })
    })

    describe('尺寸管理', () => {
        it('新图片的尺寸默认为 0', () => {
            const store = useImageStore()
            store.addImage('test.png', 'data:image/png;base64,mockdata')

            expect(store.currentImage?.width).toBe(0)
            expect(store.currentImage?.height).toBe(0)
        })

        it('更新尺寸后应该正确存储', () => {
            const store = useImageStore()
            store.addImage('test.png', 'data:image/png;base64,mockdata')

            store.updateCurrentImageDimensions(800, 600)

            expect(store.currentImage?.width).toBe(800)
            expect(store.currentImage?.height).toBe(600)
        })
    })
})
