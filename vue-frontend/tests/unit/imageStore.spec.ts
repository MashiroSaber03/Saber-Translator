import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { setActivePinia, createPinia } from 'pinia'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { useImageStore } from '@/stores/imageStore'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'
import type { ImageDataLoadInput } from '@/types/image'
import { addTestImage } from '../helpers/imageFixtures'

describe('imageStore', () => {
    beforeEach(() => {
        setActivePinia(createPinia())
    })

    it('contains only the backend page load contract without tutorial narration', () => {
        const source = readFileSync(resolve(process.cwd(), 'src/stores/imageStore.ts'), 'utf8')

        expect(source).not.toContain('imageBubbleMirrors')
        expect(source).not.toContain('function applyBubbleStateMirrors')
        expect(source).not.toContain('function addImage(')
        expect(source).not.toContain('function addImages(')
        expect(source).not.toContain('function sortImagesByFileName(')
        for (const staleNarration of [
            '/' + '**',
            '@' + 'param',
            '@' + 'returns',
            '图片数据' + '数组',
            '当前图片' + '索引',
            '如果是第一张' + '图片',
            '按文件路径' + '/文件名',
        ]) {
            expect(source).not.toContain(staleNarration)
        }
    })

    describe('图片管理', () => {
        it('应该能载入后端图片', () => {
            const store = useImageStore()
            const sourceAssetUrl = '/api/v2/assets/source-1'

            const image = addTestImage(store, 'test.png', sourceAssetUrl)

            expect(image.fileName).toBe('test.png')
            expect(image.sourceAssetUrl).toBe(sourceAssetUrl)
            expect(store.imageCount).toBe(1)
            expect(store.currentImageIndex).toBe(0)
        })

        it('应该能更新图片尺寸', () => {
            const store = useImageStore()
            addTestImage(store, 'test.png', '/api/v2/assets/source-1')

            store.updateCurrentImageDimensions(1920, 1080)

            expect(store.currentImage?.width).toBe(1920)
            expect(store.currentImage?.height).toBe(1080)
        })

        it('应该在切换图片时更新 currentImage', () => {
            const store = useImageStore()
            addTestImage(store, 'test1.png', '/api/v2/assets/source-1')
            addTestImage(store, 'test2.png', '/api/v2/assets/source-2')

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
            addTestImage(store, 'test1.png', '/api/v2/assets/source-1')
            addTestImage(store, 'test2.png', '/api/v2/assets/source-2')

            store.deleteCurrentImage()

            expect(store.imageCount).toBe(1)
            expect(store.currentImage?.fileName).toBe('test2.png')
            expect(store.currentImageIndex).toBe(0)
        })

        it('加载缺少样式字段的图片时应补齐统一的文字样式默认值', () => {
            const store = useImageStore()
            const imageWithoutStyleFields: ImageDataLoadInput = {
                id: 'unstyled-image',
                fileName: 'unstyled.png',
                width: 0,
                height: 0,
                sourceAssetUrl: '/api/v2/assets/unstyled',
                translatedAssetUrl: null,
                cleanAssetUrl: null,
                bubbleStates: null,
                translationStatus: 'pending',
                hasUnsavedChanges: false,
            }

            store.setImages([imageWithoutStyleFields])

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

        it('从 failed 切回 processing 时应更新唯一状态字段', () => {
            const store = useImageStore()
            addTestImage(store, 'test.png', '/api/v2/assets/source-1')

            store.setTranslationStatus(0, 'failed')
            expect(store.currentImage?.translationStatus).toBe('failed')

            store.setTranslationStatus(0, 'processing')

            expect(store.currentImage?.translationStatus).toBe('processing')
        })

        it('正常状态变更不应输出常规控制台日志', () => {
            const store = useImageStore()
            const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {})

            try {
                addTestImage(store, 'test1.png', '/api/v2/assets/source-1')
                addTestImage(store, 'test2.png', '/api/v2/assets/source-2')
                store.setCurrentImageIndex(1)
                store.updateCurrentImage({ translatedAssetUrl: '/api/v2/assets/translated' })
                store.updateImageByIndex(0, { translationStatus: 'processing' })
                store.updateCurrentImageDimensions(800, 600)
                store.setTranslationInProgress(true)
                store.deleteCurrentImage()
                store.clearImages()

                expect(logSpy).not.toHaveBeenCalled()
            } finally {
                logSpy.mockRestore()
            }
        })
    })

    describe('尺寸管理', () => {
        it('新图片的尺寸默认为 0', () => {
            const store = useImageStore()
            addTestImage(store, 'test.png', '/api/v2/assets/source-1')

            expect(store.currentImage?.width).toBe(0)
            expect(store.currentImage?.height).toBe(0)
        })

        it('更新尺寸后应该正确存储', () => {
            const store = useImageStore()
            addTestImage(store, 'test.png', '/api/v2/assets/source-1')

            store.updateCurrentImageDimensions(800, 600)

            expect(store.currentImage?.width).toBe(800)
            expect(store.currentImage?.height).toBe(600)
        })
    })
})
