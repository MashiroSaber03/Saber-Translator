import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  addErasureToUserMask,
  addRestorationToUserMask,
  createInitialUserMask,
} from '@/utils/maskMerger'

type CanvasCall = {
  name: string
  args: unknown[]
  fillStyle?: string
}

const calls: CanvasCall[] = []
let lastImageSrc = ''

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

function createCanvasContext() {
  let fillStyle = ''
  return {
    get fillStyle() {
      return fillStyle
    },
    set fillStyle(value: string) {
      fillStyle = value
      calls.push({ name: 'fillStyle', args: [value] })
    },
    fillRect: vi.fn((...args: unknown[]) => calls.push({ name: 'fillRect', args, fillStyle })),
    drawImage: vi.fn((...args: unknown[]) => calls.push({ name: 'drawImage', args, fillStyle })),
    beginPath: vi.fn(() => calls.push({ name: 'beginPath', args: [], fillStyle })),
    arc: vi.fn((...args: unknown[]) => calls.push({ name: 'arc', args, fillStyle })),
    fill: vi.fn(() => calls.push({ name: 'fill', args: [], fillStyle })),
  }
}

function installCanvasMocks() {
  vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
    if (tagName !== 'canvas') {
      return document.createElement(tagName)
    }

    return {
      width: 0,
      height: 0,
      getContext: vi.fn(() => createCanvasContext()),
      toDataURL: vi.fn(() => 'data:image/png;base64,next-mask'),
    } as unknown as HTMLCanvasElement
  })

  class TestImage {
    onload: (() => void) | null = null
    onerror: ((error: unknown) => void) | null = null

    set src(value: string) {
      lastImageSrc = value
      this.onload?.()
    }
  }

  vi.stubGlobal('Image', TestImage)
}

describe('mask merger utility', () => {
  beforeEach(() => {
    calls.length = 0
    lastImageSrc = ''
    installCanvasMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('creates a gray initial mask and paints white or black brush paths', async () => {
    expect(createInitialUserMask(100, 80)).toBe('next-mask')
    expect(calls).toContainEqual({ name: 'fillStyle', args: ['rgb(127, 127, 127)'] })
    expect(calls).toContainEqual({ name: 'fillRect', args: [0, 0, 100, 80], fillStyle: 'rgb(127, 127, 127)' })

    calls.length = 0
    await expect(addErasureToUserMask('current-mask', 100, 80, [{ x: 4, y: 8 }], 12)).resolves.toBe('next-mask')
    expect(lastImageSrc).toBe('data:image/png;base64,current-mask')
    expect(calls).toContainEqual({ name: 'drawImage', args: [expect.any(Object), 0, 0, 100, 80], fillStyle: '' })
    expect(calls).toContainEqual({ name: 'arc', args: [4, 8, 12, 0, Math.PI * 2], fillStyle: 'white' })
    expect(calls).toContainEqual({ name: 'fill', args: [], fillStyle: 'white' })

    calls.length = 0
    await expect(addRestorationToUserMask('', 100, 80, [{ x: 2, y: 3 }], 5)).resolves.toBe('next-mask')
    expect(calls).toContainEqual({ name: 'fillStyle', args: ['rgb(127, 127, 127)'] })
    expect(calls).toContainEqual({ name: 'arc', args: [2, 3, 5, 0, Math.PI * 2], fillStyle: 'black' })
    expect(calls).toContainEqual({ name: 'fill', args: [], fillStyle: 'black' })
  })

  it('keeps the source compact and routes brush modes through one painter', () => {
    const content = source('src/utils/maskMerger.ts')

    for (const staleNarration of [
      '/**',
      '@param',
      '@returns',
      '掩膜工具',
      '前端职责',
      '初始化用户掩膜',
      '更新用户掩膜',
      '如果没有现有掩膜',
      '绘制现有掩膜',
      '用白色绘制笔刷路径',
      '用黑色绘制笔刷路径',
      '转为 Base64',
    ]) {
      expect(content).not.toContain(staleNarration)
    }

    expect(content).toContain('function paintUserMask')
    expect(content).toContain("return paintUserMask(currentUserMask, width, height, path, radius, 'white')")
    expect(content).toContain("return paintUserMask(currentUserMask, width, height, path, radius, 'black')")
  })
})
