// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ImageCandidate } from './discovery'
import { scanGeneric } from './discovery'
import { ReplacementManager } from './replacement'

function candidate(element: HTMLImageElement, identity = 'image:original'): ImageCandidate {
  return {
    id: identity,
    kind: 'image',
    element,
    bindings: [element],
    sourceUrl: element.src,
    sourceIdentity: identity,
    width: 800,
    height: 1_200,
  }
}

beforeEach(() => {
  document.body.replaceChildren()
  class LoadableImage {
    decoding = ''
    onload: (() => void) | null = null
    onerror: (() => void) | null = null
    set src(_value: string) {
      queueMicrotask(() => this.onload?.())
    }
  }
  vi.stubGlobal('Image', LoadableImage)
})

describe('progressive image replacement', () => {
  it('restores src/srcset and picture sources exactly', async () => {
    const picture = document.createElement('picture')
    const source = document.createElement('source')
    source.srcset = '/original-2x.webp 2x'
    const image = document.createElement('img')
    image.setAttribute('src', '/original.webp')
    image.setAttribute('srcset', '/original.webp 1x')
    picture.append(source, image)
    document.body.append(picture)
    const manager = new ReplacementManager()
    const item = candidate(image)

    await manager.apply(item, 'http://127.0.0.1:5000/result.webp')
    expect(image.dataset.saberTranslated).toBe('true')
    expect(image.src).toBe('http://127.0.0.1:5000/result.webp')
    expect(image.hasAttribute('srcset')).toBe(false)
    expect(source.hasAttribute('srcset')).toBe(false)

    await manager.toggleGlobal()
    expect(image.getAttribute('src')).toBe('/original.webp')
    expect(image.getAttribute('srcset')).toBe('/original.webp 1x')
    expect(source.getAttribute('srcset')).toBe('/original-2x.webp 2x')
  })

  it('applies an existing result to a newly lazy-loaded duplicate binding', async () => {
    const first = document.createElement('img')
    first.src = '/original.webp'
    const second = document.createElement('img')
    second.src = '/original.webp'
    document.body.append(first, second)
    const manager = new ReplacementManager()
    const item = candidate(first)
    await manager.apply(item, 'http://127.0.0.1:5000/result.webp')

    item.bindings.push(second)
    await manager.syncBindings(item)

    expect(second.src).toBe('http://127.0.0.1:5000/result.webp')
    await manager.restoreAll()
    expect(first.getAttribute('src')).toBe('/original.webp')
    expect(second.getAttribute('src')).toBe('/original.webp')
  })

  it('owns the display state for results added after a global toggle', async () => {
    const image = document.createElement('img')
    image.src = '/original.webp'
    document.body.append(image)
    const manager = new ReplacementManager()
    const item = candidate(image)

    expect(await manager.toggleGlobal()).toBe(false)
    expect(await manager.apply(item, 'http://127.0.0.1:5000/result.webp')).toBe(false)
    expect(image.getAttribute('src')).toBe('/original.webp')
    expect(await manager.toggle(item)).toBe(true)
    expect(image.src).toBe('http://127.0.0.1:5000/result.webp')
  })

  it('toggles one completed page without changing another translated page', async () => {
    const first = document.createElement('img')
    first.src = '/page-1.webp'
    const second = document.createElement('img')
    second.src = '/page-2.webp'
    document.body.append(first, second)
    const manager = new ReplacementManager()
    const firstPage = candidate(first, 'image:page-1')
    const secondPage = candidate(second, 'image:page-2')
    await manager.apply(firstPage, 'http://127.0.0.1:5000/result-1.webp')
    await manager.apply(secondPage, 'http://127.0.0.1:5000/result-2.webp')

    expect(await manager.toggle(firstPage)).toBe(false)
    expect(first.getAttribute('src')).toBe('/page-1.webp')
    expect(second.src).toBe('http://127.0.0.1:5000/result-2.webp')
    expect(await manager.toggle(firstPage)).toBe(true)
    expect(first.src).toBe('http://127.0.0.1:5000/result-1.webp')
  })

  it('keeps a data URL intact when validating the bound image source', async () => {
    const image = document.createElement('img')
    image.src = 'data:image/png;base64,aW1hZ2U='
    document.body.append(image)
    const manager = new ReplacementManager()

    await manager.apply(candidate(image, `image:${image.src}`), 'blob:translated')

    expect(image.src).toBe('blob:translated')
    expect(image.dataset.saberTranslated).toBe('true')
  })

  it('reasserts a translated image when a lazy-loader restores the same source', async () => {
    const image = document.createElement('img')
    image.src = '/original.webp'
    document.body.append(image)
    const manager = new ReplacementManager()
    await manager.apply(candidate(image), 'http://127.0.0.1:5000/result.webp')

    image.src = '/original.webp'
    await manager.reconcileDisplayedResults()

    expect(image.src).toBe('http://127.0.0.1:5000/result.webp')
  })

  it('releases an image node when the reader reuses it for a different page', async () => {
    const image = document.createElement('img')
    image.src = '/page-1.webp'
    Object.defineProperties(image, {
      naturalWidth: { configurable: true, value: 800 },
      naturalHeight: { configurable: true, value: 1_200 },
    })
    document.body.append(image)
    const manager = new ReplacementManager()
    const firstPage = candidate(image, 'image:http://localhost:3000/page-1.webp')
    firstPage.sourceUrl = image.src
    await manager.apply(firstPage, 'http://127.0.0.1:5000/result-1.webp')

    image.src = '/page-2.webp'
    await manager.reconcileDisplayedResults()

    expect(image.src).toBe('http://localhost:3000/page-2.webp')
    expect(image.dataset.saberTranslated).toBeUndefined()
    expect(scanGeneric()[0]?.sourceUrl).toBe('http://localhost:3000/page-2.webp')
  })

  it('does not apply a late result after the bound image node changed pages', async () => {
    const image = document.createElement('img')
    image.src = '/page-1.webp'
    document.body.append(image)
    const manager = new ReplacementManager()
    const firstPage = candidate(image, 'image:http://localhost:3000/page-1.webp')
    firstPage.sourceUrl = image.src

    image.src = '/page-2.webp'
    await manager.apply(firstPage, 'http://127.0.0.1:5000/result-1.webp')

    expect(image.src).toBe('http://localhost:3000/page-2.webp')
    expect(image.dataset.saberTranslated).toBeUndefined()
  })

  it('draws and restores readable Canvas pixels with a neutral transform', async () => {
    const canvas = document.createElement('canvas')
    canvas.width = 800
    canvas.height = 1_200
    const context = {
      save: vi.fn(),
      restore: vi.fn(),
      setTransform: vi.fn(),
      clearRect: vi.fn(),
      drawImage: vi.fn(),
      globalAlpha: 0.5,
      globalCompositeOperation: 'multiply',
      filter: 'blur(2px)',
      shadowColor: '#000',
      shadowBlur: 4,
      shadowOffsetX: 2,
      shadowOffsetY: 2,
    } as unknown as CanvasRenderingContext2D
    vi.spyOn(canvas, 'toDataURL').mockReturnValue('data:image/png;base64,b3JpZ2luYWw=')
    vi.spyOn(canvas, 'getContext').mockReturnValue(context)
    document.body.append(canvas)
    const item: ImageCandidate = {
      id: 'canvas-1',
      kind: 'canvas',
      element: canvas,
      bindings: [canvas],
      sourceUrl: null,
      sourceIdentity: 'canvas:canvas-1',
      width: canvas.width,
      height: canvas.height,
    }
    const manager = new ReplacementManager()

    await manager.apply(item, 'blob:translated')
    await manager.toggleGlobal()

    expect(context.setTransform).toHaveBeenCalledWith(1, 0, 0, 1, 0, 0)
    expect(context.drawImage).toHaveBeenCalledTimes(2)
    expect(context.save).toHaveBeenCalledTimes(2)
    expect(context.restore).toHaveBeenCalledTimes(2)
    expect(canvas.dataset.saberTranslated).toBeUndefined()
  })
})
