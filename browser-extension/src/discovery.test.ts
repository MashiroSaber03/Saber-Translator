// @vitest-environment jsdom

import { beforeEach, describe, expect, it } from 'vitest'
import {
  adapterFor,
  candidateForSource,
  ruleFromCandidate,
  scanGeneric,
  scanRule,
  validateSuggestedRule,
} from './discovery'

function comicImage(attributes: Record<string, string>): HTMLImageElement {
  const image = document.createElement('img')
  for (const [name, value] of Object.entries(attributes)) image.setAttribute(name, value)
  Object.defineProperties(image, {
    naturalWidth: { configurable: true, value: 900 },
    naturalHeight: { configurable: true, value: 1_400 },
  })
  return image
}

beforeEach(() => {
  document.body.replaceChildren()
  Object.defineProperty(globalThis, 'CSS', {
    configurable: true,
    value: { escape: (value: string) => value.replace(/[^a-z0-9_-]/gi, '\\$&') },
  })
})

describe('comic image discovery', () => {
  it('prefers lazy/high-resolution sources, deduplicates bindings, and keeps DOM order', () => {
    const first = comicImage({
      src: 'https://cdn.example/thumb-1.jpg',
      'data-src': 'https://cdn.example/page-1.webp',
      class: 'comic-page',
    })
    const duplicate = comicImage({
      src: 'https://cdn.example/thumb-copy.jpg',
      'data-src': 'https://cdn.example/page-1.webp',
      class: 'comic-page',
    })
    const second = comicImage({
      src: 'https://cdn.example/fallback.jpg',
      srcset: 'https://cdn.example/page-2-small.webp 600w, https://cdn.example/page-2.webp 1600w',
      class: 'comic-page',
    })
    document.body.append(first, duplicate, second)

    const candidates = scanGeneric()

    expect(candidates).toHaveLength(2)
    expect(candidates[0]?.sourceUrl).toBe('https://cdn.example/page-1.webp')
    expect(candidates[0]?.bindings).toEqual([first, duplicate])
    expect(candidates[1]?.sourceUrl).toBe('https://cdn.example/page-2.webp')
    expect(candidateForSource(candidates, '/page-2.webp')).toBeNull()
    expect(candidateForSource(candidates, 'https://cdn.example/page-2.webp')).toBe(
      candidates[1],
    )
  })

  it('learns a bounded same-kind rule and reports known adapters', () => {
    const reader = document.createElement('main')
    reader.className = 'reader'
    const first = comicImage({ src: '/one.png', class: 'comic-page' })
    const second = comicImage({ src: '/two.png', class: 'comic-page' })
    reader.append(first, second)
    document.body.append(reader)
    const candidate = scanGeneric()[0]
    expect(candidate).toBeDefined()

    const rule = ruleFromCandidate(candidate!)

    expect(rule.kind).toBe('image')
    expect(rule.selector).toContain('comic-page')
    expect(scanRule(rule)).toHaveLength(2)
    expect(adapterFor('mangadex.org')?.name).toBe('MangaDex')
    expect(adapterFor('example.com')).toBeNull()
  })

  it('treats a stale or invalid learned selector as an empty rule', () => {
    expect(scanRule({
      selector: 'main >>> img',
      kind: 'image',
      confirmedAt: Date.now(),
    })).toEqual([])
  })

  it('accepts a DOM Agent selector only when it covers the confirmed nodes', () => {
    const first = comicImage({ src: '/one.png', class: 'comic-page' })
    const second = comicImage({ src: '/two.png', class: 'comic-page' })
    document.body.append(first, second)
    const candidates = scanGeneric()

    const accepted = validateSuggestedRule('img.comic-page', candidates)
    const rejected = validateSuggestedRule('img:first-child', candidates)

    const advertisement = comicImage({ src: '/advertisement.png', class: 'advertisement' })
    document.body.append(advertisement)
    const overlyBroad = validateSuggestedRule('img', candidates)

    expect(accepted?.candidates).toHaveLength(2)
    expect(accepted?.rule.selector).toBe('img.comic-page')
    expect(rejected).toBeNull()
    expect(overlyBroad).toBeNull()
  })

  it('filters small page chrome from generic discovery', () => {
    const icon = document.createElement('img')
    icon.src = '/icon.png'
    Object.defineProperties(icon, {
      naturalWidth: { configurable: true, value: 64 },
      naturalHeight: { configurable: true, value: 64 },
    })
    document.body.append(icon)
    expect(scanGeneric()).toEqual([])
  })

  it('discovers data/blob images and readable CSS backgrounds', () => {
    const root = document.createElement('main')
    const dataImage = comicImage({ src: 'data:image/png;base64,aW1hZ2U=' })
    const blobImage = comicImage({ src: 'blob:https://reader.example/page-2' })
    const background = document.createElement('div')
    background.style.backgroundImage = 'url("https://cdn.example/page-3.webp")'
    background.getBoundingClientRect = () => ({
      width: 800,
      height: 1_200,
      top: 0,
      left: 0,
      right: 800,
      bottom: 1_200,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    })
    root.append(dataImage, blobImage, background)
    document.body.append(root)

    const candidates = scanGeneric()

    expect(candidates.map(candidate => candidate.kind)).toEqual([
      'image',
      'image',
      'background',
    ])
    expect(candidates.map(candidate => candidate.sourceUrl)).toEqual([
      'data:image/png;base64,aW1hZ2U=',
      'blob:https://reader.example/page-2',
      'https://cdn.example/page-3.webp',
    ])
  })

  it('does not rediscover a translated binding as a new source', () => {
    const image = comicImage({ src: '/original.png' })
    image.dataset.saberTranslated = 'true'
    document.body.append(image)
    expect(scanGeneric()).toEqual([])
  })
})
