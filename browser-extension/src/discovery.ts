import type { DomNodeSummary, LearnedRule } from './types'

export type CandidateKind = 'image' | 'canvas' | 'background'

export interface ImageCandidate {
  id: string
  kind: CandidateKind
  element: HTMLElement
  bindings: HTMLElement[]
  sourceUrl: string | null
  previewUrl: string | null
  sourceIdentity: string
  width: number
  height: number
}

export interface SiteAdapter {
  name: string
  hosts: RegExp[]
  selectors: string[]
  includeCanvas?: boolean
}

const ADAPTERS: SiteAdapter[] = [
  {
    name: 'MangaDex',
    hosts: [/^(?:www\.)?mangadex\.org$/i],
    selectors: [
      '[data-page] img',
      '.reader--page img',
      'main img[class*="page"]',
      'main img[src*="mangadex"]',
    ],
  },
  {
    name: 'Pixiv Comic',
    hosts: [/^(?:comic\.)?pixiv\.net$/i],
    selectors: ['main img', '[class*="episode"] img', '[class*="viewer"] img'],
  },
  {
    name: 'Shonen Jump+',
    hosts: [/^(?:www\.)?shonenjumpplus\.com$/i],
    selectors: ['[class*="viewer"] img', '[class*="page"] img'],
    includeCanvas: true,
  },
  {
    name: 'ComicWalker',
    hosts: [/^(?:www\.)?comic-walker\.com$/i],
    selectors: ['[class*="viewer"] img', '[class*="page"] img', 'main img'],
    includeCanvas: true,
  },
  {
    name: '漫画柜',
    hosts: [/^(?:www\.)?manhuagui\.com$/i],
    selectors: ['#mangaFile img', '.comic-contain img', '[class*="chapter"] img'],
  },
]

const KNOWN_HOSTS = [
  ...ADAPTERS.flatMap(adapter => adapter.hosts),
  /(^|\.)wn\d+\./i,
  /(^|\.)aman\d*\./i,
]

const NODE_IDS = new WeakMap<Element, string>()
let nodeSequence = 0

function nodeId(element: Element): string {
  const existing = NODE_IDS.get(element)
  if (existing) return existing
  const id = `node-${++nodeSequence}`
  NODE_IDS.set(element, id)
  return id
}

function absoluteUrl(value: string): string | null {
  const normalized = value.trim()
  if (!normalized) return null
  if (normalized.startsWith('data:') || normalized.startsWith('blob:')) return normalized
  try {
    return new URL(normalized, document.baseURI).toString()
  } catch {
    return null
  }
}

function imageSource(image: HTMLImageElement): string | null {
  const srcsetSource = (value: string | null): string | null => {
    if (!value) return null
    const entries = value.split(',').map((entry, index) => {
      const [url, descriptor = ''] = entry.trim().split(/\s+/, 2)
      const score = Number.parseFloat(descriptor) || index + 1
      return { url: url ?? '', score }
    }).filter(entry => entry.url)
    entries.sort((left, right) => right.score - left.score)
    return entries[0]?.url ?? null
  }
  const hints = [
    image.getAttribute('data-original'),
    image.getAttribute('data-src'),
    image.getAttribute('data-lazy-src'),
    image.getAttribute('data-url'),
    srcsetSource(image.getAttribute('data-srcset')),
    srcsetSource(image.getAttribute('srcset')),
    image.currentSrc,
    image.src,
  ]
  for (const hint of hints) {
    if (!hint) continue
    const resolved = absoluteUrl(hint)
    if (resolved && resolved !== location.href) return resolved
  }
  return null
}

function backgroundSource(element: HTMLElement): string | null {
  const value = getComputedStyle(element).backgroundImage
  const match = /url\((?:"|')?(.*?)(?:"|')?\)/i.exec(value)
  return match?.[1] ? absoluteUrl(match[1]) : null
}

export function sourceUrlForElement(element: HTMLElement): string | null {
  if (element instanceof HTMLImageElement) return imageSource(element)
  if (element instanceof HTMLCanvasElement) return null
  return backgroundSource(element)
}

function dimensions(element: HTMLElement): { width: number; height: number } {
  if (element instanceof HTMLImageElement) {
    return {
      width: element.naturalWidth || element.width || element.clientWidth,
      height: element.naturalHeight || element.height || element.clientHeight,
    }
  }
  if (element instanceof HTMLCanvasElement) {
    return { width: element.width, height: element.height }
  }
  const rect = element.getBoundingClientRect()
  return { width: Math.round(rect.width), height: Math.round(rect.height) }
}

function likelyComicSize(width: number, height: number): boolean {
  if (width < 180 || height < 180) return false
  return width * height >= 90_000
}

function candidateFromElement(
  element: HTMLElement,
  { bypassSizeFilter = false }: { bypassSizeFilter?: boolean } = {},
): ImageCandidate | null {
  if (element.dataset.saberTranslated === 'true') return null
  let kind: CandidateKind
  let sourceUrl: string | null
  if (element instanceof HTMLImageElement) {
    kind = 'image'
    sourceUrl = sourceUrlForElement(element)
  } else if (element instanceof HTMLCanvasElement) {
    kind = 'canvas'
    sourceUrl = null
  } else {
    kind = 'background'
    sourceUrl = sourceUrlForElement(element)
  }
  const { width, height } = dimensions(element)
  if (!bypassSizeFilter && !likelyComicSize(width, height)) return null
  if (kind !== 'canvas' && !sourceUrl) return null
  const identity = kind === 'canvas'
    ? `canvas:${nodeId(element)}`
    : `${kind}:${sourceUrl}`
  return {
    id: nodeId(element),
    kind,
    element,
    bindings: [element],
    sourceUrl,
    previewUrl: sourceUrl,
    sourceIdentity: identity,
    width,
    height,
  }
}

function deduplicate(candidates: ImageCandidate[]): ImageCandidate[] {
  const bySource = new Map<string, ImageCandidate>()
  for (const candidate of candidates) {
    const existing = bySource.get(candidate.sourceIdentity)
    if (existing) {
      if (!existing.bindings.includes(candidate.element)) {
        existing.bindings.push(candidate.element)
      }
      continue
    }
    bySource.set(candidate.sourceIdentity, candidate)
  }
  return [...bySource.values()].sort((left, right) => {
    if (left.element === right.element) return 0
    const position = left.element.compareDocumentPosition(right.element)
    return position & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1
  })
}

function elementsForSelectors(selectors: string[]): HTMLElement[] {
  const found: HTMLElement[] = []
  for (const selector of selectors) {
    try {
      for (const element of document.querySelectorAll<HTMLElement>(selector)) {
        if (!found.includes(element)) found.push(element)
      }
    } catch {
      // A stale learned selector is handled as an empty result by the caller.
    }
  }
  return found
}

function backgroundElements(): HTMLElement[] {
  const root = document.querySelector<HTMLElement>('main, article, [role="main"]')
    ?? document.body
  const elements = [root, ...root.querySelectorAll<HTMLElement>('*')].slice(0, 5_000)
  return elements.filter(element => getComputedStyle(element).backgroundImage.includes('url('))
}

export function adapterFor(hostname: string): SiteAdapter | null {
  return ADAPTERS.find(adapter => adapter.hosts.some(pattern => pattern.test(hostname))) ?? null
}

export function isKnownComicHost(hostname: string): boolean {
  return KNOWN_HOSTS.some(pattern => pattern.test(hostname))
}

export function scanAdapter(adapter: SiteAdapter): ImageCandidate[] {
  const elements = elementsForSelectors(adapter.selectors)
  if (adapter.includeCanvas) {
    elements.push(...document.querySelectorAll<HTMLCanvasElement>('canvas'))
  }
  return deduplicate(
    elements
      .map(element => candidateFromElement(element, { bypassSizeFilter: true }))
      .filter((candidate): candidate is ImageCandidate => candidate !== null),
  )
}

export function scanGeneric(): ImageCandidate[] {
  const elements: HTMLElement[] = [
    ...document.querySelectorAll<HTMLImageElement>('img'),
    ...document.querySelectorAll<HTMLCanvasElement>('canvas'),
    ...backgroundElements(),
  ]
  return deduplicate(
    elements
      .map(element => candidateFromElement(element))
      .filter((candidate): candidate is ImageCandidate => candidate !== null),
  )
}

export function scanRule(rule: LearnedRule): ImageCandidate[] {
  return deduplicate(
    elementsForSelectors([rule.selector])
      .map(element => candidateFromElement(element))
      .filter((candidate): candidate is ImageCandidate => (
        candidate !== null && candidate.kind === rule.kind
      )),
  )
}

export function validateSuggestedRule(
  selector: string,
  selected: ImageCandidate[],
): { rule: LearnedRule; candidates: ImageCandidate[] } | null {
  const normalizedSelector = selector.trim()
  const kind = selected[0]?.kind
  if (!normalizedSelector || !kind || selected.some(candidate => candidate.kind !== kind)) {
    return null
  }
  const rule: LearnedRule = {
    selector: normalizedSelector,
    kind,
    confirmedAt: Date.now(),
  }
  const candidates = scanRule(rule)
  if (!candidates.length || candidates.length > 1_000) return null
  const matched = new Set(candidates.map(candidate => candidate.sourceIdentity))
  const confirmed = new Set(selected.map(candidate => candidate.sourceIdentity))
  if (
    selected.some(candidate => !matched.has(candidate.sourceIdentity))
    || matched.size !== confirmed.size
  ) return null
  return { rule, candidates }
}

function safeClassTokens(element: Element): string[] {
  return [...element.classList]
    .filter(value => /^[a-z_][a-z0-9_-]{1,60}$/i.test(value))
    .filter(value => !/[a-f0-9]{8,}/i.test(value))
    .slice(0, 3)
}

function tagSelector(element: Element): string {
  const classes = safeClassTokens(element)
  const suffix = classes.map(value => `.${CSS.escape(value)}`).join('')
  return `${element.tagName.toLowerCase()}${suffix}`
}

export function ruleFromCandidate(candidate: ImageCandidate): LearnedRule {
  const target = candidate.element
  const targetSelector = tagSelector(target)
  const parent = target.parentElement
  const parentSelector = parent ? tagSelector(parent) : ''
  const selector = parentSelector && document.querySelectorAll(
    `${parentSelector} > ${targetSelector}`,
  ).length > 1
    ? `${parentSelector} > ${targetSelector}`
    : targetSelector
  return {
    selector,
    kind: candidate.kind,
    confirmedAt: Date.now(),
  }
}

export function similarTo(candidate: ImageCandidate): ImageCandidate[] {
  return scanRule(ruleFromCandidate(candidate))
}

function parentSummary(element: Element): string {
  const segments: string[] = []
  let current = element.parentElement
  for (let depth = 0; current && depth < 3; depth += 1) {
    segments.push(tagSelector(current))
    current = current.parentElement
  }
  return segments.join(' > ')
}

function sanitizedAttribute(name: string, value: string): string {
  if (!['data-src', 'data-original'].includes(name)) {
    return value.replace(/[\u0000-\u001f\u007f]/g, ' ').slice(0, 160)
  }
  const normalized = value.trim()
  if (normalized.startsWith('data:')) {
    return `data:${normalized.slice(5).split(/[;,]/, 1)[0] || 'image'}`
  }
  if (normalized.startsWith('blob:')) return 'blob:image'
  try {
    const path = new URL(normalized, document.baseURI).pathname
    const extension = /\.[a-z0-9]{2,8}$/i.exec(path)?.[0]?.toLowerCase() ?? ''
    return `image-url${extension}`
  } catch {
    return 'image-source-present'
  }
}

export function domSummary(candidates: ImageCandidate[]): DomNodeSummary[] {
  const allowedAttributes = [
    'alt',
    'role',
    'aria-label',
    'data-src',
    'data-original',
    'data-page',
    'loading',
  ]
  return candidates.slice(0, 600).map((candidate) => {
    const rect = candidate.element.getBoundingClientRect()
    const attributes: Record<string, string> = {}
    for (const name of allowedAttributes) {
      const value = candidate.element.getAttribute(name)
      if (value) attributes[name] = sanitizedAttribute(name, value)
    }
    return {
      id: candidate.id,
      tag: candidate.element.tagName.toLowerCase(),
      classes: safeClassTokens(candidate.element),
      parent: parentSummary(candidate.element),
      attributes,
      rect: {
        width: Math.round(rect.width),
        height: Math.round(rect.height),
        top: Math.round(rect.top + window.scrollY),
        left: Math.round(rect.left + window.scrollX),
      },
      naturalSize: { width: candidate.width, height: candidate.height },
    }
  })
}

export function candidateForSource(
  candidates: ImageCandidate[],
  sourceUrl: string,
): ImageCandidate | null {
  const absolute = absoluteUrl(sourceUrl)
  return candidates.find(candidate => candidate.sourceUrl === absolute) ?? null
}

export function elementCandidate(element: Element): ImageCandidate | null {
  if (!(element instanceof HTMLElement)) return null
  if (
    !(element instanceof HTMLImageElement)
    && !(element instanceof HTMLCanvasElement)
    && !backgroundSource(element)
  ) return null
  return candidateFromElement(element, { bypassSizeFilter: true })
}
