import { sourceUrlForElement, type ImageCandidate } from './discovery'

interface ImageOriginal {
  kind: 'image'
  element: HTMLImageElement
  src: string | null
  srcset: string | null
  pictureSources: Array<{ element: HTMLSourceElement; srcset: string | null }>
  sourceUrl: string | null
}

interface BackgroundOriginal {
  kind: 'background'
  element: HTMLElement
  backgroundImage: string
  sourceUrl: string | null
}

interface CanvasOriginal {
  kind: 'canvas'
  element: HTMLCanvasElement
  dataUrl: string
}

type OriginalBinding = ImageOriginal | BackgroundOriginal | CanvasOriginal

interface ReplacementRecord {
  resultUrl: string
  originals: OriginalBinding[]
  showTranslated: boolean
}

function normalizedSource(value: string): string | null {
  const source = value.trim()
  if (!source) return null
  if (source.startsWith('data:') || source.startsWith('blob:')) return source
  try {
    return new URL(source, document.baseURI).toString()
  } catch {
    return null
  }
}

function capture(element: HTMLElement, sourceUrl: string | null): OriginalBinding {
  if (element instanceof HTMLImageElement) {
    const picture = element.closest('picture')
    return {
      kind: 'image',
      element,
      src: element.getAttribute('src'),
      srcset: element.getAttribute('srcset'),
      pictureSources: picture
        ? [...picture.querySelectorAll<HTMLSourceElement>('source')].map(source => ({
            element: source,
            srcset: source.getAttribute('srcset'),
          }))
        : [],
      sourceUrl: sourceUrl ? normalizedSource(sourceUrl) : null,
    }
  }
  if (element instanceof HTMLCanvasElement) {
    return { kind: 'canvas', element, dataUrl: element.toDataURL('image/png') }
  }
  return {
    kind: 'background',
    element,
    backgroundImage: element.style.backgroundImage,
    sourceUrl: sourceUrl ? normalizedSource(sourceUrl) : null,
  }
}

function bindingMatchesCandidate(binding: HTMLElement, candidate: ImageCandidate): boolean {
  if (candidate.kind === 'canvas') return binding instanceof HTMLCanvasElement
  if (!candidate.sourceUrl) return false
  const expected = normalizedSource(candidate.sourceUrl)
  if (!expected) return false
  const current = sourceUrlForElement(binding)
  return current ? normalizedSource(current) === expected : false
}

function bindingHasNewSource(binding: OriginalBinding, resultUrl: string): boolean {
  if (binding.kind === 'canvas') return false
  const currentSource = sourceUrlForElement(binding.element)
  if (!currentSource) return false
  const current = normalizedSource(currentSource)
  const result = normalizedSource(resultUrl)
  return current !== result && current !== binding.sourceUrl
}

function detachBinding(binding: OriginalBinding): void {
  delete binding.element.dataset.saberTranslated
}

async function drawCanvas(canvas: HTMLCanvasElement, source: string): Promise<void> {
  const image = new Image()
  image.decoding = 'async'
  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve()
    image.onerror = () => reject(new Error('译图加载失败'))
    image.src = source
  })
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Canvas 2D 上下文不可用')
  context.save()
  try {
    context.setTransform(1, 0, 0, 1, 0, 0)
    context.globalAlpha = 1
    context.globalCompositeOperation = 'source-over'
    context.filter = 'none'
    context.shadowColor = 'rgba(0, 0, 0, 0)'
    context.shadowBlur = 0
    context.shadowOffsetX = 0
    context.shadowOffsetY = 0
    context.clearRect(0, 0, canvas.width, canvas.height)
    context.drawImage(image, 0, 0, canvas.width, canvas.height)
  } finally {
    context.restore()
  }
}

async function applyBinding(binding: OriginalBinding, resultUrl: string): Promise<void> {
  if (binding.kind === 'image') {
    for (const source of binding.pictureSources) source.element.removeAttribute('srcset')
    binding.element.removeAttribute('srcset')
    binding.element.src = resultUrl
    binding.element.dataset.saberTranslated = 'true'
    return
  }
  if (binding.kind === 'background') {
    binding.element.style.backgroundImage = `url("${resultUrl.replaceAll('"', '%22')}")`
    binding.element.dataset.saberTranslated = 'true'
    return
  }
  await drawCanvas(binding.element, resultUrl)
  binding.element.dataset.saberTranslated = 'true'
}

async function restoreBinding(binding: OriginalBinding): Promise<void> {
  if (binding.kind === 'image') {
    if (binding.src === null) binding.element.removeAttribute('src')
    else binding.element.setAttribute('src', binding.src)
    if (binding.srcset === null) binding.element.removeAttribute('srcset')
    else binding.element.setAttribute('srcset', binding.srcset)
    for (const source of binding.pictureSources) {
      if (source.srcset === null) source.element.removeAttribute('srcset')
      else source.element.setAttribute('srcset', source.srcset)
    }
    delete binding.element.dataset.saberTranslated
    return
  }
  if (binding.kind === 'background') {
    binding.element.style.backgroundImage = binding.backgroundImage
    delete binding.element.dataset.saberTranslated
    return
  }
  await drawCanvas(binding.element, binding.dataUrl)
  delete binding.element.dataset.saberTranslated
}

function isApplied(binding: OriginalBinding, resultUrl: string): boolean {
  if (binding.element.dataset.saberTranslated !== 'true') return false
  if (binding.kind === 'image') {
    return binding.element.src === resultUrl
      && !binding.element.hasAttribute('srcset')
      && binding.pictureSources.every(source => !source.element.hasAttribute('srcset'))
  }
  if (binding.kind === 'background') {
    return binding.element.style.backgroundImage.includes(resultUrl)
  }
  return true
}

export class ReplacementManager {
  private readonly records = new Map<string, ReplacementRecord>()
  private readonly resultLoads = new Map<string, Promise<void>>()
  private showTranslatedGlobally = true

  async apply(candidate: ImageCandidate, resultUrl: string): Promise<void> {
    await this.loadResult(resultUrl)
    let record = this.records.get(candidate.sourceIdentity)
    if (!record) {
      const originals = candidate.bindings
        .filter(binding => bindingMatchesCandidate(binding, candidate))
        .map(binding => capture(binding, candidate.sourceUrl))
      record = {
        resultUrl,
        originals,
        showTranslated: this.showTranslatedGlobally,
      }
      this.records.set(candidate.sourceIdentity, record)
    } else {
      record.resultUrl = resultUrl
      for (const binding of candidate.bindings) {
        if (!record.originals.some(original => original.element === binding)) {
          if (bindingMatchesCandidate(binding, candidate)) {
            record.originals.push(capture(binding, candidate.sourceUrl))
          }
        }
      }
    }
    if (record.showTranslated) {
      await Promise.all(record.originals.map(binding => applyBinding(binding, resultUrl)))
    }
  }

  async syncBindings(candidate: ImageCandidate): Promise<void> {
    const record = this.records.get(candidate.sourceIdentity)
    if (!record) return
    const added: OriginalBinding[] = []
    for (const binding of candidate.bindings) {
      if (record.originals.some(original => original.element === binding)) continue
      if (!bindingMatchesCandidate(binding, candidate)) continue
      const original = capture(binding, candidate.sourceUrl)
      record.originals.push(original)
      added.push(original)
    }
    if (record.showTranslated && added.length) {
      await Promise.all(added.map(binding => applyBinding(binding, record.resultUrl)))
    }
  }

  async reconcileDisplayedResults(): Promise<void> {
    const reapply: Promise<void>[] = []
    for (const record of this.records.values()) {
      const retained: OriginalBinding[] = []
      for (const binding of record.originals) {
        if (bindingHasNewSource(binding, record.resultUrl)) {
          detachBinding(binding)
          continue
        }
        retained.push(binding)
        if (record.showTranslated && !isApplied(binding, record.resultUrl)) {
          reapply.push(applyBinding(binding, record.resultUrl))
        }
      }
      record.originals = retained
    }
    await Promise.all(reapply)
  }

  async toggleGlobal(): Promise<boolean> {
    this.showTranslatedGlobally = !this.showTranslatedGlobally
    for (const record of this.records.values()) {
      record.showTranslated = this.showTranslatedGlobally
    }
    await Promise.all(
      [...this.records.values()].flatMap(record => (
        this.showTranslatedGlobally
          ? record.originals.map(binding => applyBinding(binding, record.resultUrl))
          : record.originals.map(restoreBinding)
      )),
    )
    return this.showTranslatedGlobally
  }

  async toggle(candidate: ImageCandidate): Promise<boolean | null> {
    const record = this.records.get(candidate.sourceIdentity)
    if (!record) return null
    record.showTranslated = !record.showTranslated
    await Promise.all(
      record.showTranslated
        ? record.originals.map(binding => applyBinding(binding, record.resultUrl))
        : record.originals.map(restoreBinding),
    )
    return record.showTranslated
  }

  async restoreAll(): Promise<void> {
    await Promise.all(
      [...this.records.values()].flatMap(record => record.originals.map(restoreBinding)),
    )
    this.records.clear()
    this.resultLoads.clear()
    this.showTranslatedGlobally = true
  }

  private loadResult(resultUrl: string): Promise<void> {
    const existing = this.resultLoads.get(resultUrl)
    if (existing) return existing
    const pending = new Promise<void>((resolve, reject) => {
      const image = new Image()
      image.decoding = 'async'
      image.onload = () => resolve()
      image.onerror = () => reject(new Error('浏览器无法加载 Saber 译图'))
      image.src = resultUrl
    }).catch((error) => {
      this.resultLoads.delete(resultUrl)
      throw error
    })
    this.resultLoads.set(resultUrl, pending)
    return pending
  }
}
