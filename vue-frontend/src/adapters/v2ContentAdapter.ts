import type { V2PageDocument, V2PageSummary } from '@/api/v2/content'
import type {
  BubbleCoords,
  BubbleState,
  BubbleTextline,
  InpaintMethod,
  LogicalAlign,
  PolygonCoords,
  ResolvedTextDirection,
} from '@/types/bubble'
import type { OcrResult } from '@/types/ocr'
import type { ImageDataLoadInput, TranslationStatus } from '@/types/image'

const BUBBLE_PAYLOAD_KEYS = [
  'originalText',
  'translatedText',
  'textboxText',
  'coords',
  'polygon',
  'fontSize',
  'textDirection',
  'autoTextDirection',
  'textColor',
  'fillColor',
  'rotationAngle',
  'position',
  'strokeEnabled',
  'strokeColor',
  'strokeWidth',
  'lineSpacing',
  'inlineAlign',
  'blockAlign',
  'inpaintMethod',
  'autoFgColor',
  'autoBgColor',
  'colorConfidence',
  'textlines',
  'ocrResult',
] as const

const OCR_RESULT_KEYS = [
  'text',
  'confidence',
  'confidenceSupported',
  'engine',
  'primaryEngine',
  'fallbackUsed',
] as const

function exactObject(
  value: unknown,
  keys: readonly string[],
  label: string,
): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error(`${label} 必须是对象`)
  }
  const record = value as Record<string, unknown>
  const actualKeys = Object.keys(record)
  if (
    actualKeys.length !== keys.length
    || actualKeys.some(key => !keys.includes(key))
  ) {
    throw new Error(`${label} 不符合当前数据结构`)
  }
  return record
}

function stringValue(value: unknown, label: string, nonEmpty = false): string {
  if (typeof value !== 'string' || (nonEmpty && value.length === 0)) {
    throw new Error(`${label} 必须是${nonEmpty ? '非空' : ''}字符串`)
  }
  return value
}

function booleanValue(value: unknown, label: string): boolean {
  if (typeof value !== 'boolean') throw new Error(`${label} 必须是布尔值`)
  return value
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`${label} 必须是有限数字`)
  }
  return value
}

function integerValue(value: unknown, label: string, minimum?: number): number {
  const number = finiteNumber(value, label)
  if (!Number.isInteger(number) || (minimum !== undefined && number < minimum)) {
    throw new Error(`${label} 必须是${minimum === undefined ? '' : `不小于 ${minimum} 的`}整数`)
  }
  return number
}

function confidenceValue(value: unknown, label: string): number {
  const number = finiteNumber(value, label)
  if (number < 0 || number > 1) throw new Error(`${label} 必须介于 0 和 1 之间`)
  return number
}

function enumValue<T extends string>(
  value: unknown,
  choices: readonly T[],
  label: string,
): T {
  if (typeof value !== 'string' || !choices.includes(value as T)) {
    throw new Error(`${label} 的值无效`)
  }
  return value as T
}

function pointValue(value: unknown, label: string): [number, number] {
  if (!Array.isArray(value) || value.length !== 2) {
    throw new Error(`${label} 必须包含两个整数`)
  }
  return [
    integerValue(value[0], `${label}[0]`),
    integerValue(value[1], `${label}[1]`),
  ]
}

function polygonValue(value: unknown, label: string, allowEmpty: boolean): PolygonCoords {
  if (!Array.isArray(value) || (!allowEmpty && value.length !== 4)) {
    throw new Error(`${label} 必须包含四个点`)
  }
  if (allowEmpty && value.length === 0) return []
  if (value.length !== 4) throw new Error(`${label} 必须为空或包含四个点`)
  return value.map((point, index) => pointValue(point, `${label}[${index}]`))
}

function coordsValue(value: unknown, label: string): BubbleCoords {
  if (!Array.isArray(value) || value.length !== 4) {
    throw new Error(`${label} 必须包含四个整数`)
  }
  const coords = value.map((part, index) => integerValue(part, `${label}[${index}]`))
  if (coords[0]! >= coords[2]! || coords[1]! >= coords[3]!) {
    throw new Error(`${label} 必须描述正面积区域`)
  }
  return coords as BubbleCoords
}

function rgbValue(value: unknown, label: string): [number, number, number] | null {
  if (value === null) return null
  if (!Array.isArray(value) || value.length !== 3) {
    throw new Error(`${label} 必须是 RGB 三元组或 null`)
  }
  const channels = value.map((channel, index) => integerValue(
    channel,
    `${label}[${index}]`,
    0,
  ))
  if (channels.some(channel => channel > 255)) {
    throw new Error(`${label} 的颜色通道必须介于 0 和 255 之间`)
  }
  return channels as [number, number, number]
}

function ocrResultValue(value: unknown, label: string): OcrResult | null {
  if (value === null) return null
  const record = exactObject(value, OCR_RESULT_KEYS, label)
  const confidence = record.confidence === null
    ? null
    : confidenceValue(record.confidence, `${label}.confidence`)
  return {
    text: stringValue(record.text, `${label}.text`),
    confidence,
    confidenceSupported: booleanValue(
      record.confidenceSupported,
      `${label}.confidenceSupported`,
    ),
    engine: stringValue(record.engine, `${label}.engine`, true),
    primaryEngine: stringValue(record.primaryEngine, `${label}.primaryEngine`, true),
    fallbackUsed: booleanValue(record.fallbackUsed, `${label}.fallbackUsed`),
  }
}

function textlinesValue(value: unknown, label: string): BubbleTextline[] {
  if (!Array.isArray(value)) throw new Error(`${label} 必须是数组`)
  return value.map((item, index) => {
    const itemLabel = `${label}[${index}]`
    const record = exactObject(item, ['polygon', 'direction', 'confidence'], itemLabel)
    return {
      polygon: polygonValue(record.polygon, `${itemLabel}.polygon`, false),
      direction: enumValue(record.direction, ['h', 'v'], `${itemLabel}.direction`),
      confidence: confidenceValue(record.confidence, `${itemLabel}.confidence`),
    }
  })
}

type PersistedBubbleState = Omit<
  BubbleState,
  'backendBubbleId' | 'clientMutationId' | 'fontFamily'
>

function currentBubblePayload(value: unknown, label: string): PersistedBubbleState {
  const payload = exactObject(value, BUBBLE_PAYLOAD_KEYS, label)
  const position = exactObject(payload.position, ['x', 'y'], `${label}.position`)
  const fontSize = integerValue(payload.fontSize, `${label}.fontSize`, 1)
  const strokeWidth = integerValue(payload.strokeWidth, `${label}.strokeWidth`, 0)
  const lineSpacing = finiteNumber(payload.lineSpacing, `${label}.lineSpacing`)
  if (lineSpacing <= 0) throw new Error(`${label}.lineSpacing 必须大于 0`)
  const colorConfidence = confidenceValue(
    payload.colorConfidence,
    `${label}.colorConfidence`,
  )
  const textColor = stringValue(payload.textColor, `${label}.textColor`)
  const fillColor = stringValue(payload.fillColor, `${label}.fillColor`)
  const strokeColor = stringValue(payload.strokeColor, `${label}.strokeColor`)
  for (const [field, color] of [
    ['textColor', textColor],
    ['fillColor', fillColor],
    ['strokeColor', strokeColor],
  ] as const) {
    if (!/^#[0-9A-Fa-f]{6}$/.test(color)) {
      throw new Error(`${label}.${field} 必须是 #RRGGBB 颜色`)
    }
  }
  return {
    originalText: stringValue(payload.originalText, `${label}.originalText`),
    translatedText: stringValue(payload.translatedText, `${label}.translatedText`),
    textboxText: stringValue(payload.textboxText, `${label}.textboxText`),
    coords: coordsValue(payload.coords, `${label}.coords`),
    polygon: polygonValue(payload.polygon, `${label}.polygon`, true),
    fontSize,
    textDirection: enumValue<ResolvedTextDirection>(
      payload.textDirection,
      ['vertical', 'horizontal'],
      `${label}.textDirection`,
    ),
    autoTextDirection: enumValue<ResolvedTextDirection>(
      payload.autoTextDirection,
      ['vertical', 'horizontal'],
      `${label}.autoTextDirection`,
    ),
    textColor,
    fillColor,
    rotationAngle: finiteNumber(payload.rotationAngle, `${label}.rotationAngle`),
    position: {
      x: finiteNumber(position.x, `${label}.position.x`),
      y: finiteNumber(position.y, `${label}.position.y`),
    },
    strokeEnabled: booleanValue(payload.strokeEnabled, `${label}.strokeEnabled`),
    strokeColor,
    strokeWidth,
    lineSpacing,
    inlineAlign: enumValue<LogicalAlign>(
      payload.inlineAlign,
      ['start', 'center', 'end'],
      `${label}.inlineAlign`,
    ),
    blockAlign: enumValue<LogicalAlign>(
      payload.blockAlign,
      ['start', 'center', 'end'],
      `${label}.blockAlign`,
    ),
    inpaintMethod: enumValue<InpaintMethod>(
      payload.inpaintMethod,
      ['solid', 'lama_mpe', 'litelama'],
      `${label}.inpaintMethod`,
    ),
    autoFgColor: rgbValue(payload.autoFgColor, `${label}.autoFgColor`),
    autoBgColor: rgbValue(payload.autoBgColor, `${label}.autoBgColor`),
    colorConfidence,
    textlines: textlinesValue(payload.textlines, `${label}.textlines`),
    ocrResult: ocrResultValue(payload.ocrResult, `${label}.ocrResult`),
  }
}

function pagePathFields(logicalSourcePath: string): Pick<ImageDataLoadInput, 'fileName' | 'folderPath'> {
  const separator = logicalSourcePath.lastIndexOf('/')
  if (separator < 0) return { fileName: logicalSourcePath }
  return {
    fileName: logicalSourcePath.slice(separator + 1),
    folderPath: logicalSourcePath.slice(0, separator),
  }
}

function translationStatus(page: V2PageSummary): TranslationStatus {
  if (page.renderStatus === 'render_failed' || page.renderStatus === 'repair_failed') {
    return 'failed'
  }
  if (
    page.renderStatus === 'ready'
    && page.translatedUrl
    && page.renderedRevision === page.documentRevision
  ) return 'completed'
  if (page.renderStatus === 'rendering' || page.renderStatus === 'awaiting_repair') {
    return 'processing'
  }
  return 'pending'
}

export function pageSummaryToImage(page: V2PageSummary): ImageDataLoadInput {
  const status = translationStatus(page)
  return {
    id: page.id,
    chapterId: page.chapterId,
    documentRevision: page.documentRevision,
    renderedRevision: page.renderedRevision,
    ...pagePathFields(page.logicalSourcePath),
    width: page.width ?? 0,
    height: page.height ?? 0,
    sourceAssetUrl: page.sourceUrl,
    cleanAssetUrl: page.cleanUrl ?? null,
    thumbnailSourceUrl: page.thumbnailSourceUrl,
    translatedAssetUrl: page.translatedUrl ?? null,
    bubbleStates: null,
    translationStatus: status,
    hasUnsavedChanges: false,
  }
}

export function pageDocumentToBubbles(document: V2PageDocument): BubbleState[] {
  return document.bubbles.map(bubble => {
    const fontId = bubble.fontId ?? document.defaultFontId
    if (!fontId) {
      throw new Error(`页面 ${document.pageId} 的气泡 ${bubble.bubbleId} 缺少后端字体 ID`)
    }
    const payload = currentBubblePayload(
      bubble.payload,
      `页面 ${document.pageId} 的气泡 ${bubble.bubbleId}`,
    )
    return {
      ...payload,
      backendBubbleId: bubble.bubbleId,
      fontFamily: fontId,
    }
  })
}
