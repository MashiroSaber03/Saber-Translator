import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

const insightResponseTypeNames = [
  'InsightStatusResponse',
  'InsightOverviewResponse',
  'InsightTimelineResponse',
]

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('type source contracts', () => {
  it('imports Insight response types from the Insight domain type file', () => {
    for (const file of [
      'src/api/insight.ts',
      'src/utils/insightStatus.ts',
      'src/components/insight/timeline/useTimelinePanel.ts',
    ]) {
      const content = source(file)
      const barrelImportBlocks = content.match(/import\s+type\s+\{[\s\S]*?\}\s+from\s+['"]@\/types['"]/g) ?? []

      for (const importBlock of barrelImportBlocks) {
        for (const typeName of insightResponseTypeNames) {
          expect(importBlock, `${file} should import ${typeName} from @/types/insight`).not.toContain(typeName)
        }
      }
    }
  })

  it('keeps Insight response interfaces out of the generic API type file', () => {
    const apiTypes = source('src/types/api.ts')

    for (const typeName of insightResponseTypeNames) {
      expect(apiTypes).not.toContain(`export interface ${typeName}`)
    }
  })

  it('keeps shared API types split by owner behind the public API barrel', () => {
    const apiTypes = source('src/types/api.ts')

    for (const typeName of [
      'ApiResponse',
      'ApiError',
      'ReRenderResponse',
      'BookData',
      'PluginData',
      'FetchModelsResponse',
    ]) {
      expect(apiTypes, `${typeName} should be re-exported from an owner file`).not.toContain(`export interface ${typeName}`)
    }

    for (const exportPath of [
      './apiCore',
      './translationApi',
      './bookshelf',
      './plugin',
      './diagnostics',
    ]) {
      expect(apiTypes).toContain(`from '${exportPath}'`)
    }

    expect(source('src/types/apiCore.ts')).toContain('export interface ApiResponse')
    expect(source('src/types/translationApi.ts')).toContain('export interface ReRenderResponse')
    expect(source('src/types/bookshelf.ts')).toContain('export interface BookData')
    expect(source('src/types/plugin.ts')).toContain('export interface PluginData')
    expect(source('src/types/diagnostics.ts')).toContain('export interface FetchModelsResponse')
  })

  it('keeps settings schema types split by owner behind the public settings barrel', () => {
    const settingsTypes = source('src/types/settings.ts')

    for (const typeName of [
      'OcrEngine',
      'OpenAICompatibleOptions',
      'BaiduOcrSettings',
      'TextStyleSettings',
      'TranslationSettings',
    ]) {
      expect(settingsTypes, `${typeName} should be re-exported from an owner file`).not.toContain(`export interface ${typeName}`)
      expect(settingsTypes, `${typeName} should be re-exported from an owner file`).not.toContain(`export type ${typeName}`)
    }

    for (const staleNarration of [
      '/**',
      '设置类型定义',
      '定义翻译设置',
      '翻译服务商类型',
      'OpenAI-compatible 统一选项',
      '完整的翻译设置',
    ]) {
      expect(settingsTypes).not.toContain(staleNarration)
    }

    for (const exportPath of [
      './settingsProviders',
      './openaiSettings',
      './ocrSettings',
      './textStyleSettings',
      './translationSettings',
    ]) {
      expect(settingsTypes).toContain(`from '${exportPath}'`)
    }

    expect(source('src/types/settingsProviders.ts')).toContain('export type OcrEngine')
    expect(source('src/types/openaiSettings.ts')).toContain('export interface OpenAICompatibleOptions')
    expect(source('src/types/ocrSettings.ts')).toContain('export interface BaiduOcrSettings')
    expect(source('src/types/textStyleSettings.ts')).toContain('export interface TextStyleSettings')
    expect(source('src/types/translationSettings.ts')).toContain('export interface TranslationSettings')
  })

  it('keeps workflow UI metadata out of the shared workflow type module', () => {
    const workflowTypes = source('src/types/workflow.ts')
    const workflowConfig = source('src/components/translate/workflowModeConfig.ts')

    for (const staleTypeOwnerContent of [
      '/**',
      'WORKFLOW_MODE_CONFIGS',
      'DEFAULT_WORKFLOW_MODE',
      'isWorkflowMode',
      'label:',
      'startLabel:',
      '翻译当前图片',
      '清除所有图片',
    ]) {
      expect(workflowTypes).not.toContain(staleTypeOwnerContent)
    }

    expect(workflowTypes).toContain('export type WorkflowMode')
    expect(workflowTypes).toContain('export interface WorkflowRunRequest')
    expect(workflowConfig).toContain('export const WORKFLOW_MODE_CONFIGS')
    expect(workflowConfig).toContain('export const DEFAULT_WORKFLOW_MODE')
    expect(workflowConfig).toContain('export function isWorkflowMode')
  })

  it('keeps WebImport types aligned with manifest-driven providers and grouped settings owners', () => {
    const webImportTypes = source('src/types/webImport.ts')

    for (const staleNarration of [
      '/**',
      '网页导入相关类型定义',
      "| 'openai'",
      "| 'siliconflow'",
      'Firecrawl 配置',
      'AI Agent 配置',
      '网页导入完整设置',
      '网页导入运行时状态',
    ]) {
      expect(webImportTypes).not.toContain(staleNarration)
    }

    expect(webImportTypes).toContain('export type WebImportAgentProvider = string')

    for (const ownerInterface of [
      'WebImportFirecrawlSettings',
      'WebImportAgentSettings',
      'WebImportExtractionSettings',
      'WebImportDownloadSettings',
      'WebImportAdvancedSettings',
      'WebImportUiSettings',
      'WebImportDownloadProgress',
    ]) {
      expect(webImportTypes).toContain(`export interface ${ownerInterface}`)
    }
  })

  it('keeps Character Studio types split by owner behind the public Studio barrel', () => {
    const studioTypes = source('src/types/characterStudio.ts')

    for (const typeName of [
      'CharacterStudioCandidate',
      'CharacterStudioDocument',
      'LorebookEntryNode',
      'RegexScript',
      'StateTask',
      'CharacterStudioChatSession',
      'CharacterStudioEditorPendingState',
      'CharacterStudioAgentPatchV2',
      'CharacterStudioIndexResponse',
    ]) {
      expect(studioTypes, `${typeName} should be re-exported from an owner file`).not.toContain(`export interface ${typeName}`)
      expect(studioTypes, `${typeName} should be re-exported from an owner file`).not.toContain(`export type ${typeName}`)
    }

    for (const exportPath of [
      './characterStudioApi',
      './characterStudioChat',
      './characterStudioDocument',
      './characterStudioEditor',
      './characterStudioPatch',
    ]) {
      expect(studioTypes).toContain(`from '${exportPath}'`)
    }

    expect(source('src/types/characterStudioApi.ts')).toContain('export interface CharacterStudioIndexResponse')
    expect(source('src/types/characterStudioChat.ts')).toContain('export interface CharacterStudioChatSession')
    expect(source('src/types/characterStudioDocument.ts')).toContain('export interface CharacterStudioDocument')
    expect(source('src/types/characterStudioEditor.ts')).toContain('export interface CharacterStudioEditorPendingState')
    expect(source('src/types/characterStudioPatch.ts')).toContain('export interface CharacterStudioAgentPatchV2')
  })

  it('keeps Insight store OpenAI option types sourced from settings types', () => {
    const insightTypes = existsSync(resolve(process.cwd(), 'src/types/insightStoreTypes.ts'))
      ? source('src/types/insightStoreTypes.ts')
      : source('src/types/insight.ts')

    expect(insightTypes).toContain("import type { OpenAICompatibleOptions } from './settings'")
    expect(insightTypes).toContain('export type StoreOpenAICompatibleOptions = OpenAICompatibleOptions')
    expect(insightTypes).not.toContain('export interface StoreOpenAICompatibleRequestOptions')
    expect(insightTypes).not.toContain('export interface StoreOpenAICompatibleExecutionOptions')
    expect(insightTypes).not.toContain('export interface StoreOpenAICompatibleOptions')

    for (const staleNarration of [
      'Manga Insight 类型定义',
      '统一的类型定义单一来源',
      '// ==================== Store 状态类型 ====================',
      '// ==================== Store 专用配置类型（camelCase）====================',
      '// ==================== 配置类型 ====================',
      '// ==================== 分析数据类型 ====================',
      '// ==================== 任务类型 ====================',
      '// ==================== 时间线类型 ====================',
      '// ==================== 笔记类型 ====================',
      '// ==================== 问答类型 ====================',
      '// ==================== 概览模板类型 ====================',
      '// ==================== API 响应类型 ====================',
    ]) {
      expect(insightTypes).not.toContain(staleNarration)
    }
  })

  it('keeps Insight types split by owner behind the public Insight barrel', () => {
    const insightTypes = source('src/types/insight.ts')

    for (const typeName of [
      'AnalysisStatus',
      'StoreInsightConfig',
      'VlmConfig',
      'PageAnalysis',
      'TimelineData',
      'NoteData',
      'InsightStatusResponse',
    ]) {
      expect(insightTypes, `${typeName} should be re-exported from an owner file`).not.toContain(`export interface ${typeName}`)
      expect(insightTypes, `${typeName} should be re-exported from an owner file`).not.toContain(`export type ${typeName}`)
    }

    for (const exportPath of [
      './insightStoreTypes',
      './insightConfigTypes',
      './insightAnalysisTypes',
      './insightTimelineTypes',
      './insightNotesQaTypes',
      './insightResponseTypes',
    ]) {
      expect(insightTypes).toContain(`from '${exportPath}'`)
    }

    expect(source('src/types/insightStoreTypes.ts')).toContain('export interface StoreInsightConfig')
    expect(source('src/types/insightConfigTypes.ts')).toContain('export interface InsightConfig')
    expect(source('src/types/insightAnalysisTypes.ts')).toContain('export interface PageAnalysis')
    expect(source('src/types/insightTimelineTypes.ts')).toContain('export interface TimelineData')
    expect(source('src/types/insightNotesQaTypes.ts')).toContain('export interface NoteData')
    expect(source('src/types/insightResponseTypes.ts')).toContain('export interface InsightStatusResponse')
  })

  it('keeps Insight runtime converters out of the type package', () => {
    expect(existsSync(resolve(process.cwd(), 'src/types/insight/index.ts'))).toBe(false)
    expect(existsSync(resolve(process.cwd(), 'src/types/insight/converters.ts'))).toBe(false)

    const utility = source('src/utils/insightConverters.ts')
    expect(utility).toContain('export function toSnakeCase')
    expect(utility).toContain('export function toCamelCase')
    expect(utility).toContain('export function configToApi')
    expect(utility).toContain('export function configFromApi')
  })

  it('keeps the public type barrel free of mechanical export narration', () => {
    const barrel = source('src/types/index.ts')

    for (const staleNarration of [
      '类型定义索引文件',
      '统一导出所有类型定义',
      '气泡状态类型',
      '图片数据类型',
      '设置类型',
      'API 响应类型',
      'OCR 类型',
      '文件夹类型',
      'Manga Insight 类型',
      '角色工坊类型',
      '网页导入类型',
      '翻译页工作流类型',
      '书籍级翻译约束',
      '术语表 / 禁翻表类型',
    ]) {
      expect(barrel).not.toContain(staleNarration)
    }

    const exportPrefix = 'export * ' + 'from '
    const expectedLines = [
      './bubble',
      './image',
      './settings',
      './api',
      './ocr',
      './folder',
      './insight',
      './characterStudio',
      './webImport',
      './workflow',
      './bookTranslationConstraints',
      './translationConstraints',
    ].flatMap(modulePath => [`${exportPrefix}'${modulePath}'`, ''])
    expectedLines.pop()

    expect(barrel.trim().split(/\r?\n/)).toEqual(expectedLines)
  })

  it('keeps bubble type helper comments focused on render-direction behavior', () => {
    const bubbleTypes = source('src/types/bubble.ts')

    expect(bubbleTypes).toContain('Stable backend identity')
    expect(bubbleTypes).not.toContain('// ============================================================')
    expect(bubbleTypes).not.toContain('气泡状态类型定义')
    expect(bubbleTypes).not.toContain('与后端 BubbleState 数据类对应')
    expect(bubbleTypes).not.toContain('包含气泡的所有渲染参数')
    expect(bubbleTypes).not.toContain('文本内容')
    expect(bubbleTypes).not.toContain('渲染参数')
    expect(bubbleTypes).not.toContain('工具函数')
    expect(bubbleTypes).not.toContain('@param')
    expect(bubbleTypes).not.toContain('@returns')
    expect(bubbleTypes).toContain('异常输入会按检测方向和气泡宽高比回退')
  })

  it('keeps translate image state grouped by owner fields', () => {
    const imageTypes = source('src/types/image.ts')

    for (const staleNarration of [
      '/**',
      '图片数据类型定义',
      '包含图片的所有状态信息',
      '图片尺寸',
      '图片数据（Base64）',
      '气泡状态',
      '双掩膜系统',
      '手动标注标记',
      '翻译状态',
      '图片级别设置',
      '文件夹导入信息',
    ]) {
      expect(imageTypes).not.toContain(staleNarration)
    }

    for (const ownerInterface of [
      'ImageSourceFields',
      'ImageDetectionFields',
      'ImageWorkflowFields',
      'ImageTextStyleFields',
      'ImageUiFields',
      'ImageFolderFields',
    ]) {
      expect(imageTypes).toContain(`export interface ${ownerInterface}`)
    }

    expect(imageTypes).toMatch(/export interface ImageData\s+extends/)
    expect(imageTypes).toContain('ImageSourceFields')
    expect(imageTypes).toContain('ImageFolderFields')
    for (const removedMirror of [
      'originalDataURL',
      'translatedDataURL',
      'bubbleCoords',
      'originalTexts',
      'bubbleTexts',
      'ocrResults',
    ]) {
      expect(imageTypes).not.toContain(removedMirror)
    }
  })

  it('keeps cross-cutting barrel files free of mechanical narration', () => {
    const files = [
      'src/utils/index.ts',
    ]

    for (const file of files) {
      const content = source(file)

      for (const staleNarration of [
        '索引文件',
        '统一导出',
        '类型转换器',
        '主类型文件',
        '工具函数',
        '工厂函数',
        '计算',
      ]) {
        expect(content, file).not.toContain(staleNarration)
      }
    }
  })

  it('keeps the folder node type as a compact current contract', () => {
    const folderTypes = source('src/types/folder.ts')

    for (const staleNarration of [
      '文件夹树节点类型定义',
      '文件夹名称',
      '文件夹路径',
      '该文件夹下的图片',
      '子文件夹',
      '/**',
    ]) {
      expect(folderTypes).not.toContain(staleNarration)
    }
  })

  it('keeps the OCR result type as a compact current contract', () => {
    const ocrTypes = source('src/types/ocr.ts')

    for (const staleNarration of [
      '/**',
      'OCR 结果类型定义',
      '类型定义',
    ]) {
      expect(ocrTypes).not.toContain(staleNarration)
    }

    expect(ocrTypes).toContain('export interface OcrResult')
  })

  it('keeps translation constraint types as compact current contracts', () => {
    const constraintTypes = source('src/types/translationConstraints.ts')

    for (const staleNarration of [
      '/**',
      '术语表 / 禁翻表相关类型',
      '相关类型',
    ]) {
      expect(constraintTypes).not.toContain(staleNarration)
    }

    expect(constraintTypes).toContain('export interface GlossaryEntry')
    expect(constraintTypes).toContain('export interface NonTranslateEntry')
  })

  it('keeps the frontend environment declarations compact', () => {
    const envTypes = source('src/env.d.ts')

    for (const staleNarration of [
      '/**',
      'Vue 单文件组件类型声明',
      '让 TypeScript 能够识别 .vue 文件的导入',
    ]) {
      expect(envTypes).not.toContain(staleNarration)
    }

    expect(envTypes).toContain('/// <reference types="vite/client" />')
    expect(envTypes).toContain("declare module '*.vue'")
  })
})
