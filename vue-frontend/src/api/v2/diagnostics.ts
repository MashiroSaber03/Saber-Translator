import { normalizeProviderId } from '@/config/aiProviders'
import type {
  DiagnosticConnectionTestResponse,
  FetchModelsResponse,
} from '@/types'

import {
  fetchV2ModelCatalog,
  runV2ConnectionTest,
} from './settings'

export interface AiVisionOcrTestParams {
  provider: string
  apiKey: string
  modelName: string
  customBaseUrl?: string
  prompt?: string
  domain?: string
}

export interface AiTranslateTestParams {
  provider: string
  apiKey: string
  modelName?: string
  baseUrl?: string
  domain?: string
}

function secretOrDomain(
  domain: string,
  secret: Record<string, string>,
): { domain?: string; secret?: Record<string, string> } {
  const present = Object.fromEntries(
    Object.entries(secret).filter(([, value]) => value.trim().length > 0),
  )
  return Object.keys(present).length > 0 ? { secret: present } : { domain }
}

export function fetchModels(
  provider: string,
  apiKey: string,
  baseUrl?: string,
  domain = 'translation',
): Promise<FetchModelsResponse> {
  const secretField = domain === 'ai_vision_ocr' ? 'ai_vision_api_key' : 'api_key'
  return fetchV2ModelCatalog({
    provider: normalizeProviderId(provider),
    baseUrl: baseUrl || undefined,
    ...secretOrDomain(domain, { [secretField]: apiKey }),
  })
}

export function testOllamaConnection(baseUrl?: string): Promise<DiagnosticConnectionTestResponse> {
  return runV2ConnectionTest('ollama', { baseUrl, domain: 'translation' })
}

export function testSakuraConnection(baseUrl?: string): Promise<DiagnosticConnectionTestResponse> {
  return runV2ConnectionTest('sakura', { baseUrl, domain: 'translation' })
}

export function testBaiduOcrConnection(
  apiKey: string,
  secretKey: string,
): Promise<DiagnosticConnectionTestResponse> {
  return runV2ConnectionTest('baidu_ocr', {
    ...secretOrDomain('ocr', {
      baidu_api_key: apiKey,
      baidu_secret_key: secretKey,
    }),
  })
}

export function testLamaRepair(): Promise<DiagnosticConnectionTestResponse> {
  return runV2ConnectionTest('lama_repair')
}

export function testAiVisionOcrConnection(
  params: AiVisionOcrTestParams,
): Promise<DiagnosticConnectionTestResponse> {
  return runV2ConnectionTest('ai_vision_ocr', {
    provider: normalizeProviderId(params.provider),
    model: params.modelName,
    baseUrl: params.customBaseUrl || undefined,
    prompt: params.prompt || undefined,
    ...secretOrDomain(params.domain || 'ai_vision_ocr', {
      ai_vision_api_key: params.apiKey,
    }),
  })
}

export function testAiTranslateConnection(
  params: AiTranslateTestParams,
): Promise<DiagnosticConnectionTestResponse> {
  return runV2ConnectionTest('ai_translate', {
    provider: normalizeProviderId(params.provider),
    model: params.modelName || undefined,
    baseUrl: params.baseUrl || undefined,
    ...secretOrDomain(params.domain || 'translation', { api_key: params.apiKey }),
  })
}

export function testBaiduTranslateConnection(
  appId: string,
  appKey: string,
): Promise<DiagnosticConnectionTestResponse> {
  return runV2ConnectionTest('baidu_translate', {
    provider: 'baidu_translate',
    ...secretOrDomain('translation', { app_id: appId, app_key: appKey }),
  })
}

export function testYoudaoTranslateConnection(
  appKey: string,
  appSecret: string,
): Promise<DiagnosticConnectionTestResponse> {
  return runV2ConnectionTest('youdao_translate', {
    provider: 'youdao_translate',
    ...secretOrDomain('translation', { app_key: appKey, app_secret: appSecret }),
  })
}
