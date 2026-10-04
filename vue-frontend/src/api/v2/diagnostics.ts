import { normalizeProviderId } from '@/config/aiProviders'
import type { FetchModelsResponse } from '@/types'

import { fetchV2ModelCatalog, runV2ConnectionTest, type V2ConnectionTestResult } from './settings'
import { providerKeyField } from '@/utils/providerSettings'

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

export function fetchModels(
  provider: string,
  apiKey: string,
  baseUrl?: string,
  domain = 'translation'
): Promise<FetchModelsResponse> {
  const secretField = providerKeyField(domain)
  return fetchV2ModelCatalog({
    provider: normalizeProviderId(provider),
    baseUrl: baseUrl || undefined,
    domain,
    secret: { [secretField]: apiKey },
  })
}

export function testSakuraConnection(baseUrl?: string): Promise<V2ConnectionTestResult> {
  return runV2ConnectionTest('sakura', { baseUrl, domain: 'translation' })
}

export function testBaiduOcrConnection(
  apiKey: string,
  secretKey: string
): Promise<V2ConnectionTestResult> {
  return runV2ConnectionTest('baidu_ocr', {
    domain: 'ocr',
    secret: {
      baidu_api_key: apiKey,
      baidu_secret_key: secretKey,
    },
  })
}

export function testAiVisionOcrConnection(
  params: AiVisionOcrTestParams
): Promise<V2ConnectionTestResult> {
  return runV2ConnectionTest('ai_vision_ocr', {
    provider: normalizeProviderId(params.provider),
    model: params.modelName,
    baseUrl: params.customBaseUrl || undefined,
    prompt: params.prompt || undefined,
    domain: params.domain || 'ai_vision_ocr',
    secret: {
      ai_vision_api_key: params.apiKey,
    },
  })
}

export function testAiTranslateConnection(
  params: AiTranslateTestParams
): Promise<V2ConnectionTestResult> {
  return runV2ConnectionTest('ai_translate', {
    provider: normalizeProviderId(params.provider),
    model: params.modelName || undefined,
    baseUrl: params.baseUrl || undefined,
    domain: params.domain || 'translation',
    secret: { api_key: params.apiKey },
  })
}

export function testBaiduTranslateConnection(
  appId: string,
  appKey: string
): Promise<V2ConnectionTestResult> {
  return runV2ConnectionTest('baidu_translate', {
    domain: 'translation',
    secret: { app_id: appId, app_key: appKey },
  })
}

export function testYoudaoTranslateConnection(
  appKey: string,
  appSecret: string
): Promise<V2ConnectionTestResult> {
  return runV2ConnectionTest('youdao_translate', {
    domain: 'translation',
    secret: { app_key: appKey, app_secret: appSecret },
  })
}
