import { extractGlossaryEntries } from '@/api/translate'
import {
  getProviderManifest,
  providerRequiresApiKey,
  providerRequiresBaseUrl,
  providerSupportsCapability,
} from '@/config/aiProviders'
import { useBookTranslationConstraintsStore } from '@/stores/bookTranslationConstraintsStore'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'
import type { TranslationSettings } from '@/types/settings'
import type { GlossaryEntry, GlossaryExtractionStats } from '@/types/translationConstraints'
import { serializeOpenAICompatibleOptionsForApi } from '@/utils/openaiOptions'

export interface AutoGlossaryInput {
  originalTexts: string[]
  settingsSnapshot: TranslationSettings
  bookTranslationConstraints: BookTranslationConstraints
  isBookshelfMode: boolean
}

export interface AutoGlossaryOutput {
  bookTranslationConstraints: BookTranslationConstraints
  autoGlossaryStats: GlossaryExtractionStats
}

function shouldRunAutoGlossary(input: AutoGlossaryInput): boolean {
  if (!input.isBookshelfMode) return false
  if (!input.originalTexts.some(text => String(text || '').trim())) return false

  const glossary = input.bookTranslationConstraints.glossary
  if (!glossary.enabled || !glossary.autoExtractEnabled) return false

  const translation = input.settingsSnapshot.translation
  const provider = translation.provider
  const manifest = getProviderManifest(provider)
  if (!providerSupportsCapability(provider, 'translation') || !manifest?.supportsJsonResponse) {
    return false
  }
  if (providerRequiresApiKey(provider) && !translation.apiKey?.trim()) {
    return false
  }
  if (!translation.modelName?.trim()) {
    return false
  }
  if (providerRequiresBaseUrl(provider) && !translation.customBaseUrl?.trim()) {
    return false
  }
  return true
}

function cloneConstraints(constraints: BookTranslationConstraints): BookTranslationConstraints {
  return JSON.parse(JSON.stringify(constraints)) as BookTranslationConstraints
}

export async function executeAutoGlossary(input: AutoGlossaryInput): Promise<AutoGlossaryOutput> {
  const fallback: AutoGlossaryOutput = {
    bookTranslationConstraints: cloneConstraints(input.bookTranslationConstraints),
    autoGlossaryStats: {
      added: 0,
      duplicates: 0,
      failedPages: 0,
    },
  }

  if (!shouldRunAutoGlossary(input)) {
    return fallback
  }

  const settings = input.settingsSnapshot
  const existingEntries = input.bookTranslationConstraints.glossary.entries || []

  try {
    const response = await extractGlossaryEntries({
      original_texts: input.originalTexts,
      source_language: settings.sourceLanguage,
      target_language: settings.targetLanguage,
      model_provider: settings.translation.provider,
      api_key: settings.translation.apiKey,
      model_name: settings.translation.modelName,
      custom_base_url: settings.translation.customBaseUrl,
      existing_entries: existingEntries,
      openai_options: serializeOpenAICompatibleOptionsForApi(settings.translation.openaiOptions),
    })

    if (!response.success) {
      return {
        ...fallback,
        autoGlossaryStats: {
          added: 0,
          duplicates: response.duplicate_count || 0,
          failedPages: 1,
        },
      }
    }

    const newEntries = (response.new_entries || []) as GlossaryEntry[]
    const duplicateCount = response.duplicate_count || 0
    if (newEntries.length === 0) {
      return {
        ...fallback,
        autoGlossaryStats: {
          added: 0,
          duplicates: duplicateCount,
          failedPages: 0,
        },
      }
    }

    const nextConstraints = cloneConstraints(input.bookTranslationConstraints)
    nextConstraints.glossary.entries = [...existingEntries, ...newEntries]

    const store = useBookTranslationConstraintsStore()
    const saved = await store.saveBookConstraints(nextConstraints)
    if (!saved) {
      return {
        ...fallback,
        autoGlossaryStats: {
          added: 0,
          duplicates: duplicateCount,
          failedPages: 1,
        },
      }
    }

    return {
      bookTranslationConstraints: cloneConstraints(store.constraints),
      autoGlossaryStats: {
        added: newEntries.length,
        duplicates: duplicateCount,
        failedPages: 0,
      },
    }
  } catch (error) {
    console.warn('[AutoGlossary] 自动术语提取失败，继续执行翻译', error)
    return {
      ...fallback,
      autoGlossaryStats: {
        added: 0,
        duplicates: 0,
        failedPages: 1,
      },
    }
  }
}
