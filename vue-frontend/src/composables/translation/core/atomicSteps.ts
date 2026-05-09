import {
  executeAiTranslate,
  executeColor,
  executeDetection,
  executeInpaint,
  executeOcr,
  executeRender,
  executeTranslate,
} from './steps'
import { persistPage } from './persistenceService'
import type { PipelineRuntime, TaskContext } from './runtime'

export type AtomicStepName =
  | 'detection'
  | 'ocr'
  | 'color'
  | 'translate'
  | 'inpaint'
  | 'render'
  | 'save'

export type BatchAtomicStepName = 'aiTranslate'

export async function executeAtomicStep(
  step: AtomicStepName,
  context: TaskContext,
  runtime: PipelineRuntime
): Promise<TaskContext> {
  switch (step) {
    case 'detection': {
      const result = await executeDetection({
        imageIndex: context.imageIndex,
        image: context.sourceImage,
        translationMode: runtime.mode,
        forceDetect: false,
        settingsSnapshot: runtime.settingsSnapshot,
      })
      return {
        ...context,
        status: 'processing',
        bubbleCoords: result.bubbleCoords,
        bubbleAngles: result.bubbleAngles,
        bubblePolygons: result.bubblePolygons,
        autoDirections: result.autoDirections,
        textMask: result.textMask,
        textlinesPerBubble: result.textlinesPerBubble,
        originalTexts: result.originalTexts || [],
        bubbleStates: result.bubbleStates,
      }
    }
    case 'ocr': {
      const result = await executeOcr({
        imageIndex: context.imageIndex,
        image: context.sourceImage,
        translationMode: runtime.mode,
        bubbleCoords: context.bubbleCoords,
        bubbleStates: context.bubbleStates,
        textlinesPerBubble: context.textlinesPerBubble,
        settingsSnapshot: runtime.settingsSnapshot,
      })
      return {
        ...context,
        status: 'processing',
        originalTexts: result.originalTexts,
        ocrResults: result.ocrResults,
      }
    }
    case 'color': {
      const result = await executeColor({
        imageIndex: context.imageIndex,
        image: context.sourceImage,
        translationMode: runtime.mode,
        bubbleCoords: context.bubbleCoords,
        bubbleStates: context.bubbleStates,
        textlinesPerBubble: context.textlinesPerBubble,
      })
      return {
        ...context,
        status: 'processing',
        colors: result.colors,
      }
    }
    case 'translate': {
      const result = await executeTranslate({
        imageIndex: context.imageIndex,
        translationMode: runtime.mode,
        originalTexts: context.originalTexts,
        settingsSnapshot: runtime.settingsSnapshot,
      })
      return {
        ...context,
        status: 'processing',
        translatedTexts: result.translatedTexts,
        textboxTexts: result.textboxTexts,
        warnings: result.warnings,
      }
    }
    case 'inpaint': {
      const result = await executeInpaint({
        imageIndex: context.imageIndex,
        image: context.sourceImage,
        translationMode: runtime.mode,
        bubbleCoords: context.bubbleCoords,
        bubblePolygons: context.bubblePolygons,
        textMask: context.textMask,
        userMask: context.sourceImage.userMask || undefined,
        settingsSnapshot: runtime.settingsSnapshot,
      })
      return {
        ...context,
        status: 'processing',
        cleanImage: result.cleanImage,
      }
    }
    case 'render': {
      if (!context.cleanImage && runtime.mode === 'removeText' && context.bubbleCoords.length === 0) {
        return {
          ...context,
          status: 'processing',
          finalImage: context.sourceImage.originalDataURL,
          bubbleStates: [],
        }
      }
      if (runtime.mode === 'removeText' && context.translatedTexts.length === 0 && context.textboxTexts.length === 0) {
        return {
          ...context,
          status: 'processing',
          finalImage: context.cleanImage || context.sourceImage.cleanImageData || context.sourceImage.originalDataURL,
          bubbleStates: Array.isArray(context.bubbleStates) ? context.bubbleStates : [],
        }
      }
      const result = await executeRender({
        imageIndex: context.imageIndex,
        cleanImage: context.cleanImage || '',
        bubbleCoords: context.bubbleCoords,
        bubbleAngles: context.bubbleAngles,
        autoDirections: context.autoDirections,
        textlinesPerBubble: context.bubbleStates?.map((bubble) => bubble.textlines || []) || context.textlinesPerBubble,
        existingBubbleStates: context.bubbleStates,
        originalTexts: context.originalTexts,
        ocrResults: context.ocrResults,
        translatedTexts: context.translatedTexts,
        textboxTexts: context.textboxTexts,
        colors: context.colors,
        savedTextStyles: runtime.savedTextStyles,
        currentMode: runtime.mode,
        settingsSnapshot: runtime.settingsSnapshot,
        renderStylePolicy: {
          fontSize: runtime.savedTextStyles?.autoFontSize ? 'initialize_auto' : 'preserve',
          color: runtime.savedTextStyles?.useAutoTextColor ? 'initialize_auto' : 'preserve',
        },
      })
      return {
        ...context,
        status: 'processing',
        finalImage: result.finalImage,
        bubbleStates: result.bubbleStates,
      }
    }
    case 'save':
      return await persistPage(context, runtime)
  }
}

export async function executeBatchAtomicStep(
  step: BatchAtomicStepName,
  contexts: TaskContext[],
  runtime: PipelineRuntime
): Promise<TaskContext[]> {
  switch (step) {
    case 'aiTranslate': {
      const result = await executeAiTranslate({
        mode: runtime.mode === 'proofread' ? 'proofread' : 'hq',
        tasks: contexts.map((context) => ({
          imageIndex: context.imageIndex,
          image: context.sourceImage,
          originalTexts: context.originalTexts,
          autoDirections: context.autoDirections,
        })),
        settingsSnapshot: runtime.settingsSnapshot,
      })

      return contexts.map((context) => {
        const taskResult = result.results.find((item) => item.imageIndex === context.imageIndex)
        return {
          ...context,
          status: 'processing',
          translatedTexts: taskResult?.translatedTexts || [],
          textboxTexts: taskResult?.textboxTexts || [],
          warnings: taskResult?.warnings || [],
        }
      })
    }
  }
}
