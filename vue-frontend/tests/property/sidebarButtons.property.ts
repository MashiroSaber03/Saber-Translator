import { mount, type VueWrapper } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
import { afterEach, describe, expect, it, vi } from 'vitest'
import * as fc from 'fast-check'
import {
  useSettingsSidebar,
  type SettingsSidebarEmit,
} from '@/components/translate/useSettingsSidebar'
import {
  WORKFLOW_MODE_CONFIGS,
} from '@/components/translate/workflowModeConfig'
import { useImageStore } from '@/stores/imageStore'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import type { V2Job } from '@/api/v2/jobs'
import type { WorkflowMode, WorkflowRunRequest } from '@/types/workflow'
import type { TranslationStatus } from '@/types/image'
import { setTestImages } from '../helpers/imageFixtures'

const {
  getFontListMock,
  getPreferencesMock,
  savePreferencesMock,
  uploadFontMock,
} = vi.hoisted(() => ({
  getFontListMock: vi.fn(),
  getPreferencesMock: vi.fn(),
  savePreferencesMock: vi.fn(),
  uploadFontMock: vi.fn(),
}))

vi.mock('@/api/v2/settings', async importOriginal => ({
  ...await importOriginal<typeof import('@/api/v2/settings')>(),
  listV2Fonts: getFontListMock,
  uploadV2Font: uploadFontMock,
}))

interface GeneratedImage {
  failed: boolean
}

interface SidebarScenario {
  images: GeneratedImage[]
  currentImageIndex?: number
  isTranslationInProgress?: boolean
}

type SidebarApi = ReturnType<typeof useSettingsSidebar>

interface SidebarHarness {
  sidebar: SidebarApi
  emitted: Array<{ event: string; payload?: WorkflowRunRequest }>
  wrapper: VueWrapper
}

const workflowModes = WORKFLOW_MODE_CONFIGS.map(config => config.mode) as [WorkflowMode, ...WorkflowMode[]]
const selectableWorkflowModes = WORKFLOW_MODE_CONFIGS
  .filter(config => config.supportsPageSelection)
  .map(config => config.mode) as [WorkflowMode, ...WorkflowMode[]]
const generatedImageArb = fc.record({ failed: fc.boolean() })

function createSidebarHarness(scenario: SidebarScenario): SidebarHarness {
  setActivePinia(createPinia())
  getFontListMock.mockResolvedValue([])
  uploadFontMock.mockResolvedValue({
    id: 'font-uploaded',
    kind: 'uploaded',
    displayName: 'UploadedFont',
    builtinKey: null,
    assetUrl: '/api/v2/assets/font',
  })
  getPreferencesMock.mockResolvedValue({
    success: true,
    preferences: {
      rememberWorkflowModeEnabled: false,
      lastWorkflowMode: 'translate-current',
    },
  })
  savePreferencesMock.mockResolvedValue({ success: true })

  const imageStore = useImageStore()
  setTestImages(imageStore, scenario.images.map((image, index) => {
    const status: TranslationStatus = image.failed ? 'failed' : 'pending'
    return {
      fileName: `page-${index + 1}.png`,
      sourceAssetUrl: `/api/v2/assets/source-${index + 1}`,
      overrides: {
        chapterId: 'chapter-1',
        translationStatus: status,
      },
    }
  }))
  imageStore.setCurrentImageIndex(scenario.currentImageIndex ?? (scenario.images.length > 0 ? 0 : -1))
  if (imageStore.currentImage) {
    imageStore.updateCurrentImage({ bubbleStates: [] })
  }
  imageStore.setTranslationInProgress(scenario.isTranslationInProgress ?? false)
  const failedCount = scenario.images.filter(image => image.failed).length
  const jobStatus = failedCount > 0 ? 'completed_with_errors' : 'completed'
  useTaskCenterStore().history = [{
    jobId: 'translation-job',
    batchId: null,
    batchDisplayName: null,
    kind: 'translation',
    retryOfJobId: null,
    retryMode: null,
    status: jobStatus,
    queueRank: null,
    bookId: 'book-1',
    chapterId: 'chapter-1',
    pageId: null,
    blockedReason: null,
    blockedByJobId: null,
    progress: {
      executionMode: 'sequential',
      jobStatus,
      totalItems: scenario.images.length,
      completedItems: scenario.images.length - failedCount,
      failedItems: failedCount,
      skippedItems: 0,
      cancelledItems: 0,
      pools: [],
    },
    target: {},
    createdAt: '2026-08-23T04:00:00Z',
    startedAt: null,
    finishedAt: '2026-08-23T04:01:00Z',
  } satisfies V2Job]

  let sidebar: SidebarApi | null = null
  const emitted: SidebarHarness['emitted'] = []
  const emit = ((event: string, payload?: WorkflowRunRequest) => {
    emitted.push({ event, payload })
  }) as SettingsSidebarEmit

  const wrapper = mount(defineComponent({
    setup() {
      sidebar = useSettingsSidebar(emit)
      return () => h('div')
    },
  }))

  if (!sidebar) {
    throw new Error('Settings sidebar harness did not initialize')
  }

  return { sidebar, emitted, wrapper }
}

function withSidebarHarness(scenario: SidebarScenario, run: (harness: SidebarHarness) => void): void {
  const harness = createSidebarHarness(scenario)
  try {
    run(harness)
  } finally {
    harness.wrapper.unmount()
  }
}

function setWorkflow(sidebar: SidebarApi, mode: WorkflowMode): void {
  sidebar.selectedWorkflowMode.value = mode
}

function hasFailure(images: GeneratedImage[]): boolean {
  return images.some(image => image.failed)
}

describe('settings sidebar workflow properties', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('disables every workflow when no image is loaded', () => {
    fc.assert(
      fc.property(fc.constantFrom(...workflowModes), fc.boolean(), (mode, isTranslationInProgress) => {
        withSidebarHarness({ images: [], isTranslationInProgress }, ({ sidebar, emitted }) => {
          setWorkflow(sidebar, mode)

          expect(sidebar.canRunWorkflow.value).toBe(false)
          sidebar.handleRunWorkflow()
          expect(emitted).toEqual([])
        })
      }),
      { numRuns: 100 },
    )
  })

  it('enables image-backed workflows according to the translation lock', () => {
    const workflowGroups = {
      translation: ['translate-current', 'translate-batch', 'hq-batch', 'proofread-batch'] as const,
      singleImageActions: ['remove-current', 'delete-current'] as const,
      collectionActions: ['remove-batch', 'clear-all'] as const,
    }

    fc.assert(
      fc.property(
        fc.array(generatedImageArb, { minLength: 1, maxLength: 8 }),
        fc.boolean(),
        (images, isTranslationInProgress) => {
          withSidebarHarness({ images, isTranslationInProgress }, ({ sidebar }) => {
            for (const mode of workflowGroups.translation) {
              setWorkflow(sidebar, mode)
              expect(sidebar.canRunWorkflow.value).toBe(!isTranslationInProgress)
            }

            for (const mode of [
              ...workflowGroups.singleImageActions,
              ...workflowGroups.collectionActions,
            ]) {
              setWorkflow(sidebar, mode)
              expect(sidebar.canRunWorkflow.value).toBe(!isTranslationInProgress)
            }
          })
        },
      ),
      { numRuns: 100 },
    )
  })

  it('requires a valid page selection only for workflows that support it', () => {
    fc.assert(
      fc.property(
        fc.array(generatedImageArb, { minLength: 1, maxLength: 8 }),
        fc.constantFrom(...selectableWorkflowModes),
        (images, mode) => {
          withSidebarHarness({ images }, ({ sidebar, emitted }) => {
            setWorkflow(sidebar, mode)
            sidebar.isPageSelectionEnabled.value = true
            sidebar.selectedPages.value = []

            expect(sidebar.canRunWorkflow.value).toBe(false)
            sidebar.handleRunWorkflow()
            expect(emitted).toEqual([])

            sidebar.handlePageSelectionConfirm([1])

            expect(sidebar.canRunWorkflow.value).toBe(true)
            sidebar.handleRunWorkflow()
            expect(emitted).toEqual([
              {
                event: 'runWorkflow',
                payload: {
                  mode,
                  pageSelection: { pages: [1] },
                },
              },
            ])
          })
        },
      ),
      { numRuns: 100 },
    )
  })

  it('enables failed retry only when the durable task has failures and no translation is running', () => {
    fc.assert(
      fc.property(
        fc.array(generatedImageArb, { minLength: 1, maxLength: 8 }),
        fc.boolean(),
        (images, isTranslationInProgress) => {
          withSidebarHarness({ images, isTranslationInProgress }, ({ sidebar }) => {
            setWorkflow(sidebar, 'retry-failed')

            expect(sidebar.canRunWorkflow.value).toBe(
              hasFailure(images) && !isTranslationInProgress,
            )
          })
        },
      ),
      { numRuns: 100 },
    )
  })

  it('mirrors navigation availability from the real image store', () => {
    fc.assert(
      fc.property(
        fc.array(generatedImageArb, { minLength: 1, maxLength: 8 }),
        fc.nat({ max: 7 }),
        (images, indexOffset) => {
          const currentImageIndex = Math.min(indexOffset, images.length - 1)

          withSidebarHarness({ images, currentImageIndex }, ({ sidebar }) => {
            expect(sidebar.canGoPrevious.value).toBe(currentImageIndex > 0)
            expect(sidebar.canGoNext.value).toBe(currentImageIndex < images.length - 1)
          })
        },
      ),
      { numRuns: 100 },
    )
  })
})
