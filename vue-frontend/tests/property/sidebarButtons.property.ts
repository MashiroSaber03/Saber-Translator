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
  isBatchTranslationInProgress?: boolean
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
  uploadFontMock.mockResolvedValue({ id: 'font-uploaded', assetUrl: '/api/v2/assets/font' })
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
        translationStatus: status,
        translationFailed: image.failed,
      },
    }
  }))
  imageStore.setCurrentImageIndex(scenario.currentImageIndex ?? (scenario.images.length > 0 ? 0 : -1))
  if (imageStore.currentImage) {
    imageStore.updateCurrentImage({ bubbleStates: [] })
  }
  imageStore.setBatchTranslationInProgress(scenario.isBatchTranslationInProgress ?? false)

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
      fc.property(fc.constantFrom(...workflowModes), fc.boolean(), (mode, isBatchTranslationInProgress) => {
        withSidebarHarness({ images: [], isBatchTranslationInProgress }, ({ sidebar, emitted }) => {
          setWorkflow(sidebar, mode)

          expect(sidebar.canRunWorkflow.value).toBe(false)
          sidebar.handleRunWorkflow()
          expect(emitted).toEqual([])
        })
      }),
      { numRuns: 100 },
    )
  })

  it('enables image-backed workflows according to the batch lock', () => {
    const workflowGroups = {
      translation: ['translate-current', 'translate-batch', 'hq-batch', 'proofread-batch'] as const,
      singleImageActions: ['remove-current', 'delete-current'] as const,
      collectionActions: ['remove-batch', 'clear-all'] as const,
    }

    fc.assert(
      fc.property(
        fc.array(generatedImageArb, { minLength: 1, maxLength: 8 }),
        fc.boolean(),
        (images, isBatchTranslationInProgress) => {
          withSidebarHarness({ images, isBatchTranslationInProgress }, ({ sidebar }) => {
            for (const mode of workflowGroups.translation) {
              setWorkflow(sidebar, mode)
              expect(sidebar.canRunWorkflow.value).toBe(!isBatchTranslationInProgress)
            }

            for (const mode of workflowGroups.singleImageActions) {
              setWorkflow(sidebar, mode)
              expect(sidebar.canRunWorkflow.value).toBe(true)
            }

            for (const mode of workflowGroups.collectionActions) {
              setWorkflow(sidebar, mode)
              expect(sidebar.canRunWorkflow.value).toBe(true)
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

  it('enables failed retry only when a failed image exists and no batch is running', () => {
    fc.assert(
      fc.property(
        fc.array(generatedImageArb, { minLength: 1, maxLength: 8 }),
        fc.boolean(),
        (images, isBatchTranslationInProgress) => {
          withSidebarHarness({ images, isBatchTranslationInProgress }, ({ sidebar }) => {
            setWorkflow(sidebar, 'retry-failed')

            expect(sidebar.canRunWorkflow.value).toBe(
              hasFailure(images) && !isBatchTranslationInProgress,
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
