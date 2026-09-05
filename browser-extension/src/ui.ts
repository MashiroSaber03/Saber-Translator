import type { ImageCandidate } from './discovery'
import type {
  BrowserLibraryBook,
  BrowserSessionDto,
  BrowserSessionImportCommand,
  BrowserSessionImportResult,
  DetectionMethod,
  DomainPreference,
  PanelPosition,
  TranslationMode,
} from './types'
import { UI_STYLES } from './uiStyles'

export interface UiCallbacks {
  onActivity(): void
  onDiscover(method: DetectionMethod): void
  onConfirm(candidateIds: string[]): void
  onPreferenceChange(preference: DomainPreference): void
  onPanelOpenChange(open: boolean): void
  onPanelPositionChange(position: PanelPosition): void
  onFabPositionChange(position: PanelPosition): void
  onToggleGlobal(): Promise<boolean>
  onTogglePage(browserPageId: string): Promise<boolean | null>
  onRetryPage(browserPageId: string): void
  onRetryUploads(): void
  onRetryStart(): void
  onStopDiscovery(): void
  onCancel(): void
  onLoadLibraryBooks(): Promise<BrowserLibraryBook[]>
  onImport(command: BrowserSessionImportCommand): Promise<BrowserSessionImportResult>
  onDisableSite(): void
  onDeleteAdaptation(): void
  onCopyDiagnostics(): void
}

type PanelView = 'idle' | 'candidates' | 'progress'

interface DragState {
  pointerId: number
  startClientX: number
  startClientY: number
  startElementX: number
  startElementY: number
  lastPosition: PanelPosition
  moved: boolean
}

const PANEL_MARGIN = 8

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
): HTMLElementTagNameMap[K] {
  const result = document.createElement(tag)
  if (className) result.className = className
  return result
}

export class ExtensionUi {
  readonly host: HTMLDivElement
  readonly shadow: ShadowRoot
  private readonly panel: HTMLDivElement
  private readonly fab: HTMLButtonElement
  private readonly banner: HTMLDivElement
  private readonly bannerTitle: HTMLElement
  private readonly bannerMessage: HTMLElement
  private readonly errorActions: HTMLElement
  private readonly retryUploadsButton: HTMLButtonElement
  private readonly retryStartButton: HTMLButtonElement
  private readonly settingsDetails: HTMLDetailsElement
  private readonly idleActions: HTMLElement
  private readonly discoverButton: HTMLButtonElement
  private readonly candidatesSection: HTMLElement
  private readonly candidatesGrid: HTMLElement
  private readonly candidateCount: HTMLElement
  private readonly confirmButton: HTMLButtonElement
  private readonly progressSection: HTMLElement
  private readonly preparationSection: HTMLElement
  private readonly preparationCount: HTMLElement
  private readonly preparationMeter: HTMLProgressElement
  private readonly preparationDetail: HTMLElement
  private readonly pageActionsDetails: HTMLDetailsElement
  private readonly pageActionsSummary: HTMLElement
  private readonly pageActions: HTMLElement
  private readonly toggleButton: HTMLButtonElement
  private readonly stopButton: HTMLButtonElement
  private readonly cancelButton: HTMLButtonElement
  private readonly importButton: HTMLButtonElement
  private readonly statElements = new Map<string, HTMLElement>()
  private readonly termsSection: HTMLDetailsElement
  private readonly termsSummary: HTMLElement
  private readonly termsElement: HTMLElement
  private readonly pickMask: HTMLElement
  private readonly methodSelect: HTMLSelectElement
  private readonly modeSelect: HTMLSelectElement
  private readonly glossaryInput: HTMLInputElement
  private readonly autoTermsInput: HTMLInputElement
  private readonly deleteAdaptationButton: HTMLButtonElement
  private readonly importOverlay: HTMLDivElement
  private readonly importNewInput: HTMLInputElement
  private readonly importExistingInput: HTMLInputElement
  private readonly importBookTitleInput: HTMLInputElement
  private readonly importChapterTitleInput: HTMLInputElement
  private readonly importSearchInput: HTMLInputElement
  private readonly importBookSelect: HTMLSelectElement
  private readonly importNewFields: HTMLElement
  private readonly importExistingFields: HTMLElement
  private readonly importSummary: HTMLElement
  private readonly importConfirmButton: HTMLButtonElement
  private latestSession: BrowserSessionDto | null = null
  private libraryBooks: BrowserLibraryBook[] = []
  private importing = false
  private preference: DomainPreference
  private dragState: DragState | null = null
  private fabDragState: DragState | null = null
  private suppressFabClick = false

  private readonly fabPointerMoveHandler = (event: PointerEvent): void => this.moveFabDrag(event)
  private readonly fabPointerEndHandler = (event: PointerEvent): void => this.finishFabDrag(event)

  private readonly resizeHandler = (): void => {
    if (this.panel.dataset.open === 'true' && this.preference.panelPosition) {
      this.preference = {
        ...this.preference,
        panelPosition: this.placePanel(this.preference.panelPosition),
      }
    }
    if (this.preference.fabPosition) {
      this.preference = {
        ...this.preference,
        fabPosition: this.placeFab(this.preference.fabPosition),
      }
    }
  }

  constructor(
    private readonly callbacks: UiCallbacks,
    preference: DomainPreference,
    private readonly pageLabel: string,
    knownSite: boolean,
  ) {
    this.preference = { ...preference }
    this.host = element('div')
    this.host.id = 'saber-translator-extension-root'
    this.shadow = this.host.attachShadow({ mode: 'closed' })
    for (const type of ['click', 'input', 'change']) {
      this.shadow.addEventListener(type, event => {
        if (!event.isTrusted) {
          event.preventDefault()
          event.stopImmediatePropagation()
          return
        }
        if (type === 'click') this.callbacks.onActivity()
      }, true)
    }
    const style = element('style')
    style.textContent = UI_STYLES
    this.shadow.append(style)
    const root = element('div', 'saber-root')
    this.shadow.append(root)
    document.documentElement.append(this.host)

    this.fab = element('button', 'saber-fab')
    this.fab.type = 'button'
    this.fab.title = knownSite ? '检测到漫画站，点击开始' : 'Saber 漫画翻译'
    this.fab.setAttribute('aria-label', this.fab.title)
    this.fab.dataset.state = 'ready'
    const glyph = element('span', 'saber-fab__glyph')
    glyph.textContent = 'S'
    const dot = element('span', 'saber-fab__dot')
    this.fab.append(glyph, dot)
    root.append(this.fab)

    this.panel = element('div', 'saber-panel')
    this.panel.dataset.open = String(preference.panelOpen)
    this.panel.dataset.view = 'idle'
    root.append(this.panel)

    const header = element('header', 'saber-header')
    header.title = '拖动调整面板位置'
    const logo = element('div', 'saber-logo')
    logo.textContent = 'S'
    const heading = element('div', 'saber-heading')
    const title = element('h2', 'saber-title')
    title.textContent = 'Saber 漫画翻译'
    const subtitle = element('div', 'saber-subtitle')
    subtitle.textContent = pageLabel
    heading.append(title, subtitle)
    const close = element('button', 'saber-icon-button')
    close.type = 'button'
    close.title = '关闭面板'
    close.setAttribute('aria-label', '关闭面板')
    close.textContent = '✕'
    header.append(logo, heading, close)
    this.panel.append(header)

    const body = element('div', 'saber-body')
    this.panel.append(body)
    this.banner = element('div', 'saber-banner')
    this.banner.setAttribute('role', 'status')
    this.banner.setAttribute('aria-live', 'polite')
    const bannerDot = element('span', 'saber-banner__dot')
    const bannerText = element('div', 'saber-banner__text')
    this.bannerTitle = element('strong')
    this.bannerTitle.textContent = knownSite ? '已识别漫画站点' : '等待开始'
    this.bannerMessage = element('span')
    this.bannerMessage.textContent = knownSite
      ? '确认设置后识别当前页面中的漫画图片。'
      : 'Saber 只会在你点击后识别页面。'
    bannerText.append(this.bannerTitle, this.bannerMessage)
    this.banner.append(bannerDot, bannerText)
    body.append(this.banner)

    this.errorActions = element('div', 'saber-error-actions')
    this.errorActions.hidden = true
    this.retryUploadsButton = this.button('重试上传', 'saber-button--primary')
    this.retryUploadsButton.hidden = true
    this.retryStartButton = this.button('重试启动', 'saber-button--primary')
    this.retryStartButton.hidden = true
    const diagnostics = this.button('复制诊断', 'saber-button--quiet')
    this.errorActions.append(this.retryUploadsButton, this.retryStartButton, diagnostics)
    body.append(this.errorActions)

    this.settingsDetails = element('details', 'saber-settings')
    this.settingsDetails.open = true
    const settingsSummary = element('summary', 'saber-settings__summary')
    settingsSummary.textContent = '翻译设置'
    this.settingsDetails.append(settingsSummary)
    const settingsGrid = element('div', 'saber-grid')
    this.methodSelect = this.selectField('识别方式', [
      ['adapter', '站点适配 / 通用'],
      ['dom-agent', 'DOM Agent'],
      ['similar', '点选同类图片'],
    ], preference.method)
    this.modeSelect = this.selectField('翻译模式', [
      ['standard', '标准翻译'],
      ['hq', '高质量翻译'],
    ], preference.mode)
    settingsGrid.append(
      this.methodSelect.closest('.saber-field')!,
      this.modeSelect.closest('.saber-field')!,
    )
    this.glossaryInput = this.checkField('启用术语表', preference.glossaryEnabled)
    this.autoTermsInput = this.checkField('自动添加术语', preference.autoTermsEnabled)
    settingsGrid.append(
      this.glossaryInput.closest('.saber-check')!,
      this.autoTermsInput.closest('.saber-check')!,
    )
    const settingsActions = element('div', 'saber-actions saber-actions--end')
    this.deleteAdaptationButton = this.button(
      '删除已保存适配',
      'saber-button--quiet saber-button--danger',
    )
    this.deleteAdaptationButton.hidden = !preference.rule
    const disable = this.button('在此站点停用', 'saber-button--quiet')
    settingsActions.append(this.deleteAdaptationButton, disable)
    this.settingsDetails.append(settingsGrid, settingsActions)
    body.append(this.settingsDetails)

    this.idleActions = element('div', 'saber-actions saber-idle-actions')
    this.discoverButton = this.button('', 'saber-button--primary')
    this.updateDiscoverLabel()
    this.idleActions.append(this.discoverButton)
    body.append(this.idleActions)

    this.candidatesSection = element('section', 'saber-section saber-candidate-section')
    this.candidatesSection.hidden = true
    const candidateHeading = element('div', 'saber-section__title')
    const candidateTitle = element('span')
    candidateTitle.textContent = '确认候选图片'
    const candidateTools = element('span', 'saber-section__tools')
    this.candidateCount = element('span', 'saber-counter')
    const selectAll = this.linkButton('全选')
    const clearAll = this.linkButton('清空')
    candidateTools.append(this.candidateCount, selectAll, clearAll)
    candidateHeading.append(candidateTitle, candidateTools)
    this.candidatesGrid = element('div', 'saber-candidates')
    const candidateActions = element('div', 'saber-actions saber-candidate-actions')
    const back = this.button('返回设置', 'saber-button--quiet')
    this.confirmButton = this.button('开始翻译', 'saber-button--primary')
    candidateActions.append(back, this.confirmButton)
    this.candidatesSection.append(candidateHeading, this.candidatesGrid, candidateActions)
    body.append(this.candidatesSection)

    this.progressSection = element('section', 'saber-section saber-progress-section')
    this.progressSection.hidden = true
    this.preparationSection = element('div', 'saber-preparation')
    this.preparationSection.hidden = true
    const preparationHeading = element('div', 'saber-preparation__heading')
    const preparationLabel = element('span')
    preparationLabel.textContent = '图片准备进度'
    this.preparationCount = element('strong', 'saber-preparation__count')
    this.preparationCount.textContent = '0 / 0'
    preparationHeading.append(preparationLabel, this.preparationCount)
    this.preparationMeter = element('progress', 'saber-preparation__meter')
    this.preparationMeter.setAttribute('aria-label', '图片准备进度')
    this.preparationMeter.max = 1
    this.preparationMeter.value = 0
    this.preparationDetail = element('div', 'saber-preparation__detail')
    this.preparationDetail.textContent = '成功 0 · 失败 0'
    this.preparationSection.append(
      preparationHeading,
      this.preparationMeter,
      this.preparationDetail,
    )
    const progressHeading = element('div', 'saber-section__title')
    progressHeading.textContent = '页面任务'
    const progress = element('div', 'saber-progress')
    for (const [key, label] of ([
      ['total', '发现'],
      ['queued', '排队'],
      ['translating', '处理中'],
      ['completed', '完成'],
      ['failed', '失败'],
    ] as const)) {
      const stat = element('div', 'saber-stat')
      const value = element('strong')
      value.textContent = '0'
      const text = element('span')
      text.textContent = label
      stat.append(value, text)
      progress.append(stat)
      this.statElements.set(key, value)
    }
    const taskActions = element('div', 'saber-actions saber-progress-actions')
    this.toggleButton = this.button('显示原图')
    this.stopButton = this.button('停止发现')
    this.cancelButton = this.button('取消任务', 'saber-button--danger')
    this.importButton = this.button('导入到书架')
    taskActions.append(
      this.toggleButton,
      this.stopButton,
      this.cancelButton,
      this.importButton,
    )
    this.pageActionsDetails = element('details', 'saber-page-actions')
    this.pageActionsSummary = element('summary', 'saber-page-actions__summary')
    this.pageActionsSummary.textContent = '单页操作'
    this.pageActions = element('div', 'saber-page-actions__list')
    this.pageActionsDetails.append(this.pageActionsSummary, this.pageActions)
    this.progressSection.append(
      this.preparationSection,
      progressHeading,
      progress,
      this.pageActionsDetails,
      taskActions,
    )
    body.append(this.progressSection)

    this.termsSection = element('details', 'saber-section saber-terms-section')
    this.termsSection.hidden = true
    this.termsSummary = element('summary', 'saber-terms-summary')
    this.termsSummary.textContent = '实时术语'
    this.termsElement = element('div', 'saber-terms')
    this.termsElement.textContent = '尚未提取术语'
    this.termsSection.append(this.termsSummary, this.termsElement)
    body.append(this.termsSection)

    this.pickMask = element('div', 'saber-pick-mask')
    const pickTip = element('div', 'saber-pick-tip')
    pickTip.textContent = '点击一张漫画图片，Esc 取消'
    this.pickMask.append(pickTip)
    root.append(this.pickMask)

    this.importOverlay = element('div', 'saber-import-overlay')
    this.importOverlay.dataset.open = 'false'
    const importDialog = element('section', 'saber-import-dialog')
    importDialog.setAttribute('role', 'dialog')
    importDialog.setAttribute('aria-modal', 'true')
    importDialog.setAttribute('aria-labelledby', 'saber-import-title')
    const importHeader = element('div', 'saber-import-header')
    const importHeading = element('div')
    const importTitle = element('h3', 'saber-import-title')
    importTitle.id = 'saber-import-title'
    importTitle.textContent = '导入到书架'
    this.importSummary = element('p', 'saber-import-summary')
    importHeading.append(importTitle, this.importSummary)
    const importClose = this.button('✕', 'saber-icon-button')
    importClose.title = '关闭导入面板'
    importClose.setAttribute('aria-label', '关闭导入面板')
    importHeader.append(importHeading, importClose)

    const destination = element('div', 'saber-import-destinations')
    this.importNewInput = document.createElement('input')
    this.importNewInput.type = 'radio'
    this.importNewInput.name = 'saber-import-destination'
    this.importNewInput.value = 'new'
    this.importNewInput.checked = true
    const newChoice = element('label', 'saber-import-choice')
    newChoice.append(this.importNewInput, document.createTextNode('新建书籍'))
    this.importExistingInput = document.createElement('input')
    this.importExistingInput.type = 'radio'
    this.importExistingInput.name = 'saber-import-destination'
    this.importExistingInput.value = 'existing'
    const existingChoice = element('label', 'saber-import-choice')
    existingChoice.append(this.importExistingInput, document.createTextNode('加入已有书籍'))
    destination.append(newChoice, existingChoice)

    this.importNewFields = element('div', 'saber-import-fields')
    this.importBookTitleInput = this.textField('书籍名称')
    this.importNewFields.append(this.importBookTitleInput.closest('.saber-input-field')!)
    this.importExistingFields = element('div', 'saber-import-fields')
    this.importSearchInput = this.textField('搜索已有书籍')
    this.importBookSelect = element('select', 'saber-input saber-import-book-select')
    this.importBookSelect.setAttribute('aria-label', '选择已有书籍')
    const importBookSelectWrap = element('div', 'saber-import-select-wrap')
    importBookSelectWrap.append(this.importBookSelect)
    this.importExistingFields.append(
      this.importSearchInput.closest('.saber-input-field')!,
      importBookSelectWrap,
    )
    this.importExistingFields.hidden = true
    this.importChapterTitleInput = this.textField('章节名称')

    const importActions = element('div', 'saber-actions saber-actions--end saber-import-actions')
    const importCancel = this.button('取消', 'saber-button--quiet')
    this.importConfirmButton = this.button('确认导入', 'saber-button--primary')
    importActions.append(importCancel, this.importConfirmButton)
    importDialog.append(
      importHeader,
      destination,
      this.importNewFields,
      this.importExistingFields,
      this.importChapterTitleInput.closest('.saber-input-field')!,
      importActions,
    )
    this.importOverlay.append(importDialog)
    root.append(this.importOverlay)

    this.fab.addEventListener('click', (event) => {
      if (this.suppressFabClick) {
        this.suppressFabClick = false
        event.preventDefault()
        return
      }
      this.togglePanel()
    })
    this.fab.addEventListener('pointerdown', event => this.startFabDrag(event))
    close.addEventListener('click', () => this.setOpen(false, true))
    this.discoverButton.addEventListener('click', () => this.callbacks.onDiscover(this.method()))
    disable.addEventListener('click', () => this.callbacks.onDisableSite())
    this.deleteAdaptationButton.addEventListener(
      'click',
      () => this.callbacks.onDeleteAdaptation(),
    )
    diagnostics.addEventListener('click', () => this.callbacks.onCopyDiagnostics())
    this.retryUploadsButton.addEventListener('click', () => this.callbacks.onRetryUploads())
    this.retryStartButton.addEventListener('click', () => {
      this.retryStartButton.disabled = true
      this.callbacks.onRetryStart()
    })
    this.confirmButton.addEventListener('click', () => {
      const selected = [...this.candidatesGrid.querySelectorAll<HTMLInputElement>('input')]
        .filter(input => input.checked)
        .map(input => input.value)
      this.callbacks.onConfirm(selected)
    })
    selectAll.addEventListener('click', () => this.setAllCandidates(true))
    clearAll.addEventListener('click', () => this.setAllCandidates(false))
    back.addEventListener('click', () => this.showIdle())
    this.toggleButton.addEventListener('click', async () => {
      this.toggleButton.disabled = true
      try {
        const showingTranslated = await this.callbacks.onToggleGlobal()
        this.toggleButton.textContent = showingTranslated ? '显示原图' : '显示译图'
      } catch {
        // The controller already presents the actionable error in the panel.
      } finally {
        this.toggleButton.disabled = false
      }
    })
    this.stopButton.addEventListener('click', () => this.callbacks.onStopDiscovery())
    this.cancelButton.addEventListener('click', () => this.callbacks.onCancel())
    this.importButton.addEventListener('click', () => void this.openImportDialog())
    importClose.addEventListener('click', () => this.closeImportDialog())
    importCancel.addEventListener('click', () => this.closeImportDialog())
    this.importOverlay.addEventListener('click', event => {
      if (event.target === this.importOverlay) this.closeImportDialog()
    })
    this.importNewInput.addEventListener('change', () => this.updateImportDestination())
    this.importExistingInput.addEventListener('change', () => this.updateImportDestination())
    this.importSearchInput.addEventListener('input', () => this.renderLibraryBooks())
    this.importBookTitleInput.addEventListener('input', () => this.validateImport())
    this.importChapterTitleInput.addEventListener('input', () => this.validateImport())
    this.importBookSelect.addEventListener('change', () => this.validateImport())
    this.importConfirmButton.addEventListener('click', () => void this.confirmImport())
    for (const control of [
      this.methodSelect,
      this.modeSelect,
      this.glossaryInput,
      this.autoTermsInput,
    ]) control.addEventListener('change', () => this.emitPreference())
    this.methodSelect.addEventListener('change', () => this.updateDiscoverLabel())
    this.settingsDetails.addEventListener('toggle', () => this.reclampPanel())
    this.pageActionsDetails.addEventListener('toggle', () => this.reclampPanel())
    this.termsSection.addEventListener('toggle', () => this.reclampPanel())

    header.addEventListener('pointerdown', event => this.startDrag(event, header))
    header.addEventListener('pointermove', event => this.moveDrag(event))
    header.addEventListener('pointerup', event => this.finishDrag(event, header))
    header.addEventListener('pointercancel', event => this.finishDrag(event, header))
    window.addEventListener('resize', this.resizeHandler)
    window.addEventListener('pointermove', this.fabPointerMoveHandler)
    window.addEventListener('pointerup', this.fabPointerEndHandler)
    window.addEventListener('pointercancel', this.fabPointerEndHandler)

    if (preference.panelOpen && preference.panelPosition) {
      this.afterLayout(() => {
        this.preference = {
          ...this.preference,
          panelPosition: this.placePanel(preference.panelPosition!),
        }
      })
    }
    if (preference.fabPosition) {
      this.afterLayout(() => {
        this.preference = {
          ...this.preference,
          fabPosition: this.placeFab(preference.fabPosition!),
        }
      })
    }
  }

  private selectField(
    label: string,
    options: Array<[string, string]>,
    value: string,
  ): HTMLSelectElement {
    const field = element('label', 'saber-field')
    const text = element('span', 'saber-label')
    text.textContent = label
    const select = element('select', 'saber-select')
    for (const [optionValue, optionLabel] of options) {
      const option = element('option')
      option.value = optionValue
      option.textContent = optionLabel
      select.append(option)
    }
    select.value = value
    field.append(text, select)
    return select
  }

  private checkField(labelText: string, checked: boolean): HTMLInputElement {
    const label = element('label', 'saber-check')
    const input = element('input')
    input.type = 'checkbox'
    input.checked = checked
    const text = element('span')
    text.textContent = labelText
    label.append(input, text)
    return input
  }

  private textField(labelText: string): HTMLInputElement {
    const label = element('label', 'saber-input-field')
    const text = element('span', 'saber-label')
    text.textContent = labelText
    const input = element('input', 'saber-input')
    input.type = 'text'
    input.maxLength = 500
    label.append(text, input)
    return input
  }

  private button(label: string, modifier = ''): HTMLButtonElement {
    const button = element('button', `saber-button ${modifier}`.trim())
    button.type = 'button'
    button.textContent = label
    return button
  }

  private linkButton(label: string): HTMLButtonElement {
    const button = element('button', 'saber-link-button')
    button.type = 'button'
    button.textContent = label
    return button
  }

  private method(): DetectionMethod {
    return this.methodSelect.value as DetectionMethod
  }

  private updateDiscoverLabel(): void {
    this.discoverButton.textContent = this.method() === 'similar'
      ? '选择一张漫画图片'
      : '识别漫画图片'
  }

  private emitPreference(): void {
    this.preference = {
      ...this.preference,
      method: this.method(),
      mode: this.modeSelect.value as TranslationMode,
      glossaryEnabled: this.glossaryInput.checked,
      autoTermsEnabled: this.autoTermsInput.checked,
    }
    this.callbacks.onPreferenceChange({ ...this.preference })
  }

  private setAllCandidates(checked: boolean): void {
    for (const input of this.candidatesGrid.querySelectorAll<HTMLInputElement>('input')) {
      input.checked = checked
    }
  }

  private async openImportDialog(): Promise<void> {
    const session = this.latestSession
    if (!session || this.importButton.disabled || this.importing) return
    this.importButton.disabled = true
    try {
      this.libraryBooks = await this.callbacks.onLoadLibraryBooks()
      this.importBookTitleInput.value = this.pageLabel
      this.importChapterTitleInput.value = this.pageLabel
      this.importSearchInput.value = ''
      this.importNewInput.checked = true
      this.importExistingInput.checked = false
      this.importExistingInput.disabled = this.libraryBooks.length === 0
      this.renderLibraryBooks()
      this.updateImportDestination()
      const imported = session.pages.filter(page => page.pageId !== null).length
      const omitted = session.pages.length - imported
      this.importSummary.textContent = omitted
        ? `${imported} 张已进入任务，${omitted} 张未导入的图片不会加入书架。`
        : `${imported} 张图片将作为一个独立章节加入书架。`
      this.importOverlay.dataset.open = 'true'
      this.importBookTitleInput.focus()
    } catch {
      this.importButton.disabled = false
    }
  }

  private closeImportDialog(): void {
    this.importOverlay.dataset.open = 'false'
    if (this.latestSession) {
      const busy = this.latestSession.pages.some(page => (
        page.state === 'queued' || page.state === 'translating'
      ))
      this.importButton.disabled = busy
        || !this.latestSession.pages.some(page => page.pageId !== null)
    }
  }

  private updateImportDestination(): void {
    const existing = this.importExistingInput.checked
    this.importNewFields.hidden = existing
    this.importExistingFields.hidden = !existing
    this.validateImport()
  }

  private renderLibraryBooks(): void {
    const previous = this.importBookSelect.value
    const search = this.importSearchInput.value.trim().toLocaleLowerCase()
    const matching = this.libraryBooks.filter(book => (
      !search || book.title.toLocaleLowerCase().includes(search)
    ))
    this.importBookSelect.replaceChildren()
    for (const book of matching) {
      const option = element('option')
      option.value = book.id
      option.textContent = `${book.title} · ${book.chapterCount} 章`
      this.importBookSelect.append(option)
    }
    if (matching.some(book => book.id === previous)) {
      this.importBookSelect.value = previous
    }
    this.importBookSelect.disabled = matching.length === 0
    this.validateImport()
  }

  private validateImport(): void {
    const chapterValid = this.importChapterTitleInput.value.trim().length > 0
    const destinationValid = this.importExistingInput.checked
      ? Boolean(this.importBookSelect.value) && !this.importBookSelect.disabled
      : this.importBookTitleInput.value.trim().length > 0
    this.importConfirmButton.disabled = this.importing || !chapterValid || !destinationValid
  }

  private async confirmImport(): Promise<void> {
    if (this.importConfirmButton.disabled) return
    const chapterTitle = this.importChapterTitleInput.value.trim()
    const command: BrowserSessionImportCommand = this.importExistingInput.checked
      ? {
          destination: 'existing',
          targetBookId: this.importBookSelect.value,
          chapterTitle,
        }
      : {
          destination: 'new',
          bookTitle: this.importBookTitleInput.value.trim(),
          chapterTitle,
        }
    this.importing = true
    this.importConfirmButton.disabled = true
    this.importConfirmButton.textContent = '正在导入…'
    try {
      const result = await this.callbacks.onImport(command)
      this.closeImportDialog()
      this.showImported(result)
    } catch {
      this.closeImportDialog()
    } finally {
      this.importing = false
      this.importConfirmButton.textContent = '确认导入'
      this.validateImport()
    }
  }

  private setView(view: PanelView): void {
    this.panel.dataset.view = view
    if (view !== 'progress') this.preparationSection.hidden = true
    this.idleActions.hidden = view !== 'idle'
    this.candidatesSection.hidden = view !== 'candidates'
    this.progressSection.hidden = view !== 'progress'
    this.termsSection.hidden = view !== 'progress'
    this.settingsDetails.open = view === 'idle'
    this.reclampPanel()
  }

  private showIdle(): void {
    this.setView('idle')
    this.setStatus('准备识别漫画图片', '确认设置后重新识别当前页面。')
  }

  private afterLayout(callback: () => void): void {
    if (typeof window.requestAnimationFrame === 'function') {
      window.requestAnimationFrame(() => callback())
      return
    }
    window.setTimeout(callback, 0)
  }

  private reclampPanel(): void {
    if (this.panel.dataset.open !== 'true' || !this.preference.panelPosition) return
    this.afterLayout(() => {
      this.preference = {
        ...this.preference,
        panelPosition: this.placePanel(this.preference.panelPosition!),
      }
    })
  }

  private placeFixedElement(
    target: HTMLElement,
    position: PanelPosition,
    fallbackWidth: number,
    fallbackHeight: number,
  ): PanelPosition {
    const rect = target.getBoundingClientRect()
    const width = rect.width || target.offsetWidth || fallbackWidth
    const height = rect.height || target.offsetHeight || fallbackHeight
    const x = Math.min(
      Math.max(PANEL_MARGIN, position.x),
      Math.max(PANEL_MARGIN, window.innerWidth - width - PANEL_MARGIN),
    )
    const y = Math.min(
      Math.max(PANEL_MARGIN, position.y),
      Math.max(PANEL_MARGIN, window.innerHeight - height - PANEL_MARGIN),
    )
    target.style.left = `${x}px`
    target.style.top = `${y}px`
    target.style.right = 'auto'
    target.style.bottom = 'auto'
    return { x, y }
  }

  private placePanel(position: PanelPosition): PanelPosition {
    return this.placeFixedElement(this.panel, position, 336, 480)
  }

  private placeFab(position: PanelPosition): PanelPosition {
    return this.placeFixedElement(this.fab, position, 46, 46)
  }

  private startDrag(event: PointerEvent, header: HTMLElement): void {
    if (event.button !== 0 || (event.target as Element | null)?.closest('button')) return
    const rect = this.panel.getBoundingClientRect()
    this.dragState = {
      pointerId: event.pointerId,
      startClientX: event.clientX,
      startClientY: event.clientY,
      startElementX: rect.left,
      startElementY: rect.top,
      lastPosition: { x: rect.left, y: rect.top },
      moved: false,
    }
    header.setPointerCapture?.(event.pointerId)
    event.preventDefault()
  }

  private moveDrag(event: PointerEvent): void {
    const drag = this.dragState
    if (!drag || drag.pointerId !== event.pointerId) return
    const next = this.placePanel({
      x: drag.startElementX + event.clientX - drag.startClientX,
      y: drag.startElementY + event.clientY - drag.startClientY,
    })
    drag.lastPosition = next
    drag.moved ||= Math.abs(event.clientX - drag.startClientX) > 2
      || Math.abs(event.clientY - drag.startClientY) > 2
    event.preventDefault()
  }

  private finishDrag(event: PointerEvent, header: HTMLElement): void {
    const drag = this.dragState
    if (!drag || drag.pointerId !== event.pointerId) return
    if (header.hasPointerCapture?.(event.pointerId)) header.releasePointerCapture(event.pointerId)
    this.dragState = null
    if (!drag.moved) return
    this.preference = { ...this.preference, panelPosition: drag.lastPosition }
    this.callbacks.onPanelPositionChange(drag.lastPosition)
  }

  private startFabDrag(event: PointerEvent): void {
    if (event.button !== 0) return
    const rect = this.fab.getBoundingClientRect()
    this.fabDragState = {
      pointerId: event.pointerId,
      startClientX: event.clientX,
      startClientY: event.clientY,
      startElementX: rect.left,
      startElementY: rect.top,
      lastPosition: { x: rect.left, y: rect.top },
      moved: false,
    }
    this.fab.dataset.dragging = 'true'
    this.fab.setPointerCapture?.(event.pointerId)
    event.preventDefault()
  }

  private moveFabDrag(event: PointerEvent): void {
    const drag = this.fabDragState
    if (!drag || drag.pointerId !== event.pointerId) return
    const moved = Math.abs(event.clientX - drag.startClientX) > 3
      || Math.abs(event.clientY - drag.startClientY) > 3
    if (!moved && !drag.moved) return
    drag.moved = true
    drag.lastPosition = this.placeFab({
      x: drag.startElementX + event.clientX - drag.startClientX,
      y: drag.startElementY + event.clientY - drag.startClientY,
    })
    event.preventDefault()
  }

  private finishFabDrag(event: PointerEvent): void {
    const drag = this.fabDragState
    if (!drag || drag.pointerId !== event.pointerId) return
    if (this.fab.hasPointerCapture?.(event.pointerId)) this.fab.releasePointerCapture(event.pointerId)
    this.fab.dataset.dragging = 'false'
    this.fabDragState = null
    if (!drag.moved) return
    this.preference = { ...this.preference, fabPosition: drag.lastPosition }
    this.callbacks.onFabPositionChange(drag.lastPosition)
    this.suppressFabClick = true
    window.setTimeout(() => {
      this.suppressFabClick = false
    }, 0)
  }

  setOpen(open: boolean, persist = false): void {
    this.panel.dataset.open = String(open)
    if (open && this.preference.panelPosition) {
      this.afterLayout(() => {
        this.preference = {
          ...this.preference,
          panelPosition: this.placePanel(this.preference.panelPosition!),
        }
      })
    }
    if (!persist) return
    this.preference = { ...this.preference, panelOpen: open }
    this.callbacks.onPanelOpenChange(open)
  }

  togglePanel(): void {
    this.setOpen(this.panel.dataset.open !== 'true', true)
  }

  setAdaptationSaved(saved: boolean): void {
    this.deleteAdaptationButton.hidden = !saved
    if (!saved) delete this.preference.rule
    this.reclampPanel()
  }

  showCandidates(candidates: ImageCandidate[]): void {
    this.candidatesGrid.replaceChildren()
    for (const candidate of candidates) {
      const card = element('label', 'saber-candidate')
      const fallback = (): HTMLDivElement => {
        const result = element('div', 'saber-candidate__fallback')
        result.textContent = candidate.kind === 'canvas' ? 'Canvas 漫画页' : '页面图片'
        return result
      }
      if (candidate.sourceUrl) {
        const image = element('img')
        image.src = candidate.sourceUrl
        image.loading = 'lazy'
        image.decoding = 'async'
        image.alt = ''
        image.addEventListener('error', () => image.replaceWith(fallback()), { once: true })
        card.append(image)
      } else {
        card.append(fallback())
      }
      const input = element('input')
      input.type = 'checkbox'
      input.value = candidate.id
      input.checked = true
      const meta = element('span', 'saber-candidate__meta')
      meta.textContent = `${candidate.width} × ${candidate.height}`
      card.append(input, meta)
      this.candidatesGrid.append(card)
    }
    this.candidateCount.textContent = `${candidates.length} 张`
    this.confirmButton.disabled = candidates.length === 0
    this.setView(candidates.length ? 'candidates' : 'idle')
    this.setOpen(true)
    this.setStatus(
      candidates.length ? '请确认需要翻译的图片' : '没有找到可用漫画图片',
      candidates.length
        ? '确认后，后续匹配同一规则的懒加载图片会自动加入。'
        : '可以滚动页面后重试、切换识别方式或使用右键单图翻译。',
      candidates.length ? 'ready' : 'error',
    )
  }

  showSession(
    session: BrowserSessionDto,
    originalPageIds: ReadonlySet<string> = new Set(),
  ): void {
    this.latestSession = session
    this.retryStartButton.hidden = true
    this.preparationSection.hidden = true
    this.setView('progress')
    if (session.state === 'cancelled' && session.pages.length === 0) this.setView('idle')
    this.toggleButton.hidden = false
    this.stopButton.hidden = false
    this.cancelButton.hidden = false
    this.importButton.hidden = false
    for (const [key, elementValue] of this.statElements) {
      elementValue.textContent = String(session.counts[key as keyof typeof session.counts] ?? 0)
    }
    this.pageActions.replaceChildren()
    const actionablePages = session.pages.filter(page => (
      (page.state === 'completed' && page.resultReady)
      || page.state === 'failed'
      || page.state === 'cancelled'
    ))
    this.pageActionsDetails.hidden = actionablePages.length === 0
    this.pageActionsSummary.textContent = `单页操作 · ${actionablePages.length}`
    for (const page of actionablePages) {
      const row = element(
        'div',
        `saber-page-action${page.state === 'failed' ? ' saber-page-action--error' : ''}`,
      )
      const message = element('span', 'saber-page-action__label')
      message.textContent = page.state === 'failed'
        ? `第 ${page.ordinal} 张：${page.error?.message ?? '翻译失败'}`
        : page.state === 'cancelled'
          ? `第 ${page.ordinal} 张：已取消`
          : `第 ${page.ordinal} 张`
      const actions = element('span', 'saber-page-action__buttons')
      if (page.state === 'completed') {
        const toggle = this.button(
          originalPageIds.has(page.id) ? '显示译图' : '查看原图',
          'saber-button--quiet',
        )
        toggle.addEventListener('click', async () => {
          toggle.disabled = true
          try {
            const showingTranslated = await this.callbacks.onTogglePage(page.id)
            if (showingTranslated !== null) {
              toggle.textContent = showingTranslated ? '查看原图' : '显示译图'
            }
          } catch {
            // The controller already presents the actionable error in the panel.
          } finally {
            toggle.disabled = false
          }
        })
        actions.append(toggle)
      }
      const retry = this.button(page.state === 'completed' ? '重翻' : '重试')
      retry.disabled = session.pages.some(item => item.state === 'queued' || item.state === 'translating')
      retry.addEventListener('click', () => this.callbacks.onRetryPage(page.id))
      actions.append(retry)
      row.append(message, actions)
      this.pageActions.append(row)
    }
    const busy = session.pages.some(page => (
      page.state === 'queued' || page.state === 'translating'
    ))
    const hasFailure = session.state === 'failed' || session.state === 'partial'
    this.toggleButton.disabled = session.counts.completed === 0
    this.cancelButton.disabled = !busy && session.state !== 'idle'
    this.importButton.disabled = busy || !session.pages.some(page => page.pageId !== null)
    this.importButton.title = busy
      ? '等待当前翻译批次结束后再导入'
      : this.importButton.disabled
        ? '至少需要一张已经进入 Saber 的图片'
        : '将当前独立章节导入书架'
    this.fab.dataset.state = hasFailure ? 'error' : busy ? 'busy' : 'ready'
    this.setStatus(
      {
        idle: '等待图片',
        queued: '任务已进入 Saber 队列',
        translating: '正在逐张生成译图',
        completed: '当前图片已全部完成',
        partial: '部分图片翻译失败',
        failed: '图片翻译失败',
        cancelled: '任务已取消',
      }[session.state],
      session.state === 'cancelled'
        ? busy ? '正在等待后端任务停止。' : '可在单页操作中重试已取消的图片。'
        : '关闭标签页不会中断已提交的后端任务。',
      hasFailure ? 'error' : busy ? 'busy' : 'ready',
    )
    if (session.pendingStart) {
      this.showStartError({ code: 'pending_start', message: '图片已准备好，等待启动翻译' })
    }
  }

  showPreparationProgress(processed: number, total: number, failed: number): void {
    const safeTotal = Math.max(0, Math.floor(total))
    const safeProcessed = Math.min(safeTotal, Math.max(0, Math.floor(processed)))
    const safeFailed = Math.min(safeProcessed, Math.max(0, Math.floor(failed)))
    this.preparationCount.textContent = `${safeProcessed} / ${safeTotal}`
    this.preparationMeter.max = Math.max(1, safeTotal)
    this.preparationMeter.value = safeProcessed
    this.preparationDetail.textContent = `成功 ${safeProcessed - safeFailed} · 失败 ${safeFailed}`
    this.preparationSection.hidden = false
    this.setView('progress')
    this.setOpen(true)
    this.setStatus(
      '正在准备漫画图片',
      '正在从网页读取图片并发送到本机 Saber，完成后会自动开始翻译。',
      'busy',
    )
  }

  hidePreparationProgress(): void {
    this.preparationSection.hidden = true
    this.reclampPanel()
  }

  showImported(result: BrowserSessionImportResult): void {
    this.importOverlay.dataset.open = 'false'
    this.pageActionsDetails.hidden = true
    this.stopButton.hidden = true
    this.cancelButton.hidden = true
    this.importButton.hidden = true
    this.fab.dataset.state = 'ready'
    const details = result.omittedPages
      ? `已导入 ${result.importedPages} 张，忽略 ${result.omittedPages} 张未进入任务的图片。`
      : `已导入 ${result.importedPages} 张图片。`
    this.setStatus(
      `已导入《${result.bookTitle}》`,
      `${result.chapterTitle} · ${details}`,
      'ready',
    )
  }

  showTerms(entries: Array<{ source?: string; target?: string }>): void {
    this.termsElement.replaceChildren()
    this.termsSummary.textContent = entries.length ? `实时术语 · ${entries.length}` : '实时术语'
    if (!entries.length) {
      this.termsElement.textContent = '尚未提取术语'
      return
    }
    for (const entry of entries) {
      const row = element('div')
      row.textContent = `${entry.source ?? ''} → ${entry.target ?? ''}`
      this.termsElement.append(row)
    }
  }

  setStatus(
    title: string,
    message: string,
    tone: 'ready' | 'busy' | 'error' = 'ready',
  ): void {
    this.bannerTitle.textContent = title
    this.bannerMessage.textContent = message
    this.banner.dataset.tone = tone === 'error' ? 'error' : ''
    this.errorActions.hidden = tone !== 'error'
    this.fab.dataset.state = tone
    this.reclampPanel()
  }

  showError(error: { code: string; message: string }): void {
    const hints: Record<string, string> = {
      saber_unreachable: '请启动 Saber，并确认扩展端口与 GUI 一致。',
      not_paired: '请点击浏览器工具栏中的扩展图标完成配对。',
      invalid_extension_token: '令牌已失效，请从 Saber GUI 重新复制并配对。',
      integration_disabled: '请在 Saber GUI 中打开“允许浏览器扩展连接”。',
      source_forbidden: '先在当前网页完成登录或 Cloudflare 验证，然后重试。',
      canvas_unreadable: '该 Canvas 受跨域保护，当前版本无法读取。',
      dom_agent_unavailable: '请先在 Saber 设置的“网页漫画”中保存 Agent 配置。',
      result_expired: '扩展会自动申请新凭证并重新读取译图。',
      result_fetch_failed: '请确认 Saber 仍在运行，然后重试该图片。',
      result_too_large: '可将该章节导入书架后，在 Saber 中查看这张译图。',
      request_timeout: '请确认 Saber 仍在运行，然后重试。',
      source_timeout: '源站响应过慢，可确认网页图片已经加载后重试上传。',
    }
    this.retryUploadsButton.hidden = true
    this.retryStartButton.hidden = true
    this.setStatus(error.message, hints[error.code] ?? '可重试或切换另一种图片识别方式。', 'error')
    this.setOpen(true)
  }

  showStartError(error: { code: string; message: string }): void {
    this.showError(error)
    this.retryStartButton.hidden = false
    this.retryStartButton.disabled = false
  }

  showUploadError(
    count: number,
    error: { code: string; message: string },
  ): void {
    this.retryUploadsButton.hidden = false
    this.setStatus(
      `${count} 张图片尚未导入`,
      `${error.message}。其他已成功导入的图片会继续处理。`,
      'error',
    )
    this.setOpen(true)
  }

  clearUploadError(): void {
    this.retryUploadsButton.hidden = true
  }

  startPicking(): void {
    this.pickMask.dataset.open = 'true'
  }

  stopPicking(): void {
    this.pickMask.dataset.open = 'false'
  }

  pickingMask(): HTMLElement {
    return this.pickMask
  }

  remove(): void {
    window.removeEventListener('resize', this.resizeHandler)
    window.removeEventListener('pointermove', this.fabPointerMoveHandler)
    window.removeEventListener('pointerup', this.fabPointerEndHandler)
    window.removeEventListener('pointercancel', this.fabPointerEndHandler)
    this.host.remove()
  }
}
