import './popup.css'
import type {
  BackgroundRequest,
  BackgroundResponse,
  DomainPreference,
} from './types'

interface PopupState {
  token: string
  serverPort: number
  hostname: string
  preference: DomainPreference | null
}

async function send<T>(request: BackgroundRequest): Promise<T> {
  const response = await chrome.runtime.sendMessage(request) as BackgroundResponse<T>
  if (!response?.ok) throw new Error(response?.error?.message ?? '扩展后台没有响应')
  return response.data
}

function create<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
): HTMLElementTagNameMap[K] {
  const result = document.createElement(tag)
  if (className) result.className = className
  return result
}

async function main(): Promise<void> {
  const app = document.querySelector<HTMLElement>('#app')
  if (!app) return
  const root = create('div', 'popup')
  const brand = create('div', 'brand')
  const logo = create('div', 'brand__logo')
  logo.textContent = 'S'
  const brandText = create('div')
  const title = create('h1')
  title.textContent = 'Saber 漫画翻译'
  const subtitle = create('p')
  subtitle.textContent = `本机扩展 · v${chrome.runtime.getManifest().version}`
  brandText.append(title, subtitle)
  brand.append(logo, brandText)
  root.append(brand)

  const statusCard = create('section', 'card status')
  statusCard.setAttribute('role', 'status')
  statusCard.setAttribute('aria-live', 'polite')
  const statusDot = create('span', 'status__dot')
  const statusText = create('span')
  statusText.textContent = '正在检查本机 Saber…'
  statusCard.dataset.tone = 'busy'
  statusCard.append(statusDot, statusText)
  root.append(statusCard)

  const connection = create('form', 'card connection')
  const tokenField = create('div', 'field')
  const tokenLabel = create('label')
  tokenLabel.htmlFor = 'saber-pairing-token'
  tokenLabel.textContent = '配对令牌'
  const tokenInput = create('input')
  tokenInput.id = 'saber-pairing-token'
  tokenInput.type = 'password'
  tokenInput.autocomplete = 'off'
  tokenInput.placeholder = '从 Saber GUI 复制令牌'
  tokenField.append(tokenLabel, tokenInput)
  const portField = create('div', 'field')
  const portLabel = create('label')
  portLabel.htmlFor = 'saber-server-port'
  portLabel.textContent = '本机端口'
  const portInput = create('input')
  portInput.id = 'saber-server-port'
  portInput.type = 'number'
  portInput.min = '1'
  portInput.max = '65535'
  portField.append(portLabel, portInput)
  const connectionRow = create('div', 'row')
  connectionRow.append(tokenField, portField)
  const actions = create('div', 'actions')
  const connectButton = create('button', 'primary')
  connectButton.type = 'submit'
  connectButton.textContent = '保存并连接'
  actions.append(connectButton)
  connection.append(connectionRow, actions)
  root.append(connection)

  const siteCard = create('section', 'card site')
  const siteText = create('div')
  const siteName = create('strong')
  const siteHint = create('span')
  siteText.append(siteName, siteHint)
  const siteButton = create('button')
  siteButton.type = 'button'
  siteCard.append(siteText, siteButton)
  root.append(siteCard)

  const footer = create('div', 'footer')
  footer.textContent = '图片只发送到你本机运行的 Saber-Translator'
  root.append(footer)
  app.append(root)

  let state = await send<PopupState>({ type: 'get-popup-state' })
  tokenInput.value = state.token
  portInput.value = String(state.serverPort)

  function renderSite(): void {
    if (!state.hostname || !state.preference) {
      siteName.textContent = '当前页面不支持注入扩展'
      siteHint.textContent = '请打开普通 HTTP(S) 网页'
      siteButton.hidden = true
      return
    }
    siteName.textContent = state.hostname
    siteHint.textContent = state.preference.disabled ? '该站点已停用' : '该站点已启用'
    siteButton.hidden = false
    siteButton.textContent = state.preference.disabled ? '重新启用' : '停用'
  }

  async function checkStatus(): Promise<void> {
    connectButton.disabled = true
    connectButton.textContent = '正在连接…'
    statusCard.dataset.tone = 'busy'
    statusText.textContent = '正在检查本机 Saber…'
    try {
      await send({ type: 'status' })
      statusCard.dataset.tone = 'success'
      statusText.textContent = '已连接 Saber，本机接口可用'
    } catch (error) {
      statusCard.dataset.tone = 'error'
      statusText.textContent = error instanceof Error ? error.message : '无法连接 Saber'
    } finally {
      connectButton.disabled = false
      connectButton.textContent = '保存并连接'
    }
  }

  connection.addEventListener('submit', async (event) => {
    event.preventDefault()
    connectButton.disabled = true
    try {
      await send({
        type: 'save-connection',
        token: tokenInput.value,
        serverPort: Number(portInput.value),
      })
      state = await send<PopupState>({ type: 'get-popup-state' })
      await checkStatus()
    } catch (error) {
      statusCard.dataset.tone = 'error'
      statusText.textContent = error instanceof Error ? error.message : '保存失败'
    } finally {
      connectButton.disabled = false
      connectButton.textContent = '保存并连接'
    }
  })
  siteButton.addEventListener('click', async () => {
    if (!state.hostname || !state.preference) return
    const preference = {
      ...state.preference,
      disabled: !state.preference.disabled,
    }
    try {
      await send({
        type: 'set-preference',
        hostname: state.hostname,
        preference,
      })
      state.preference = preference
      renderSite()
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
      if (tab?.id) await chrome.tabs.reload(tab.id)
    } catch (error) {
      statusCard.dataset.tone = 'error'
      statusText.textContent = error instanceof Error ? error.message : '站点设置保存失败'
    }
  })

  renderSite()
  await checkStatus()
}

void main().catch((error) => {
  const status = document.querySelector<HTMLElement>('.status')
  const message = status?.querySelector<HTMLElement>('span:last-child')
  if (status) status.dataset.tone = 'error'
  if (message) {
    message.textContent = error instanceof Error ? error.message : '扩展后台没有响应'
  }
})
