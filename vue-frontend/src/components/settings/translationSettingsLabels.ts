export function getTranslationApiKeyLabel(provider: string): string {
  switch (provider) {
    case 'baidu_translate':
      return 'App ID'
    case 'youdao_translate':
      return 'App Key'
    case 'caiyun':
      return 'API Token'
    default:
      return 'API Key'
  }
}

export function getTranslationApiKeyPlaceholder(provider: string): string {
  switch (provider) {
    case 'baidu_translate':
      return '请输入百度翻译App ID'
    case 'youdao_translate':
      return '请输入有道翻译应用ID'
    case 'caiyun':
      return '请输入彩云小译Token'
    default:
      return '请输入API Key'
  }
}

export function getTranslationModelNameLabel(provider: string): string {
  switch (provider) {
    case 'baidu_translate':
      return 'App Key'
    case 'youdao_translate':
      return 'App Secret'
    case 'caiyun':
      return '源语言 (可选)'
    default:
      return '模型名称'
  }
}

export function getTranslationModelNamePlaceholder(provider: string): string {
  switch (provider) {
    case 'baidu_translate':
      return '请输入百度翻译App Key'
    case 'youdao_translate':
      return '请输入有道翻译应用密钥'
    case 'caiyun':
      return '可选: auto/日语/英语'
    default:
      return '请输入模型名称'
  }
}
