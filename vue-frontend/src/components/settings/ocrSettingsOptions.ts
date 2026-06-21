import { getProviderOptionsForCapability } from '@/config/aiProviders'

/** OCR引擎选项 */
export const allOcrEngineOptions = [
  { label: 'MangaOCR (日语专用)', value: 'manga_ocr' },
  { label: 'PaddleOCR (多语言)', value: 'paddle_ocr' },
  { label: 'PaddleOCR-VL', value: 'paddleocr_vl' },
  { label: '百度OCR', value: 'baidu_ocr' },
  { label: '48px OCR', value: '48px_ocr' },
  { label: 'AI视觉OCR', value: 'ai_vision' }
]

/** 百度OCR版本选项 */
export const baiduVersionOptions = [
  { label: '标准版', value: 'standard' },
  { label: '高精度版', value: 'high_precision' }
]

/** 百度OCR源语言选项 */
export const baiduSourceLanguageOptions = [
  { label: '自动检测', value: 'auto_detect' },
  { label: '中英文混合', value: 'CHN_ENG' },
  { label: '英文', value: 'ENG' },
  { label: '日语', value: 'JAP' },
  { label: '韩语', value: 'KOR' },
  { label: '法语', value: 'FRE' },
  { label: '德语', value: 'GER' },
  { label: '俄语', value: 'RUS' }
]

/** AI视觉服务商选项 */
export const aiVisionProviderOptions = getProviderOptionsForCapability('visionOcr')

/** PaddleOCR-VL 源语言选项（分组） */
export const paddleOcrVlSourceLanguageGroups = [
  {
    label: '🎌 东亚语言',
    options: [
      { label: '日语', value: 'japanese' },
      { label: '简体中文', value: 'chinese' },
      { label: '繁体中文', value: 'chinese_cht' },
      { label: '韩语', value: 'korean' }
    ]
  },
  {
    label: '🌍 拉丁语系',
    options: [
      { label: '英语', value: 'english' },
      { label: '法语', value: 'french' },
      { label: '德语', value: 'german' },
      { label: '西班牙语', value: 'spanish' },
      { label: '意大利语', value: 'italian' },
      { label: '葡萄牙语', value: 'portuguese' },
      { label: '荷兰语', value: 'dutch' },
      { label: '波兰语', value: 'polish' }
    ]
  },
  {
    label: '🌏 东南亚语言',
    options: [
      { label: '泰语', value: 'thai' },
      { label: '越南语', value: 'vietnamese' },
      { label: '印尼语', value: 'indonesian' },
      { label: '马来语', value: 'malay' }
    ]
  },
  {
    label: '🌐 其他语系',
    options: [
      { label: '俄语', value: 'russian' },
      { label: '阿拉伯语', value: 'arabic' },
      { label: '印地语', value: 'hindi' },
      { label: '土耳其语', value: 'turkish' },
      { label: '希腊语', value: 'greek' },
      { label: '希伯来语', value: 'hebrew' }
    ]
  }
]

/** 提示词模式选项 */
export const promptModeOptions = [
  { label: '普通提示词', value: 'normal' },
  { label: 'JSON提示词', value: 'json' },
  { label: 'OCR模型提示词', value: 'paddleocr_vl' }
]

/** 源语言选项（分组） */
export const sourceLanguageGroups = [
  {
    label: '🚀 常用语言',
    options: [
      { label: '日语', value: 'japanese' },
      { label: '英语', value: 'en' },
      { label: '简体中文', value: 'chinese' },
      { label: '繁体中文', value: 'chinese_cht' },
      { label: '韩语', value: 'korean' }
    ]
  },
  {
    label: '🌍 拉丁语系',
    options: [
      { label: '法语', value: 'french' },
      { label: '德语', value: 'german' },
      { label: '西班牙语', value: 'spanish' },
      { label: '意大利语', value: 'italian' },
      { label: '葡萄牙语', value: 'portuguese' }
    ]
  },
  {
    label: '🌏 其他语系',
    options: [
      { label: '俄语', value: 'russian' }
    ]
  }
]
