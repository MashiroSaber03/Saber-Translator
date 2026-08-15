import type { InpaintMethod, TextAlign, TextDirection } from './bubble'

export interface TextStyleSettings {
  fontSize: number
  autoFontSize: boolean
  fontFamily: string
  layoutDirection: TextDirection
  textColor: string
  fillColor: string
  strokeEnabled: boolean
  strokeColor: string
  strokeWidth: number
  inpaintMethod: InpaintMethod
  useAutoTextColor: boolean
  lineSpacing: number
  textAlign: TextAlign
}

export type TextStyleMutationField = Exclude<
  keyof TextStyleSettings,
  'autoFontSize' | 'useAutoTextColor'
>

export type TextStyleMutationArgs = {
  [Field in TextStyleMutationField]: [
    settingKey: Field,
    newValue: TextStyleSettings[Field],
  ]
}[TextStyleMutationField]
