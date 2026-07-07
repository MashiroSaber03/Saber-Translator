export type UiSelectValue = string | number

export interface UiSelectOption {
  label: string
  value: UiSelectValue
  disabled?: boolean
}

export interface UiSelectGroup {
  label: string
  options: UiSelectOption[]
}
