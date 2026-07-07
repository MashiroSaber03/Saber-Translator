export interface CharacterStudioEditorPendingState {
  generatingSection: string | null
  validating: boolean
  importingWorldbook: boolean
  deleting: boolean
  saving: boolean
  downloadingFormat: string | null
}
