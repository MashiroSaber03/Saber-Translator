import type { components } from '@/api/generated/v2'

export type WorkflowMode = components['schemas']['WorkflowPreferences']['lastWorkflowMode']

export interface WorkflowPageSelection {
  pages: number[]
}

export interface WorkflowRunRequest {
  mode: WorkflowMode
  pageSelection?: WorkflowPageSelection
}
