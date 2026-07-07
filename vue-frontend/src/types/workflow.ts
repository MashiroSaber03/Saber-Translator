export type WorkflowMode =
  | 'translate-current'
  | 'translate-batch'
  | 'hq-batch'
  | 'proofread-batch'
  | 'remove-current'
  | 'remove-batch'
  | 'retry-failed'
  | 'delete-current'
  | 'clear-all'

export interface WorkflowPageSelection {
  pages: number[]
}

export interface WorkflowRunRequest {
  mode: WorkflowMode
  pageSelection?: WorkflowPageSelection
}
