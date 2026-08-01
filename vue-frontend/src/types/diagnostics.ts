import type { components } from '@/api/generated/v2'

export type FetchModelsResponse = components['schemas']['ModelCatalogResponse']
export type ModelInfoItem = FetchModelsResponse['models'][number]
