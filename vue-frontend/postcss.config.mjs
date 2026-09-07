import { resolve } from 'node:path'

export default {
  plugins: {
    '@csstools/postcss-global-data': {
      files: [resolve(import.meta.dirname, 'src/styles/tokens/foundation.css')],
    },
    'postcss-custom-media': {},
  },
}
