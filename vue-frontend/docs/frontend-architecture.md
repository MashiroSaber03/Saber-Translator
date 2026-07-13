# Frontend Architecture

This document is the current architecture contract for the Vue frontend. It
describes the system as it is maintained now, not the history of its refactor.

## Ownership Layers

- Views assemble route data, choose a shell, and coordinate page-level
  workflows.
- Product shells own reusable page geometry such as headers, split panes,
  tabbed workspaces, grids, and scrolling regions.
- Business components own a coherent visual or interaction region and keep
  their styles in the same scoped SFC.
- UI primitives own generic controls and expose typed props, events, slots,
  and documented CSS custom properties.
- Stores own durable domain state. Focused helpers own schema parsing,
  lifecycle listeners, model discovery, animation, or streaming when those
  responsibilities can be understood and tested independently.

High-interaction owners remain cohesive when splitting would only create deep
prop/event plumbing. File size is a review signal, never the reason to split.

## State And Data Boundaries

- Backend wire payloads remain snake_case at the API boundary.
- Stores and components use the current application schema; retired settings,
  provider, font, and localStorage compatibility shapes are not read.
- Settings normalization and theme preference lifecycle live under
  `src/stores/settings/` while the Pinia facade owns persistence and backend
  synchronization.
- Character Studio streaming lives in
  `src/stores/characterStudio/useCharacterStudioChat.ts`; the store facade
  retains document, session, import/export, and Agent orchestration.
- Shared AI model loading uses `useAiModelDiscovery`. Provider selection and
  credentials use the shared settings field components so validation,
  loading, accessibility, and dark-theme behavior stay consistent.

## Styling And Tokens

- Global tokens are split into foundation, semantic, component, and domain
  layers and must have real production consumers.
- Stable cross-owner roles use global tokens. Owner-local spacing, sizing,
  radius, and shadow geometry may remain literal inside scoped styles.
- Raw business colors, numeric z-index values, raw pixel media queries,
  primitive-internal selectors, `:deep()`, and `:global()` are prohibited by
  `npm run lint:ui`.
- Primitive public CSS variables are consumed with fallbacks. Variants set
  private fallback variables so owner overrides remain authoritative.
- Teleport and body-state CSS is the only supported use of namespaced
  `*.global.styles.css` files.

## Verification

Run these checks for every frontend change:

```bash
npm run lint:ui
npm run lint:css
npm run lint
npm run typecheck
```

Run `npm test` for behavior changes and `npm run visual:test` for layout,
theme, modal, form, token, or primitive changes. Before release, also run
`npm run build`, `npm run lint:ui:audit`, and `git diff --check`.

The production build may report the upstream `pdfjs-dist` eval warning. It is
non-blocking while the application remains on the current PDF.js version; any
additional dynamic/static import warning is an application architecture issue
and must be investigated.
