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
- Stores project backend-owned domain state. Focused helpers own schema parsing,
  lifecycle listeners, model discovery, animation, or streaming when those
  responsibilities can be understood and tested independently.

High-interaction owners remain cohesive when splitting would only create deep
prop/event plumbing. File size is a review signal, never the reason to split.

## State And Data Boundaries

- `openapi/v2.yaml` is the only HTTP contract; generated clients keep wire
  details at the API boundary.
- Books, chapters, pages, documents, jobs, analyses, studio documents and
  plugin revisions are backend facts. Pinia never becomes their durable owner.
- The browser persists UI preferences. In `public` mode, provider credentials
  are also stored per user in IndexedDB by `src/services/browserCredentials.ts`
  and uploaded to the Launcher-owned memory broker. This exception does not
  make the browser the durable owner of documents, tasks, or results.
- In `local` mode, the backend stores versioned credentials and returns the
  current secrets in the settings document for form editing.
- Translation, analysis, PDF parsing, export and plugin loops run in the
  backend. The frontend creates commands and projects REST/SSE state.
- Thumbnail collections are lazy. Full assets are limited to the active page
  or reader visibility window and are released when they leave that window.
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

## Editing A Page

- A page has one editable text draft: `imageStore`'s `bubbleStates`. `bubbleStore`
  provides selection and editing operations over that same array. Changing modes
  never copies a second draft back into the page. Accepted projections retain a
  valid text-box selection; loading a document does not force it to the first box.
- `pageDocumentPersistence` alone accepts document projections and publishes
  committed revisions. It rejects older reads, preserves edits made during a save,
  and coalesces ordinary mutations, page defaults and native snapshots through
  the existing atomic document API. Its bounded snapshots are transport state,
  not another editable document.
- The sidebar and ordinary workspace submit data changes. Fine editing uses the
  shared Rust engine in WASM for immediate interaction and submits its native
  snapshot through that same queue. The backend native document owns translated
  text, character runs and graphics; database bubbles retain business data and
  the native read projection in the same commit.
- Saving and generating the published image are distinct outcomes. The common
  render observer updates result assets and reports failed output without
  discarding a saved document or claiming that the preview succeeded.
- Ordinary page undo keeps bounded native snapshots across mode switches and
  resets on page changes. Fine editing retains native tool history. Local rich
  text input undo remains local to the input; no additional history database or
  synchronization bus is used.
- Font fitting measures the existing Rust layout. Frames remain repair/layout
  geometry even when text overflows them. Font registration retains the complete
  fallback set so the two engines choose the same glyphs. Mask previews and
  repairs continue to use the shared backend mask rules.

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

Any dynamic/static import warning is an application architecture issue and
must be investigated.
