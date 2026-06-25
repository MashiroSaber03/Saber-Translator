# UI Maintenance Decisions

This document describes the final maintenance policy for the UI layer. It is
not a backlog or migration ledger. Daily development should use
`npm run lint:ui`; proactive architecture review can use
`npm run lint:ui:audit`.

## Current Contract

- Component styles live with the owning Vue component in `<style scoped>`.
- `*.global.styles.css` is reserved for explicit Teleport or slot
  reach-through owners such as modal containers and global overlays.
- Ordinary `*.styles.css` script imports are not allowed because they create
  hidden global style ownership.
- Token files are the only global token source. UI source must not reference
  retired short aliases, value-named tokens, generated numbered owner tokens, or
  `--palette-*` internals. Color values belong directly to semantic,
  component, or domain tokens.
- Large cohesive components are allowed when splitting would mainly add
  prop/event plumbing. Line count is a review signal, not a target or a
  forced split rule.
- High-risk visual states are protected by visual regression tests instead of
  comments about past implementations.

## Split Rules

- Split when a child can own a real UI region, modal body, list, form section,
  or independently tested interaction.
- Extract a composable when behavior is reusable or large enough to understand
  separately from the template.
- Keep a component whole when state, template, and pointer/keyboard interaction
  must be understood together.
- Never split CSS by horizontal names such as `base`, `layout`, `panels`, or
  `responsive`.

## Audit Output

`lint:ui:audit` reports current health signals:

- token file count,
- `:root` token count,
- token dependency count,
- critical visual state coverage count,
- heavy owner review signal count,
- owner token density signals.

It must not print large-owner candidates, pending split decisions, or
layout bypass todo lists. If a real violation appears, default `lint:ui` should
fail with the exact file and rule.

## Critical Visual States

The architecture lint checks that these visual contracts remain covered in
`tests/visual/ui-regression.spec.ts`:

| State | Visual Contract |
| --- | --- |
| Translate empty shell | `translate workspace empty state keeps its layout contract` |
| Translate loaded sidebars | `translate loaded workspace keeps fixed sidebar sizing contract` |
| EditWorkspace dark shell | `translate edit workspace keeps dark editor shell contract` |
| EditWorkspace selected bubble editor | `translate edit workspace selected bubble keeps editor panel contract` |
| Insight selected-book sidebars | `insight selected-book sidebars keep their gutter contract` |
| Reader immersive shell | `reader loaded state keeps its layout contract` |
| Character Studio empty shell | `character studio empty workspace keeps its layout contract` |
| Character Studio editor/preview shell | `character studio editor and preview keep split workspace contract` |
