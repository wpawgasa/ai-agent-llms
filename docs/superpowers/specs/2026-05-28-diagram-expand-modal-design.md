# Diagram Expand Modal — Design

**Date:** 2026-05-28  
**Status:** Approved

## Goal

Clicking the Mermaid workflow-diagram in the sidebar opens it in a fullscreen
modal overlay so large state machines (L4–L5, 20+ states) are readable without
scrolling or squinting.

## Scope

Single-file change: `src/llm_workflow_agents/webui/static/index.html`.
No backend, no new endpoints, no tests (frontend rendering not unit-testable).

## Design

**Modal structure** — a `<div id="diagram-modal">` div added to the page body,
hidden by default (`display:none`). When shown, it covers the full viewport as a
dark semi-transparent backdrop (`position:fixed; inset:0; z-index:1000`) with a
centered content box. The box holds the SVG at natural scale, capped at
`max-width:90vw; max-height:88vh` so it never overflows. A `×` close button sits
in the top-right corner of the box.

**Trigger** — after `renderDiagram()` injects the SVG into `#diagram`, the SVG
gets `cursor:pointer` and a `title="Click to expand"` hint. Clicking it calls
`openModal()`.

**`openModal()`** — copies the current SVG innerHTML from `#diagram` into the
modal content box and sets `display:flex` on the overlay.

**`closeModal()`** — sets `display:none` on the overlay. Called by: the `×`
button click, a click on the backdrop itself (outside the content box), and
`Escape` keydown on `document`.

**SVG in modal** — `width:auto; height:auto; max-width:90vw; max-height:88vh`
so it scales to fit the viewport. The sidebar copy remains unchanged.

**Error / empty state** — diagram panel shows text ("no graph", error message)
rather than SVG; clicking it does nothing. Guard: `openModal` is a no-op when
`#diagram` contains no `<svg>` element.

## Out of scope

- Zoom/pan inside the modal
- Download/export of the SVG
- Keyboard navigation within the modal beyond Escape-to-close
