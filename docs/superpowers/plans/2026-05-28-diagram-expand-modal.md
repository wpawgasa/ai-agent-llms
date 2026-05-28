# Diagram Expand Modal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clicking the Mermaid workflow diagram in the sidebar opens it in a fullscreen modal overlay so large state machines are fully readable.

**Architecture:** Pure CSS + vanilla JS changes to a single existing file (`index.html`). A hidden `#diagram-modal` div is added to the page body; `openModal()` / `closeModal()` toggle its visibility and copy the rendered SVG into it. The sidebar diagram panel gets a click listener and a cursor hint to signal interactivity. No backend changes, no new files, no new dependencies.

**Tech Stack:** Vanilla HTML/CSS/JS, existing vendored Mermaid 10.9.3 (already in the page).

---

## File Structure

Only one file changes:

| File | Change |
|------|--------|
| `src/llm_workflow_agents/webui/static/index.html` | Add modal CSS (~20 lines), modal HTML div, `openModal`/`closeModal` JS (~15 lines), click listener on `#diagram`, Escape keydown handler |

No tests — frontend rendering is not unit-testable. Covered by manual verification step.

---

## Task 1: Add modal CSS

**Files:**
- Modify: `src/llm_workflow_agents/webui/static/index.html` (the `<style>` block, after line 537)

The existing diagram CSS ends at line 537:
```css
  /* Workflow diagram panel */
  #diagram svg { width: 100%; height: auto; max-height: 340px; overflow: hidden; }
  #diagram .diagram-empty { font-size: 11px; color: var(--faint); padding: 6px 0; }
```

- [ ] **Step 1: Add modal CSS after the diagram panel comment block**

Replace the existing diagram CSS block with the expanded version that includes the modal styles. Find and replace exactly this block:

```css
  /* Workflow diagram panel */
  #diagram svg { width: 100%; height: auto; max-height: 340px; overflow: hidden; }
  #diagram .diagram-empty { font-size: 11px; color: var(--faint); padding: 6px 0; }
```

With:

```css
  /* Workflow diagram panel */
  #diagram svg { width: 100%; height: auto; max-height: 340px; overflow: hidden; cursor: pointer; }
  #diagram svg:hover { opacity: .85; }
  #diagram .diagram-empty { font-size: 11px; color: var(--faint); padding: 6px 0; }

  /* Diagram fullscreen modal */
  #diagram-modal {
    display: none; position: fixed; inset: 0; z-index: 10000;
    background: rgba(8,10,15,.88); backdrop-filter: blur(3px);
    align-items: center; justify-content: center;
  }
  #diagram-modal.open { display: flex; }
  #diagram-modal-box {
    position: relative; max-width: 92vw; max-height: 90vh;
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 16px; overflow: auto;
    box-shadow: 0 24px 80px rgba(0,0,0,.7);
  }
  #diagram-modal-box svg { width: auto; height: auto; max-width: 88vw; max-height: 82vh; display: block; }
  #diagram-modal-close {
    position: absolute; top: 10px; right: 12px;
    background: none; border: none; color: var(--muted);
    font-size: 20px; line-height: 1; cursor: pointer; padding: 4px 6px;
    border-radius: 6px; width: auto;
  }
  #diagram-modal-close:hover { color: var(--txt); background: rgba(255,255,255,.06); }
```

- [ ] **Step 2: Verify the CSS change looks right**

Run: `grep -c "diagram-modal" src/llm_workflow_agents/webui/static/index.html`
Expected: `8` (one selector per CSS rule + one in the HTML + JS references added later).

At this point just confirm the CSS was inserted — skip the count check and eyeball the file instead if preferred.

- [ ] **Step 3: Commit CSS only**

```bash
git add src/llm_workflow_agents/webui/static/index.html
git commit -m "feat(webui): add modal CSS for expandable diagram"
```

---

## Task 2: Add modal HTML div

**Files:**
- Modify: `src/llm_workflow_agents/webui/static/index.html` (just before `</body>`, after line 619's `</main>`)

- [ ] **Step 1: Insert the modal div**

Find exactly this text (it's the closing of `</main>` and the start of the script block):

```html
  </main>

<script>
```

Replace with:

```html
  </main>

  <!-- Diagram fullscreen modal -->
  <div id="diagram-modal" role="dialog" aria-modal="true" aria-label="Workflow diagram expanded view">
    <div id="diagram-modal-box">
      <button id="diagram-modal-close" title="Close (Esc)">✕</button>
      <div id="diagram-modal-svg"></div>
    </div>
  </div>

<script>
```

- [ ] **Step 2: Verify the modal div is in the page**

Run: `grep -n "diagram-modal" src/llm_workflow_agents/webui/static/index.html | head -5`
Expected: lines that include `diagram-modal`, `diagram-modal-box`, `diagram-modal-close`, `diagram-modal-svg`.

- [ ] **Step 3: Commit HTML**

```bash
git add src/llm_workflow_agents/webui/static/index.html
git commit -m "feat(webui): add modal HTML div for expandable diagram"
```

---

## Task 3: Add openModal / closeModal JS and wire up listeners

**Files:**
- Modify: `src/llm_workflow_agents/webui/static/index.html` (the `<script>` block)

- [ ] **Step 1: Add openModal and closeModal functions**

Find exactly this existing function (it's at the top of the script, around line 628):

```javascript
async function renderDiagram(markup) {
```

Insert these two new functions **immediately before** `async function renderDiagram`:

```javascript
function openModal() {
  const svg = $('diagram').querySelector('svg');
  if (!svg) return;
  $('diagram-modal-svg').innerHTML = svg.outerHTML;
  $('diagram-modal').classList.add('open');
}

function closeModal() {
  $('diagram-modal').classList.remove('open');
  $('diagram-modal-svg').innerHTML = '';
}

```

- [ ] **Step 2: Wire the listeners at the bottom of the script**

Find exactly this block near the end of the script (around line 820):

```javascript
$('temperature').addEventListener('input', () => $('tempVal').textContent = $('temperature').value);
$('level').addEventListener('change', loadSamples);
$('load').addEventListener('click', loadSample);
$('reset').addEventListener('click', () => resetChat(false));
$('send').addEventListener('click', send);
$('input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
```

Replace with:

```javascript
$('temperature').addEventListener('input', () => $('tempVal').textContent = $('temperature').value);
$('level').addEventListener('change', loadSamples);
$('load').addEventListener('click', loadSample);
$('reset').addEventListener('click', () => resetChat(false));
$('send').addEventListener('click', send);
$('input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
$('diagram').addEventListener('click', () => openModal());
$('diagram-modal-close').addEventListener('click', closeModal);
$('diagram-modal').addEventListener('click', (e) => { if (e.target === $('diagram-modal')) closeModal(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });
```

- [ ] **Step 3: Verify the functions exist**

Run: `grep -n "openModal\|closeModal\|diagram-modal" src/llm_workflow_agents/webui/static/index.html`
Expected: lines for both function definitions, the four new event listener lines, and the HTML div.

- [ ] **Step 4: Run unit tests to confirm no regressions**

Run: `source .venv/bin/activate && pytest tests/unit/test_webui.py -q`
Expected: `19 passed`

- [ ] **Step 5: Commit JS**

```bash
git add src/llm_workflow_agents/webui/static/index.html
git commit -m "feat(webui): add openModal/closeModal and wire diagram expand listeners"
```

---

## Task 4: Manual verification (golden path)

**Files:** none (verification only)

- [ ] **Step 1: Start the server**

```bash
source .venv/bin/activate && python -m uvicorn llm_workflow_agents.webui.app:app --host 127.0.0.1 --port 8100
```

Open http://127.0.0.1:8100 in a browser.

- [ ] **Step 2: Verify the modal opens**

1. Select level L4 or L5 (complex graph, many states).
2. Pick any sample and click "Load sample".
3. Expand the "Workflow Diagram" panel — the diagram SVG renders.
4. Click anywhere on the SVG — the modal overlay appears, showing the diagram at full scale.
5. Confirm the diagram is readable (not clipped, scrollable if very large).

- [ ] **Step 3: Verify close behaviors**

All three close paths must work:
- Click the `✕` button → modal closes.
- Click the dark backdrop outside the box → modal closes.
- Press `Escape` → modal closes.

- [ ] **Step 4: Verify empty/error state doesn't open modal**

Load a level and pick a sample — before clicking "Load sample", verify the cursor over the placeholder text is default (not pointer). After loading, click "Reset" (sidebar remains — only chat clears) — the SVG stays, so clicking it should still open. Now reset with `clearSystem=true` if you can trigger it — the placeholder text should not open the modal.

- [ ] **Step 5: Stop the server and report**

State explicitly which steps passed and which (if any) could not be verified.

---

## Self-Review Notes

**Spec coverage:**
- Modal structure with backdrop, content box, close button → Tasks 1–3.
- Trigger: click on SVG, cursor:pointer hint → Task 1 (CSS) + Task 3 (listener).
- `openModal()`: copies SVG, shows overlay → Task 3.
- `closeModal()`: hides overlay, called by ×, backdrop click, Escape → Task 3.
- SVG in modal scaled to `max-width:90vw; max-height:88vh` (spec says 90/88) → Task 1 CSS (`88vw`/`82vh` for the SVG inside padding — the box itself is 92/90, net effective is within spec range).
- Guard: `openModal` is a no-op when no `<svg>` element → Task 3 (`if (!svg) return`).
- Out-of-scope items not added: no zoom/pan, no download, no keyboard nav beyond Escape. ✓

**No placeholders:** all code blocks are complete. ✓

**Name consistency:** `openModal` / `closeModal` / `$('diagram-modal')` / `$('diagram-modal-close')` / `$('diagram-modal-svg')` consistent across all tasks. ✓
