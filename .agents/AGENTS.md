# AGENTS.md — Gemma 4 Fine-tuning Encyclopaedia Core Conventions & Rulebook

This document serves as the single source of truth for design decisions, codebase architecture, and implementation gotchas for any AI agent working on this repository.

---

## 1. System Architecture & Project Conventions

The documentation website is a **Single-Page Application (SPA)** optimized for fast load-times, offline-friendly access, and premium aesthetics.

### Codebase Layout
* **Source Shell Template**: [knowledgebase/src/shell_template.html](file:///Users/ksprashanth/code/sandbox/gemma4-finetuning/knowledgebase/src/shell_template.html) — Contains the CSS layout, dark mode toggles, fuzzy search engine, and overall layout shell.
* **Modular Chapters**: Located under [knowledgebase/src/chapters/](file:///Users/ksprashanth/code/sandbox/gemma4-finetuning/knowledgebase/src/chapters/) (e.g., `introduction.html`, `chapter1.html`, ... `chapter9.html`).
* **Compiler Script**: [scripts/compile_docs.py](file:///Users/ksprashanth/code/sandbox/gemma4-finetuning/scripts/compile_docs.py) — A Python compiler that aggregates all modular HTML chapters, injects them into the shell template's `<!-- {{CHAPTERS_CONTENT}} -->` token, and outputs the final production-ready page to `knowledgebase/index.html`.

---

## 2. Design Decisions & Product Requirements

* **Branding Guidelines**: The official application name is exactly **Gemma 4 Fine-tuning Encyclopaedia** (note the uppercase **E**). Do not use "manual" or "wiki" in branding.
* **Split Sidebar Categories**: The navigation menu is divided into two distinct logical categories to separate theory from implementation:
  1. **Chapters**: Conceptual chapters (Cover & Introduction through Chapter 7).
  2. **Project Artifacts**: Practical execution guides and code companion files (Chapters 8 and 9).
* **Rich Typography & Math**: Uses Google Fonts (`Inter` and `JetBrains Mono`) and loads KaTeX dynamically for rendering LaTeX equations (e.g. `\(\Delta W = A \times B\)`).
* **Zero Placeholders**: All tables, parameters, and python examples must be completely filled out with realistic, technical parameters (e.g., memory dimensions, epochs, compute dtypes) to maintain academic and professional rigor.

---

## 3. Implementation Gotchas & Guardrails

> [!WARNING]
> **Premature Container Closures (Tag Symmetry)**
> Ensure all HTML tags are perfectly balanced. An extra `</div>` tag at the end of a chapter will prematurely close the `<main>` container block. This causes the browser's HTML parser to eject subsequent chapters under `<body>`, rendering them completely blank/empty due to layout rules.

> [!IMPORTANT]
> **Sidebar & Mobile Select Synchronisation**
> When adding, removing, or reordering chapters, you must keep **both** navigation inputs in sync:
> 1. The desktop sidebar list: `<ul class="nav-menu" id="nav-list-kb">` & `<ul class="nav-menu" id="nav-list-artifacts">`.
> 2. The mobile dropdown menu: `<select id="mobile-chapter-select">` (uses grouped option elements `<optgroup>`).

> [!TIP]
> **Table Column Auto-Fit**
> All comparative tables employ `th:first-child, td:first-child { white-space: nowrap; }`. This prevents short codes, package names, or parameters (e.g. `bitsandbytes`) from wrapping awkwardly across multiple lines on small viewports.

---

## 4. Git Operations & Environment Guardrails

* **Remote Synchronization**: Git pushing changes to the remote origin requires network access. Ensure `BypassSandbox` is set to `true` when executing `git push origin main`.
* **Clean Compilation**: Always run `python scripts/compile_docs.py` after editing any chapters or shell templates before committing code to ensure the production `index.html` is in sync.
