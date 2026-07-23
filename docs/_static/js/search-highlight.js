(() => {
  "use strict";

  const PARAMS = ["highlight", "q"];
  // Keep executable code blocks intact, but allow inline API names in <code>
  // elements to receive the same destination-page highlighting as prose.
  const SKIP = "script, style, noscript, textarea, input, button, select, option, svg, math, pre, mark";
  const MAX_TERMS = 10;
  let matches = [];
  let activeIndex = -1;

  function termsFromLocation() {
    const params = new URLSearchParams(window.location.search);
    const raw = PARAMS.flatMap((name) => params.getAll(name))
      .join(" ")
      .replace(/\+/g, " ")
      .trim();
    if (!raw) return [];

    const parsed = [];
    const pattern = /"([^"]+)"|([^\s,]+)/g;
    let match;
    while ((match = pattern.exec(raw)) !== null && parsed.length < MAX_TERMS) {
      const term = (match[1] || match[2] || "").trim();
      if (term.length > 1 && !parsed.some((item) => item.toLowerCase() === term.toLowerCase())) {
        parsed.push(term);
      }
    }
    return parsed.sort((a, b) => b.length - a.length);
  }

  function escapeRegex(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function articleRoot() {
    return (
      document.querySelector(".bd-article") ||
      document.querySelector("article") ||
      document.querySelector("main")
    );
  }

  function markText(root, terms) {
    if (!root || !terms.length) return;
    const expression = new RegExp(`(${terms.map(escapeRegex).join("|")})`, "giu");
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (!node.nodeValue || !expression.test(node.nodeValue)) {
          expression.lastIndex = 0;
          return NodeFilter.FILTER_REJECT;
        }
        expression.lastIndex = 0;
        const parent = node.parentElement;
        if (!parent || parent.closest(SKIP) || parent.closest(".headerlink, .search-hit-toolbar")) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });

    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);

    textNodes.forEach((node) => {
      const fragment = document.createDocumentFragment();
      let cursor = 0;
      node.nodeValue.replace(expression, (matched, _group, offset) => {
        if (offset > cursor) fragment.append(node.nodeValue.slice(cursor, offset));
        const mark = document.createElement("mark");
        mark.className = "query-highlight";
        mark.dataset.searchTerm = matched;
        mark.tabIndex = -1;
        mark.textContent = matched;
        fragment.append(mark);
        cursor = offset + matched.length;
        return matched;
      });
      if (cursor < node.nodeValue.length) fragment.append(node.nodeValue.slice(cursor));
      node.replaceWith(fragment);
    });
  }

  function setActive(index, shouldScroll = true) {
    if (!matches.length) return;
    matches.forEach((node) => node.classList.remove("is-active"));
    activeIndex = (index + matches.length) % matches.length;
    const current = matches[activeIndex];
    current.classList.add("is-active");
    const counter = document.querySelector("[data-search-hit-counter]");
    if (counter) counter.textContent = `${activeIndex + 1} of ${matches.length}`;
    if (shouldScroll) {
      current.scrollIntoView({ behavior: "smooth", block: "center" });
      current.focus({ preventScroll: true });
    }
  }

  function clearHighlights() {
    document.querySelectorAll("mark.query-highlight").forEach((mark) => {
      mark.replaceWith(document.createTextNode(mark.textContent || ""));
    });
    document.querySelectorAll("span.highlighted").forEach((span) => {
      span.classList.remove("highlighted", "is-active");
    });
    const url = new URL(window.location.href);
    PARAMS.forEach((name) => url.searchParams.delete(name));
    window.history.replaceState({}, "", url);
    document.querySelector(".search-hit-toolbar")?.remove();
    matches = [];
  }

  function buildToolbar(root, terms) {
    const toolbar = document.createElement("aside");
    toolbar.className = "search-hit-toolbar";
    toolbar.setAttribute("aria-label", "On-page search matches");

    const label = document.createElement("span");
    label.className = "search-hit-label";
    label.innerHTML = matches.length
      ? `Highlighted <strong>${terms.map((term) => term.replace(/[<>&]/g, "")).join(", ")}</strong> · <span data-search-hit-counter>1 of ${matches.length}</span>`
      : `No on-page matches for <strong>${terms.map((term) => term.replace(/[<>&]/g, "")).join(", ")}</strong>`;
    label.setAttribute("aria-live", "polite");
    toolbar.append(label);

    if (matches.length) {
      const previous = document.createElement("button");
      previous.type = "button";
      previous.title = "Previous match (Shift+Enter)";
      previous.setAttribute("aria-label", "Previous search match");
      previous.textContent = "↑";
      previous.addEventListener("click", () => setActive(activeIndex - 1));

      const next = document.createElement("button");
      next.type = "button";
      next.title = "Next match (Enter)";
      next.setAttribute("aria-label", "Next search match");
      next.textContent = "↓";
      next.addEventListener("click", () => setActive(activeIndex + 1));
      toolbar.append(previous, next);
    }

    const clear = document.createElement("button");
    clear.type = "button";
    clear.title = "Clear highlighting (Escape)";
    clear.setAttribute("aria-label", "Clear search highlighting");
    clear.textContent = "Clear";
    clear.addEventListener("click", clearHighlights);
    toolbar.append(clear);
    root.prepend(toolbar);
  }

  function decorateSearchResultLinks(terms) {
    const apply = () => {
      document.querySelectorAll("ul.search a, #search-results a").forEach((link) => {
        const target = new URL(link.href, window.location.href);
        if (target.origin !== window.location.origin) return;
        target.searchParams.set("highlight", terms.join(" "));
        link.href = target.href;
      });
    };
    apply();
    const results = document.querySelector("#search-results") || document.querySelector("main");
    if (results) {
      const observer = new MutationObserver(apply);
      observer.observe(results, { childList: true, subtree: true });
      window.setTimeout(() => observer.disconnect(), 5000);
    }
  }

  function initialize() {
    const terms = termsFromLocation();
    const root = articleRoot();
    if (!terms.length) return;
    if (/\/search\/?$/.test(window.location.pathname)) {
      decorateSearchResultLinks(terms);
      return;
    }
    if (!root || root.querySelector(".search-hit-toolbar")) return;

    const existing = Array.from(root.querySelectorAll("span.highlighted"));
    if (!existing.length) markText(root, terms);
    matches = Array.from(root.querySelectorAll("mark.query-highlight, span.highlighted"));
    buildToolbar(root, terms);
    if (matches.length) setActive(0, false);

    document.addEventListener("keydown", (event) => {
      if (!matches.length) return;
      if (event.key === "Enter" && !event.metaKey && !event.ctrlKey && !event.altKey) {
        const tag = document.activeElement?.tagName?.toLowerCase();
        if (!["input", "textarea", "button", "select", "a"].includes(tag)) {
          event.preventDefault();
          setActive(activeIndex + (event.shiftKey ? -1 : 1));
        }
      } else if (event.key === "Escape") {
        clearHighlights();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => window.setTimeout(initialize, 0));
  } else {
    window.setTimeout(initialize, 0);
  }
})();
