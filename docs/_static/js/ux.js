(() => {
  "use strict";

  function installReadingProgress() {
    const article = document.querySelector(".bd-article");
    if (!article || article.scrollHeight < window.innerHeight * 1.25) return;
    const progress = document.createElement("div");
    progress.className = "pymixef-reading-progress";
    progress.setAttribute("role", "progressbar");
    progress.setAttribute("aria-label", "Reading progress");
    progress.setAttribute("aria-valuemin", "0");
    progress.setAttribute("aria-valuemax", "100");
    document.body.append(progress);

    let scheduled = false;
    const update = () => {
      const rect = article.getBoundingClientRect();
      const available = Math.max(1, article.offsetHeight - window.innerHeight);
      const ratio = Math.min(1, Math.max(0, -rect.top / available));
      const percent = Math.round(ratio * 100);
      progress.style.width = `${percent}%`;
      progress.setAttribute("aria-valuenow", String(percent));
      scheduled = false;
    };
    const requestUpdate = () => {
      if (!scheduled) {
        scheduled = true;
        window.requestAnimationFrame(update);
      }
    };
    update();
    window.addEventListener("scroll", requestUpdate, { passive: true });
    window.addEventListener("resize", requestUpdate, { passive: true });
  }

  function installSearchShortcut() {
    document.addEventListener("keydown", (event) => {
      const tag = document.activeElement?.tagName?.toLowerCase();
      if (event.key !== "/" || ["input", "textarea", "select"].includes(tag)) return;
      const trigger =
        document.querySelector(".search-button-field") ||
        document.querySelector("button.search-button");
      if (trigger) {
        event.preventDefault();
        trigger.click();
      }
    });
  }

  function resolveDocumentationLinks() {
    const root = document.documentElement.dataset.content_root || "./";
    document.querySelectorAll("[data-doc-path]").forEach((link) => {
      const path = link.getAttribute("data-doc-path");
      if (path) link.setAttribute("href", new URL(`${root}${path}`, window.location.href).href);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      resolveDocumentationLinks();
      installReadingProgress();
      installSearchShortcut();
    });
  } else {
    resolveDocumentationLinks();
    installReadingProgress();
    installSearchShortcut();
  }
})();
