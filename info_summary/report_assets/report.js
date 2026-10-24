(() => {
  "use strict";
  const input = document.getElementById("search-input");
  const counter = document.getElementById("search-count");
  const prev = document.getElementById("search-prev");
  const next = document.getElementById("search-next");
  let hits = [], current = -1, pending;

  function reveal(element) {
    for (let parent = element.parentElement; parent; parent = parent.parentElement) {
      if (parent.tagName === "DETAILS") parent.open = true;
    }
  }
  function selectHit(index) {
    if (!hits.length) return;
    if (current >= 0) hits[current].classList.remove("current-hit");
    current = (index + hits.length) % hits.length;
    const hit = hits[current];
    hit.classList.add("current-hit");
    reveal(hit);
    hit.scrollIntoView({block: "center", behavior: "smooth"});
    counter.textContent = `${current + 1} / ${hits.length}`;
  }
  function makeRange(block, start, end) {
    const walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT);
    const range = document.createRange();
    let node, offset = 0, started = false;
    while ((node = walker.nextNode())) {
      const length = node.textContent.length;
      if (!started && start < offset + length) {
        range.setStart(node, start - offset);
        started = true;
      }
      if (started && end <= offset + length) {
        range.setEnd(node, end - offset);
        return range;
      }
      offset += length;
    }
    return null;
  }
  function search() {
    clearTimeout(pending);
    pending = null;
    document.querySelectorAll("mark[data-hit]").forEach(mark => {
      const parent = mark.parentNode;
      mark.replaceWith(...mark.childNodes);
      parent.normalize();
    });
    hits = []; current = -1;
    const query = input.value.trim();
    if (query) {
      const literal = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const pattern = new RegExp(literal, "giu");
      document.querySelectorAll("[data-search]").forEach(block => {
        const matches = Array.from(block.textContent.matchAll(pattern));
        for (let i = matches.length - 1; i >= 0; i--) {
          const match = matches[i];
          const range = makeRange(block, match.index, match.index + match[0].length);
          if (!range) continue;
          const mark = document.createElement("mark");
          mark.dataset.hit = "";
          mark.append(range.extractContents());
          range.insertNode(mark);
        }
      });
      hits = Array.from(document.querySelectorAll("mark[data-hit]"));
    }
    prev.disabled = next.disabled = !hits.length;
    counter.textContent = query ? (hits.length ? `0 / ${hits.length}` : "没有匹配") : "输入关键词";
    if (hits.length) selectHit(0);
  }
  input.addEventListener("input", () => { clearTimeout(pending); pending = setTimeout(search, 150); });
  input.addEventListener("keydown", event => {
    if (event.key === "Enter") {
      event.preventDefault();
      if (pending) { search(); pending = null; }
      else selectHit(current + (event.shiftKey ? -1 : 1));
    }
    if (event.key === "Escape") { input.value = ""; search(); }
  });
  prev.addEventListener("click", () => selectHit(current - 1));
  next.addEventListener("click", () => selectHit(current + 1));
  document.addEventListener("keydown", event => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "f") {
      event.preventDefault(); input.focus(); input.select();
    }
  });
  document.querySelectorAll('.contents a, .brand').forEach(link => link.addEventListener("click", () => {
    const target = document.getElementById(link.hash.slice(1));
    if (target) {
      if (target.tagName === "DETAILS") target.open = true;
      reveal(target);
    }
  }));

  const viewer = document.getElementById("image-viewer");
  const viewerImage = document.getElementById("viewer-image");
  let imageButton;
  document.querySelectorAll(".image-expand").forEach(button => button.addEventListener("click", () => {
    imageButton = button;
    const image = button.querySelector("img");
    viewerImage.src = image.src; viewerImage.alt = image.alt;
    document.getElementById("viewer-caption").textContent = image.alt;
    viewer.showModal();
  }));
  document.getElementById("close-viewer").addEventListener("click", () => viewer.close());
  viewer.addEventListener("click", event => { if (event.target === viewer) viewer.close(); });
  viewer.addEventListener("close", () => { viewerImage.removeAttribute("src"); if (imageButton) imageButton.focus(); });

  let printState = [];
  document.getElementById("print-report").addEventListener("click", () => window.print());
  window.addEventListener("beforeprint", () => {
    if (printState.length) return;
    printState = Array.from(document.querySelectorAll(".report-section"), section => [section, section.open]);
    printState.forEach(([section]) => { section.open = true; });
  });
  window.addEventListener("afterprint", () => { printState.forEach(([section, open]) => { section.open = open; }); printState = []; });
})();
