// =============================================================================
// Content Script — Injected into all webpages
// Detects text selection, shows analyze button, calls RAG API, shows results
// =============================================================================

(function () {
  "use strict";

  const API_URL = "https://capstone-backend-5xbw.onrender.com";
  let currentButton = null;
  let currentOverlay = null;


  // Listen for context menu messages from background.js
  chrome.runtime.onMessage.addListener((message) => {
    if (message.action === "analyze" && message.text) {
      removeExistingUI();
      analyzeText(message.text);
    }
  });

  // =========================================================================
  // Text Selection Detection
  // =========================================================================

  document.addEventListener("mouseup", (e) => {
    // Ignore clicks on our own UI elements
    if (e.target.closest(".fnd-container")) return;

    const selectedText = window.getSelection().toString().trim();

    if (selectedText.length > 10) {
      showAnalyzeButton(e.clientX, e.clientY, selectedText);
    }
  });

  // Remove button when clicking elsewhere (but not on our UI)
  document.addEventListener("mousedown", (e) => {
    if (!e.target.closest(".fnd-container")) {
      removeButton();
    }
  });

  // =========================================================================
  // Floating Analyze Button
  // =========================================================================

  function showAnalyzeButton(x, y, text) {
    removeButton();

    const btn = document.createElement("div");
    btn.className = "fnd-container fnd-analyze-btn";
    btn.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"/>
        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg>
      <span>Check Claim</span>
    `;

    // Position near cursor but keep within viewport
    const posX = Math.min(x + 10, window.innerWidth - 170);
    const posY = Math.max(y - 45, 10);
    btn.style.left = `${posX}px`;
    btn.style.top = `${posY}px`;

    btn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      removeButton();
      analyzeText(text);
    });

    document.body.appendChild(btn);
    currentButton = btn;

    // Animate in
    requestAnimationFrame(() => btn.classList.add("fnd-visible"));
  }

  function removeButton() {
    if (currentButton) {
      currentButton.remove();
      currentButton = null;
    }
  }

  // =========================================================================
  // API Call
  // =========================================================================

  async function analyzeText(text) {
    showLoadingOverlay(text);

    try {
      // Route through background service worker to avoid CORS issues
      // (content scripts inherit the page's origin, not chrome-extension://)
      const result = await chrome.runtime.sendMessage({
        action: "apiRequest",
        url: `${API_URL}/rag/query`,
        method: "POST",
        body: { query: text, top_k: 3 },
      });

      if (result.error) {
        throw new Error(result.error);
      }

      showResultsOverlay(text, result.data);
    } catch (error) {
      showErrorOverlay(text, error.message);
    }
  }

  // =========================================================================
  // Parse Structured Response
  // =========================================================================

  function parseResponse(chatResponse) {
    const result = {
      reasoning: "",
      verdict: "UNKNOWN",
      confidence: null,
      evidence: "",
      sources: "",
    };

    // Helper to extract content for a given bold marker.
    // Matches **Marker**: <content> and stops at the next \n**<Word> or end.
    function extract(marker) {
      const regex = new RegExp(`\\*\\*${marker}\\*\\*:\\s*([\\s\\S]*?)(?=\\n\\*\\*[A-Za-z]|$)`, "i");
      const match = chatResponse.match(regex);
      return match ? match[1].trim() : "";
    }

    // Extract fields
    const rawVerdict = extract("Verdict");
    const rawReasoning = extract("Reasoning");
    const rawConfidence = extract("Confidence\\s*Score");
    const rawEvidence = extract("Evidence");
    const rawSources = extract("Source\\s*Files");

    // Clean up verdict (remove brackets if present)
    if (rawVerdict) {
      result.verdict = rawVerdict.replace(/[\[\]]/g, "").toUpperCase();
    }

    // Parse reasoning
    result.reasoning = rawReasoning;

    // Parse numeric confidence
    if (rawConfidence) {
      const confValue = parseFloat(rawConfidence);
      if (!isNaN(confValue)) result.confidence = confValue;
    }

    // Evidence and sources
    result.evidence = rawEvidence;
    result.sources = rawSources;

    return result;
  }

  // =========================================================================
  // Verdict Styling
  // =========================================================================

  function getVerdictConfig(verdict) {
    const configs = {
      ACCURATE: { color: "#10b981", bg: "rgba(16,185,129,0.15)", icon: "✅", label: "Accurate" },
      INACCURATE: { color: "#ef4444", bg: "rgba(239,68,68,0.15)", icon: "❌", label: "Inaccurate" },
      "PARTIALLY ACCURATE": { color: "#f59e0b", bg: "rgba(245,158,11,0.15)", icon: "⚠️", label: "Partially Accurate" },
      MISLEADING: { color: "#f97316", bg: "rgba(249,115,22,0.15)", icon: "⚡", label: "Misleading" },
      UNVERIFIABLE: { color: "#8b5cf6", bg: "rgba(139,92,246,0.15)", icon: "❓", label: "Unverifiable" },
      OUTDATED: { color: "#6366f1", bg: "rgba(99,102,241,0.15)", icon: "📅", label: "Outdated" },
      OPINION: { color: "#06b6d4", bg: "rgba(6,182,212,0.15)", icon: "💬", label: "Opinion" },
      INCONCLUSIVE: { color: "#a855f7", bg: "rgba(168,85,247,0.15)", icon: "🔄", label: "Inconclusive" },
      IRRELEVANT: { color: "#6b7280", bg: "rgba(107,114,128,0.15)", icon: "🚫", label: "Irrelevant" },
    };
    return configs[verdict] || { color: "#9ca3af", bg: "rgba(156,163,175,0.15)", icon: "❔", label: verdict };
  }

  // =========================================================================
  // Loading Overlay
  // =========================================================================

  function showLoadingOverlay(queryText) {
    removeOverlay();

    const overlay = createOverlayShell(queryText);
    const body = overlay.querySelector(".fnd-overlay-body");

    body.innerHTML = `
      <div class="fnd-loading">
        <div class="fnd-spinner"></div>
        <p class="fnd-loading-text">Analyzing claim...</p>
        <p class="fnd-loading-subtext">Searching medical knowledge base</p>
      </div>
    `;

    document.body.appendChild(overlay);
    currentOverlay = overlay;
    requestAnimationFrame(() => overlay.classList.add("fnd-visible"));
  }

  // =========================================================================
  // Results Overlay
  // =========================================================================

  function showResultsOverlay(queryText, data) {
    removeOverlay();

    const parsed = parseResponse(data.chat_response);
    const verdict = getVerdictConfig(parsed.verdict);

    const overlay = createOverlayShell(queryText);
    const body = overlay.querySelector(".fnd-overlay-body");

    // Build confidence bar HTML
    let confidenceHtml = "";
    if (parsed.confidence !== null) {
      const pct = Math.round(parsed.confidence * 100);
      confidenceHtml = `
        <div class="fnd-section">
          <div class="fnd-section-label">Confidence</div>
          <div class="fnd-confidence-bar-track">
            <div class="fnd-confidence-bar-fill" style="width:${pct}%; background:${verdict.color}"></div>
          </div>
          <div class="fnd-confidence-value">${pct}%</div>
        </div>
      `;
    }

    // Build evidence HTML
    let evidenceHtml = "";
    if (parsed.evidence && parsed.evidence !== "N/A") {
      evidenceHtml = `
        <div class="fnd-section">
          <div class="fnd-section-label">Evidence</div>
          <div class="fnd-evidence">${escapeHtml(parsed.evidence)}</div>
        </div>
      `;
    }

    // Build sources HTML
    let sourcesHtml = "";
    if (parsed.sources) {
      sourcesHtml = `
        <div class="fnd-section">
          <div class="fnd-section-label">Sources</div>
          <div class="fnd-sources">${escapeHtml(parsed.sources)}</div>
        </div>
      `;
    }

    // Build source documents HTML
    let docsHtml = "";
    if (data.source_documents && data.source_documents.length > 0) {
      const docItems = data.source_documents
        .map((doc) => {
          const score = Math.round(doc.score * 100);
          return `
            <div class="fnd-doc-item">
              <div class="fnd-doc-header">
                <span class="fnd-doc-name">${escapeHtml(doc.metadata.file_name)}</span>
                <span class="fnd-doc-score">${score}% match</span>
              </div>
              <div class="fnd-doc-content">${escapeHtml(doc.content.substring(0, 200))}${doc.content.length > 200 ? "..." : ""}</div>
            </div>
          `;
        })
        .join("");

      docsHtml = `
        <details class="fnd-section fnd-docs-toggle">
          <summary class="fnd-section-label fnd-clickable">Source Documents (${data.source_documents.length})</summary>
          <div class="fnd-docs-list">${docItems}</div>
        </details>
      `;
    }

    body.innerHTML = `
      <div class="fnd-verdict-badge" style="background:${verdict.bg}; color:${verdict.color}; border-color:${verdict.color}">
        <span class="fnd-verdict-icon">${verdict.icon}</span>
        <span class="fnd-verdict-text">${verdict.label}</span>
      </div>

      <div class="fnd-section">
        <div class="fnd-section-label">Analysis</div>
        <div class="fnd-reasoning">${escapeHtml(parsed.reasoning)}</div>
      </div>

      ${confidenceHtml}
      ${evidenceHtml}
      ${sourcesHtml}
      ${docsHtml}
    `;

    document.body.appendChild(overlay);
    currentOverlay = overlay;
    requestAnimationFrame(() => overlay.classList.add("fnd-visible"));
  }

  // =========================================================================
  // Error Overlay
  // =========================================================================

  function showErrorOverlay(queryText, errorMessage) {
    removeOverlay();

    const overlay = createOverlayShell(queryText);
    const body = overlay.querySelector(".fnd-overlay-body");

    body.innerHTML = `
      <div class="fnd-error">
        <div class="fnd-error-icon">⚠️</div>
        <div class="fnd-error-title">Analysis Failed</div>
        <div class="fnd-error-message">${escapeHtml(errorMessage)}</div>
        <div class="fnd-error-hint">Check that your API URL is correct in the extension settings and that the server is running.</div>
      </div>
    `;

    document.body.appendChild(overlay);
    currentOverlay = overlay;
    requestAnimationFrame(() => overlay.classList.add("fnd-visible"));
  }

  // =========================================================================
  // Shared Overlay Shell
  // =========================================================================

  function createOverlayShell(queryText) {
    const overlay = document.createElement("div");
    overlay.className = "fnd-container fnd-overlay";

    overlay.innerHTML = `
      <div class="fnd-overlay-header">
        <div class="fnd-overlay-title">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
          <span>MedCheck AI</span>
        </div>
        <button class="fnd-close-btn" title="Close">&times;</button>
      </div>
      <div class="fnd-claim-preview">
        <span class="fnd-claim-label">Claim:</span>
        <span class="fnd-claim-text">"${escapeHtml(truncate(queryText, 150))}"</span>
      </div>
      <div class="fnd-overlay-body"></div>
    `;

    // Close button handler
    overlay.querySelector(".fnd-close-btn").addEventListener("click", removeOverlay);

    // Allow dragging the overlay
    makeDraggable(overlay, overlay.querySelector(".fnd-overlay-header"));

    return overlay;
  }

  // =========================================================================
  // Utilities
  // =========================================================================

  function removeOverlay() {
    if (currentOverlay) {
      currentOverlay.remove();
      currentOverlay = null;
    }
  }

  function removeExistingUI() {
    removeButton();
    removeOverlay();
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function truncate(text, maxLength) {
    return text.length > maxLength ? text.substring(0, maxLength) + "..." : text;
  }

  function makeDraggable(element, handle) {
    let offsetX = 0, offsetY = 0, isDragging = false;

    handle.style.cursor = "grab";

    handle.addEventListener("mousedown", (e) => {
      if (e.target.closest(".fnd-close-btn")) return;
      isDragging = true;
      handle.style.cursor = "grabbing";
      const rect = element.getBoundingClientRect();
      offsetX = e.clientX - rect.left;
      offsetY = e.clientY - rect.top;
      e.preventDefault();
    });

    document.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      element.style.right = "auto";
      element.style.left = `${e.clientX - offsetX}px`;
      element.style.top = `${e.clientY - offsetY}px`;
    });

    document.addEventListener("mouseup", () => {
      if (isDragging) {
        isDragging = false;
        handle.style.cursor = "grab";
      }
    });
  }
})();
