# MedCheck AI — Chrome Extension

A Chrome extension that lets you fact-check medical claims directly on any webpage. Highlight text, right-click, and get an AI-powered verdict with source citations — without leaving the page.

---

## Installation

1. Open **`chrome://extensions/`** in Google Chrome
2. Enable **Developer mode** (toggle in the top-right corner)
3. Click **Load unpacked**
4. Select the `chrome-extension/` folder from this repository
5. The MedCheck AI icon will appear in your browser toolbar

---

## Usage

### Method 1: Context Menu (Right-Click)

1. **Highlight** any medical claim on a webpage
2. **Right-click** the selected text
3. Click **"Check with MedCheck AI"** from the context menu
4. A results panel will slide in from the right side of the page

### Method 2: Extension Popup

1. Click the **MedCheck AI icon** in the toolbar
2. Paste or type a medical claim into the text box
3. Click **Analyze**
4. Results appear in the popup window

### Understanding Results

Each response includes:

- **Verdict** — `TRUE`, `FALSE`, or `UNVERIFIED`
- **Explanation** — A detailed, evidence-based rationale
- **Source Documents** — The exact papers/articles cited, with similarity scores
- **Links** — Direct links to the original source material (PubMed, WHO, arXiv, etc.)

---

## Configuration

The extension communicates with the MedCheck AI backend API. The server URL is configured in `content.js`:

```javascript
const API_BASE_URL = "https://your-backend-url.onrender.com";
```

For local development, change this to `http://localhost:8000`.

---

## File Structure

```
chrome-extension/
├── manifest.json       # Extension configuration (Manifest V3)
├── background.js       # Service worker — registers context menu
├── content.js          # Injected into pages — handles text selection & API calls
├── content.css         # Styles for the inline results panel
├── popup.html          # Extension popup UI
├── popup.js            # Popup logic
├── popup.css           # Popup styles
└── icons/              # Extension icons (16, 48, 128px)
```

---

## Requirements

- Google Chrome (or any Chromium-based browser)
- A running MedCheck AI backend (local or deployed)
