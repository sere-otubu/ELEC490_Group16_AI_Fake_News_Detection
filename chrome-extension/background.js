// =============================================================================
// Background Service Worker — Context Menu + API Proxy
// Routes API calls from content scripts to avoid CORS issues
// =============================================================================



// Create right-click context menu on install
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "medcheck-analyze",
        title: "Verify with EvidenceMD",
        contexts: ["selection"]
    });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "medcheck-analyze" && info.selectionText) {
        chrome.tabs.sendMessage(tab.id, {
            action: "analyze",
            text: info.selectionText.trim()
        });
    }
});

// =============================================================================
// Handle API requests from content scripts
// Content scripts can't make cross-origin requests due to CORS,
// so they send messages here and we make the fetch from the service worker.
// =============================================================================

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "apiRequest") {
        handleApiRequest(message)
            .then(sendResponse)
            .catch((err) => sendResponse({ error: err.message }));
        return true; // Keep the message channel open for async response
    }
});

async function handleApiRequest({ url, method, body }) {
    try {
        const response = await fetch(url, {
            method: method || "GET",
            headers: { "Content-Type": "application/json" },
            body: body ? JSON.stringify(body) : undefined,
        });

        if (!response.ok) {
            throw new Error(`API returned ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        return { success: true, data };
    } catch (error) {
        return { error: error.message };
    }
}
