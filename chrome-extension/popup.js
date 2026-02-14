// =============================================================================
// Popup Script — Settings & Connection Test
// =============================================================================

const DEFAULT_API_URL = "https://capstone-backend-5xbw.onrender.com";

const apiUrlInput = document.getElementById("api-url");
const saveBtn = document.getElementById("save-btn");
const testBtn = document.getElementById("test-btn");
const statusIndicator = document.getElementById("status-indicator");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

// Load saved URL on popup open
chrome.storage.local.get(["apiUrl"], (result) => {
    apiUrlInput.value = result.apiUrl || DEFAULT_API_URL;
});

// Save URL
saveBtn.addEventListener("click", () => {
    const url = apiUrlInput.value.trim().replace(/\/+$/, ""); // Remove trailing slashes
    if (!url) {
        showStatus("error", "Please enter a valid URL");
        return;
    }
    chrome.storage.local.set({ apiUrl: url }, () => {
        showStatus("success", "Settings saved!");
        setTimeout(() => hideStatus(), 2000);
    });
});

// Test connection
testBtn.addEventListener("click", async () => {
    const url = apiUrlInput.value.trim().replace(/\/+$/, "");
    if (!url) {
        showStatus("error", "Please enter a URL first");
        return;
    }

    showStatus("loading", "Testing connection...");
    testBtn.disabled = true;

    try {
        const response = await fetch(`${url}/health`, {
            method: "GET",
            signal: AbortSignal.timeout(8000),
        });

        if (response.ok) {
            const data = await response.json();
            showStatus("success", `Connected — ${data.service} ${data.version}`);
        } else {
            showStatus("error", `Server returned ${response.status}`);
        }
    } catch (error) {
        if (error.name === "TimeoutError" || error.name === "AbortError") {
            showStatus("error", "Connection timed out");
        } else {
            showStatus("error", "Cannot connect to server");
        }
    } finally {
        testBtn.disabled = false;
    }
});

// Allow Enter key to save
apiUrlInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") saveBtn.click();
});

function showStatus(type, message) {
    statusIndicator.style.display = "flex";
    statusText.textContent = message;
    statusDot.className = `popup-status-dot popup-status-${type}`;
}

function hideStatus() {
    statusIndicator.style.display = "none";
}
