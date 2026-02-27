// =============================================================================
// Popup Script — Connection Test
// =============================================================================

const API_URL = "https://capstone-backend-5xbw.onrender.com";

const testBtn = document.getElementById("test-btn");
const statusIndicator = document.getElementById("status-indicator");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

// Test connection
testBtn.addEventListener("click", async () => {
    showStatus("loading", "Testing connection...");
    testBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/health`, {
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

function showStatus(type, message) {
    statusIndicator.style.display = "flex";
    statusText.textContent = message;
    statusDot.className = `popup-status-dot popup-status-${type}`;
}

function hideStatus() {
    statusIndicator.style.display = "none";
}
