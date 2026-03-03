import { StrictMode, lazy, Suspense, useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { SWRProvider } from "./providers/SWRProvider.tsx";
import { Loader2 } from "lucide-react";
import WelcomeScreen from "./components/WelcomeScreen.tsx";

// Lazy load the main App component for code splitting
const App = lazy(() => import("./App.tsx"));

// Loading fallback component
export const LoadingFallback = () => (
  <div className="flex min-h-screen items-center justify-center bg-background">
    <div className="flex flex-col items-center gap-4">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
      <p className="text-sm text-muted-foreground">Loading...</p>
    </div>
  </div>
);

// Root component to manage tutorial state
function Root() {
  const [startTutorial, setStartTutorial] = useState(false);

  useEffect(() => {
    // Listen for tutorial start event
    const handleTutorialStart = () => setStartTutorial(true);
    window.addEventListener('startTutorial', handleTutorialStart);
    return () => window.removeEventListener('startTutorial', handleTutorialStart);
  }, []);

  const handleAnalyzeStart = () => {
    // Mark that user chose to analyze (skip tutorial)
    localStorage.setItem('skippedTutorial', 'true');
  };

  return (
    <WelcomeScreen 
      onTutorialStart={() => setStartTutorial(true)}
      onAnalyzeStart={handleAnalyzeStart}
    >
      <Suspense fallback={<LoadingFallback />}>
        <App 
          startTutorial={startTutorial} 
          onTutorialEnd={() => setStartTutorial(false)}
        />
      </Suspense>
    </WelcomeScreen>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <SWRProvider>
      <Root />
    </SWRProvider>
  </StrictMode>
);
