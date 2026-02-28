import { StrictMode, lazy, Suspense } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { SWRProvider } from "./providers/SWRProvider.tsx";
import { Loader2 } from "lucide-react";

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

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <SWRProvider>
      <Suspense fallback={<LoadingFallback />}>
        <App />
      </Suspense>
    </SWRProvider>
  </StrictMode>
);
