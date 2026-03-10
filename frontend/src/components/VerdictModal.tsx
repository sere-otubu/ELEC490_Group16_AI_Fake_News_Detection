import { useState, memo } from "react";
import { createPortal } from "react-dom";
import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  AlertCircle,
  Info,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";

interface VerdictItem {
  name: string;
  description: string;
  bgColor: string;
  textColor: string;
  icon: React.ReactNode;
}

const VerdictModal = memo(() => {
  const [isOpen, setIsOpen] = useState(false);

  const verdicts: VerdictItem[] = [
    {
      name: "ACCURATE",
      description: "Claim is well-supported by peer-reviewed evidence",
      bgColor: "bg-emerald-500/15",
      textColor: "text-emerald-400",
      icon: <CheckCircle2 className="h-4 w-4" />,
    },
    {
      name: "PARTIALLY ACCURATE",
      description: "Claim has some accurate elements but contains inaccuracies",
      bgColor: "bg-amber-500/15",
      textColor: "text-amber-400",
      icon: <AlertTriangle className="h-4 w-4" />,
    },
    {
      name: "INACCURATE",
      description: "Claim directly contradicts available evidence",
      bgColor: "bg-red-500/15",
      textColor: "text-red-400",
      icon: <XCircle className="h-4 w-4" />,
    },
    {
      name: "MISLEADING",
      description: "Claim is deceptively framed or lacks important context",
      bgColor: "bg-red-500/15",
      textColor: "text-red-400",
      icon: <XCircle className="h-4 w-4" />,
    },
    {
      name: "UNVERIFIABLE",
      description: "Insufficient peer-reviewed data to verify this claim",
      bgColor: "bg-muted",
      textColor: "text-muted-foreground",
      icon: <AlertCircle className="h-4 w-4" />,
    },
    {
      name: "OUTDATED",
      description: "Information is based on older research or standards",
      bgColor: "bg-muted",
      textColor: "text-muted-foreground",
      icon: <AlertCircle className="h-4 w-4" />,
    },
    {
      name: "OPINION",
      description: "Statement is editorial or subjective in nature",
      bgColor: "bg-muted",
      textColor: "text-muted-foreground",
      icon: <AlertCircle className="h-4 w-4" />,
    },
    {
      name: "INCONCLUSIVE",
      description: "Research on this topic shows mixed or conflicting results",
      bgColor: "bg-muted",
      textColor: "text-muted-foreground",
      icon: <AlertCircle className="h-4 w-4" />,
    },
    {
      name: "IRRELEVANT",
      description: "Claim is off-topic or not related to medical evidence",
      bgColor: "bg-muted",
      textColor: "text-muted-foreground",
      icon: <AlertCircle className="h-4 w-4" />,
    },
  ];

  return (
    <>
      <Button
        variant="ghost"
        size="icon"
        className="h-10 w-10 rounded-xl hover:bg-accent/50"
        title="Verdict Guide"
        onClick={() => setIsOpen(true)}
        data-tutorial="verdict-guide"
      >
        <Info className="h-[18px] w-[18px]" />
      </Button>

      {isOpen &&
        typeof document !== "undefined" &&
        createPortal(
          <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4">
            <div
              className="fixed inset-0 bg-black/60 backdrop-blur-sm"
              onClick={() => setIsOpen(false)}
            />
            <div className="relative z-[10000] w-full max-w-2xl rounded-xl border border-border/70 bg-card shadow-2xl max-h-[90vh] overflow-y-auto">
              <div className="sticky top-0 bg-card border-b border-border/40 px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between gap-4">
                <div className="min-w-0">
                  <h2 className="text-xl sm:text-2xl font-bold tracking-tight">Possible Verdict Results</h2>
                  <p className="text-[12px] sm:text-[13px] text-muted-foreground/80 mt-1">
                    All possible verdicts EvidenceMD can return when analyzing medical claims.
                  </p>
                </div>
                <button
                  onClick={() => setIsOpen(false)}
                  className="rounded-lg hover:bg-muted/50 p-1 transition-colors flex-shrink-0"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              <div className="grid gap-2 sm:gap-3 p-4 sm:p-6">
                {verdicts.map((verdict) => (
                  <div
                    key={verdict.name}
                    className={`flex items-start gap-3 rounded-lg ${verdict.bgColor} border border-border/40 p-3 sm:p-4 transition-all hover:border-border/60 hover:shadow-md`}
                  >
                    <div className={`mt-0.5 flex-shrink-0 ${verdict.textColor}`}>
                      {verdict.icon}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className={`text-sm font-bold ${verdict.textColor}`}>
                        {verdict.name}
                      </p>
                      <p className="text-[12px] sm:text-[13px] text-muted-foreground/80 leading-relaxed mt-1">
                        {verdict.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="border-t border-border/40 px-4 sm:px-6 py-3 sm:py-4 bg-muted/20">
                <p className="text-[11px] sm:text-[12px] text-muted-foreground/60 text-center">
                  All verdicts are based on peer-reviewed medical literature and evidence databases.
                </p>
              </div>
            </div>
          </div>,
          document.body
        )}
    </>
  );
});

VerdictModal.displayName = "VerdictModal";

export default VerdictModal;
