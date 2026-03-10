import { useState, memo } from "react";
import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  AlertCircle,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

interface VerdictItem {
  name: string;
  description: string;
  bgColor: string;
  textColor: string;
  icon: React.ReactNode;
}

const VerdictLegend = memo(() => {
  const [isExpanded, setIsExpanded] = useState(true);

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
    <div className="rounded-xl border border-border/40 bg-card/50 backdrop-blur-sm transition-all duration-300 hover:border-border/60">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-muted/20 transition-colors rounded-t-xl"
      >
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/15">
            <AlertCircle className="h-4 w-4 text-primary" />
          </div>
          <div className="text-left">
            <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/80">
              Understanding Verdicts
            </p>
            <h3 className="font-display text-sm font-bold tracking-tight">
              Possible Verdict Types
            </h3>
          </div>
        </div>
        <div className="text-muted-foreground transition-transform duration-300">
          {isExpanded ? (
            <ChevronUp className="h-5 w-5" />
          ) : (
            <ChevronDown className="h-5 w-5" />
          )}
        </div>
      </button>

      {isExpanded && (
        <div className="border-t border-border/40 px-6 py-4">
          <div className="grid gap-3">
            {verdicts.map((verdict) => (
              <div
                key={verdict.name}
                className={`flex items-start gap-3 rounded-lg ${verdict.bgColor} border border-border/40 p-3 transition-all hover:border-border/60 hover:shadow-md`}
              >
                <div className={`mt-0.5 flex-shrink-0 ${verdict.textColor}`}>
                  {verdict.icon}
                </div>
                <div className="min-w-0 flex-1">
                  <p className={`text-sm font-semibold ${verdict.textColor}`}>
                    {verdict.name}
                  </p>
                  <p className="text-[12px] text-muted-foreground/80 leading-relaxed mt-1">
                    {verdict.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-muted-foreground/60 mt-4 text-center">
            All verdicts are based on peer-reviewed medical literature and evidence databases.
          </p>
        </div>
      )}
    </div>
  );
});

VerdictLegend.displayName = "VerdictLegend";

export default VerdictLegend;
