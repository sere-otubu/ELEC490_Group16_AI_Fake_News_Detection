import { useState, useEffect, useMemo, useCallback, memo, type ReactNode, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Analytics } from "@vercel/analytics/react";
import { SpeedInsights } from "@vercel/speed-insights/react"
import {
  Plus,
  Bot,
  Send,
  Loader2,
  AlertCircle,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Download,
  FileText,
  Shield,
  CheckCircle2,
  XCircle,
  Stethoscope,
  ExternalLink,
  Link,
  Image,
  Mic,
  MicOff,
  X,
  Globe,
  BookOpen,
} from "lucide-react";
import {
  useQueryRAG,
  convertQueryResponseToMessages,
} from "@/hooks/useApi";
import { apiClient } from "@/lib/api-client";
import type { Message, SourceDocument } from "@/types/api";
import Tutorial from "@/components/Tutorial";
import MedicalCrossLogo from "@/components/MedicalCrossLogo";

// Source Document Card Component
interface SourceDocumentCardProps {
  doc: SourceDocument;
  index: number;
}

const SourceDocumentCard = memo(({ doc, index }: SourceDocumentCardProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const docPosition = index + 1;

  // Check if the source is a web link
  const isWebLink = doc.metadata.source?.startsWith("http");

  const handleAction = () => {
    if (isWebLink && doc.metadata.source) {
      // Open URL in new tab
      window.open(doc.metadata.source, "_blank");
    } else {
      // Fallback: Download file
      const downloadUrl = `/files/download/${encodeURIComponent(
        doc.metadata.file_name
      )}`;
      window.open(downloadUrl, "_blank");
    }
  };

  const getRelevanceBadgeVariant = (score: number) => {
    if (score >= 0.7) return "default";
    if (score >= 0.5) return "secondary";
    return "outline";
  };

  const getRelevanceLabel = (score: number) => {
    if (score >= 0.7) return "High Relevance";
    if (score >= 0.5) return "Medium Relevance";
    return "Low Relevance";
  };

  return (
    <div className="group relative overflow-hidden rounded-xl border border-border/70 bg-card/90 px-4 py-4 transition-all duration-200 hover:border-primary/50 hover:bg-card">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent opacity-0 transition-opacity duration-200 group-hover:opacity-100" />
      <div className="flex flex-wrap items-start justify-between gap-3 pb-3">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/15 text-[11px] font-semibold text-primary">
            #{docPosition}
          </div>
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-foreground">
              <FileText className="h-4 w-4 text-primary" />
              {/* Make title clickable if it's a link */}
              {isWebLink ? (
                <a
                  href={doc.metadata.source || "#"}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[13px] font-semibold leading-tight hover:text-primary transition-colors"
                >
                  {doc.metadata.file_name}
                </a>
              ) : (
                <span className="text-[13px] font-semibold leading-tight">
                  {doc.metadata.file_name}
                </span>
              )}
            </div>
            {doc.metadata.page && (
              <span className="text-xs text-muted-foreground">
                Page {doc.metadata.page}
              </span>
            )}
          </div>
        </div>
        <div className="flex flex-col items-end gap-2 text-[11px] text-muted-foreground">
          <div className="flex items-center gap-2">
            <Badge
              variant={getRelevanceBadgeVariant(doc.score)}
              className="text-[10px] uppercase tracking-wide"
            >
              {getRelevanceLabel(doc.score)}
            </Badge>

            {/* Dynamic Button: Visit Source vs Save PDF */}
            <Button
              variant="ghost"
              size="sm"
              onClick={handleAction}
              className="h-7 px-2 text-[10px] hover:bg-primary/10"
            >
              {isWebLink ? (
                <>
                  <ExternalLink className="mr-1 h-3 w-3" />
                  Visit Source
                </>
              ) : (
                <>
                  <Download className="mr-1 h-3 w-3" />
                  Save PDF
                </>
              )}
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <span>Similarity</span>
            <div className="flex items-center gap-2">
              <div className="relative h-1.5 w-16 overflow-hidden rounded-full bg-muted/70">
                <div
                  className="absolute inset-y-0 left-0 rounded-full bg-primary transition-all duration-500"
                  style={{ width: `${Math.min(doc.score * 100, 100)}%` }}
                />
              </div>
              <span className="font-mono text-[10px] text-primary/80">
                {doc.score.toFixed(3)}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="relative text-[13px] leading-relaxed text-foreground/90">
        {doc.content.length > 200 ? (
          <>
            <p className="transition-all duration-300">
              {isExpanded ? doc.content : `${doc.content.substring(0, 200)}...`}
            </p>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="mt-2 h-7 px-2 text-[10px] text-primary hover:text-primary/80 hover:bg-primary/10"
            >
              {isExpanded ? (
                <>
                  <ChevronUp className="mr-1 h-3 w-3" />
                  Show less
                </>
              ) : (
                <>
                  <ChevronDown className="mr-1 h-3 w-3" />
                  Read more
                </>
              )}
            </Button>
          </>
        ) : (
          <p>{doc.content}</p>
        )}
      </div>
    </div>
  );
});

SourceDocumentCard.displayName = "SourceDocumentCard";

// Message Component (memoized)
interface MessageItemProps {
  message: Message;
}

const MessageItem = memo(({ message }: MessageItemProps) => {
  const formatTimestamp = useCallback((timestamp: string) =>
    new Date(timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    }), []);

  const parseAssistantMessage = useCallback((text: string) => {
    const sanitized = text.replace(/\*\*/g, "");

    const verdictMatch = sanitized.match(/Verdict\s*[:]\s*\[?([^\]\n]+)\]?/i);
    const reasoningMatch = sanitized.match(/Reasoning\s*[:]\s*([^\n]+(?:\n(?!\w+\s*:)[^\n]+)*)/i);
    const confidenceMatch = sanitized.match(/Confidence\s*(?:Score)?\s*[:]\s*([0-9.]+)\s*(%?)/i);
    const evidenceMatch = sanitized.match(/Evidence\s*[:]\s*["']?([^"'\n]+(?:\n(?!\w+\s*:)[^\n]+)*)["']?/i);

    return {
      verdict: verdictMatch ? verdictMatch[1].trim() : null,
      reasoning: reasoningMatch ? reasoningMatch[1].trim() : null,
      confidence: confidenceMatch ? parseFloat(confidenceMatch[1]) : null,
      confidenceIsPercent: confidenceMatch ? confidenceMatch[2] === "%" : false,
      evidence: evidenceMatch ? evidenceMatch[1].trim() : null,
    };
  }, []);

  const getVerdictStyle = useCallback((verdict: string | null) => {
    if (!verdict) return { bg: "bg-muted", text: "text-muted-foreground", border: "border-muted", icon: AlertCircle };

    const v = verdict.toLowerCase();
    if (v.includes("accurate") && !v.includes("inaccurate") && !v.includes("partially")) {
      return { bg: "bg-emerald-500/15", text: "text-emerald-400", border: "border-emerald-500/30", icon: CheckCircle2 };
    }
    if (v.includes("partially accurate")) {
      return { bg: "bg-amber-500/15", text: "text-amber-400", border: "border-amber-500/30", icon: AlertTriangle };
    }
    if (
      v.includes("misleading") ||
      v.includes("inaccurate") ||
      v.includes("not supported") ||
      v.includes("false")
    ) {
      return { bg: "bg-red-500/15", text: "text-red-400", border: "border-red-500/30", icon: XCircle };
    }
    if (v.includes("irrelevant")) {
      return { bg: "bg-muted", text: "text-muted-foreground", border: "border-muted", icon: AlertCircle };
    }
    return { bg: "bg-muted", text: "text-muted-foreground", border: "border-muted", icon: AlertCircle };
  }, []);

  const getConfidenceBarClass = useCallback((score: number) => {
    if (score < 40) return "bg-gradient-to-r from-red-500 to-red-400";
    if (score < 70) return "bg-gradient-to-r from-amber-500 to-amber-400";
    return "bg-gradient-to-r from-emerald-500 to-emerald-400";
  }, []);

  const confidenceScore = useMemo(
    () => (message.type === "assistant" ? parseConfidenceScore(message.content) : null),
    [message.content, message.type]
  );

  const parsedMessage = useMemo(
    () => (message.type === "assistant" ? parseAssistantMessage(message.content) : null),
    [message.content, message.type, parseAssistantMessage]
  );

  return (
    <div className="flex flex-col gap-2">
      <div
        className={`flex ${message.type === "assistant" ? "max-w-[97%]" : "max-w-[95%]"} flex-col gap-2`}
      >
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-wide text-muted-foreground/80">
          {message.type === "assistant" ? (
            <>
              <Shield className="h-3 w-3 text-primary" />
              <span>Verification Result</span>
            </>
          ) : (
            <>
              <FileText className="h-3 w-3 text-secondary" />
              <span>Claim Submitted</span>
            </>
          )}
        </div>
        <div
          className={`rounded-xl border text-[13px] leading-relaxed shadow-sm transition ${message.type === "user"
            ? "border-primary/40 bg-primary/10 text-foreground px-4 py-3"
            : "border-border/70 bg-card/90 shadow-lg"
            }`}
        >
          {message.type === "assistant" && parsedMessage ? (
            <div className="space-y-4 p-5">
              <div className="flex flex-wrap items-center gap-3">
                {parsedMessage.verdict && (() => {
                  const style = getVerdictStyle(parsedMessage.verdict);
                  const VerdictIcon = style.icon;
                  return (
                    <div className={`flex items-center gap-2 rounded-full border px-4 py-2 ${style.bg} ${style.border}`}>
                      <VerdictIcon className={`h-5 w-5 ${style.text}`} />
                      <span className={`text-[14px] font-semibold uppercase tracking-wide ${style.text}`}>
                        {parsedMessage.verdict}
                      </span>
                    </div>
                  );
                })()}
              </div>

              {parsedMessage.reasoning && (
                <div className="space-y-2">
                  <p className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">Reasoning</p>
                  <p className="max-w-none text-[14px] leading-relaxed text-foreground/90">
                    {parsedMessage.reasoning}
                  </p>
                </div>
              )}


              {confidenceScore !== null && (
                <div className="space-y-2 pt-2">
                  <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
                    <span>Analysis Confidence</span>
                    <span className="font-mono text-[11px] font-semibold text-foreground">
                      {confidenceScore.toFixed(0)}%
                    </span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-muted/70">
                    <div
                      className={`h-full rounded-full transition-all duration-1000 ease-out ${getConfidenceBarClass(confidenceScore)}`}
                      style={{ width: `${confidenceScore}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
          ) : message.type === "user" ? (
            <div className="space-y-2 px-4 py-3">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-primary/90">Claim</p>
              <div className="rounded-lg border border-primary/30 bg-background/50 px-3 py-2">
                <p className="whitespace-pre-wrap text-[13px] leading-relaxed text-foreground md:text-[14px]">
                  {message.content as ReactNode}
                </p>
              </div>
            </div>
          ) : (
            <div className="whitespace-pre-wrap px-3 py-2 text-[13px] md:text-[14px]">
              {message.content as ReactNode}
            </div>
          )}
        </div>
        <span className="text-[10px] text-muted-foreground/80">
          {formatTimestamp(message.timestamp)}
        </span>
      </div>
    </div>
  );
});

MessageItem.displayName = "MessageItem";

const parseConfidenceScore = (content: string): number | null => {
  const sanitized = content.replace(/\*\*/g, "");
  const patterns = [
    /confidence\s*score\s*[:]\s*([0-9]+(?:\.[0-9]+)?)\s*(%?)/i,
    /confidence\s*[:]\s*([0-9]+(?:\.[0-9]+)?)\s*(%?)/i,
  ];

  const match = patterns.map((pattern) => sanitized.match(pattern)).find(Boolean);
  if (!match) return null;

  const rawValue = Number(match[1]);
  if (Number.isNaN(rawValue)) return null;

  const isPercent = match[2] === "%";
  const normalized = isPercent ? rawValue : rawValue <= 1 ? rawValue * 100 : rawValue;
  const clamped = Math.min(100, Math.max(0, normalized));
  return clamped;
};

// Extend Window for Web Speech API
interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}
interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}
interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
}
declare global {
  interface Window {
    SpeechRecognition: new () => SpeechRecognition;
    webkitSpeechRecognition: new () => SpeechRecognition;
  }
}

interface AppProps {
  startTutorial?: boolean;
  onTutorialEnd?: () => void;
}

function App({ startTutorial = false, onTutorialEnd }: AppProps = {}) {
  const [inputMessage, setInputMessage] = useState("");
  const [currentMessages, setCurrentMessages] = useState<Message[]>([]);
  const [currentSourceDocuments, setCurrentSourceDocuments] = useState<
    SourceDocument[]
  >([]);

  // Multi-modal input state
  const [showUrlInput, setShowUrlInput] = useState(false);
  const [urlValue, setUrlValue] = useState("");
  const [isExtractingUrl, setIsExtractingUrl] = useState(false);
  const [isExtractingImage, setIsExtractingImage] = useState(false);
  const [extractError, setExtractError] = useState<string | null>(null);
  const [attachedSource, setAttachedSource] = useState<{ type: "url" | "image"; label: string } | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [isTutorialActive, setIsTutorialActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  // Handle tutorial trigger from parent
  useEffect(() => {
    if (startTutorial) {
      setIsTutorialActive(true);
    }
  }, [startTutorial]);
  const {
    sendQuery,
    isLoading: isSendingMessage,
    error: sendError,
  } = useQueryRAG();

  const suggestionPills = useMemo(() => [
    "Long-term use of ibuprofen causes autism",
    "Garlic can cure the flu",
    "Alkaline water has proven health benefits",
    "Vaccines cause autism in children",
  ], []);

  const showWelcomeState = useMemo(
    () => currentMessages.length === 0 && !isSendingMessage,
    [currentMessages.length, isSendingMessage]
  );

  // Check if session has a completed response (not a chatbot, so only one query per session)
  const hasCompletedResponse = useMemo(
    () => currentMessages.length > 0 && !isSendingMessage && currentMessages.some(msg => msg.type === "assistant"),
    [currentMessages, isSendingMessage]
  );

  const handleSuggestionClick = useCallback((value: string) => {
    // Only allow suggestions if no response has been received yet
    if (!hasCompletedResponse) {
      setInputMessage(value);
    }
  }, [hasCompletedResponse]);

  // ===== Multi-modal handlers =====

  const handleExtractUrl = useCallback(async () => {
    if (!urlValue.trim()) return;
    setIsExtractingUrl(true);
    setExtractError(null);
    try {
      const result = await apiClient.extractURL({ url: urlValue.trim() });
      setInputMessage(result.extracted_text);
      setAttachedSource({ type: "url", label: result.page_title || new URL(urlValue.trim()).hostname });
      setShowUrlInput(false);
      setUrlValue("");
    } catch (err) {
      setExtractError(err instanceof Error ? err.message : "Failed to extract text from URL");
    } finally {
      setIsExtractingUrl(false);
    }
  }, [urlValue]);

  const processImage = useCallback(async (file: File) => {
    setIsExtractingImage(true);
    setExtractError(null);
    try {
      const result = await apiClient.extractImage(file);
      setInputMessage(result.extracted_text);
      setAttachedSource({ type: "image", label: file.name });
    } catch (err) {
      setExtractError(err instanceof Error ? err.message : "Failed to extract text from image");
    } finally {
      setIsExtractingImage(false);
    }
  }, []);

  const handleImageUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await processImage(file);
    // Reset file input
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [processImage]);

  const handlePaste = useCallback(async (e: React.ClipboardEvent) => {
    const items = e.clipboardData.items;
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf("image") !== -1) {
        e.preventDefault();
        const file = items[i].getAsFile();
        if (file) {
          // Give pasted images a name
          const renamedFile = new File([file], "pasted-image.png", { type: file.type });
          await processImage(renamedFile);
          return;
        }
      }
    }
  }, [processImage]);

  const handleToggleVoice = useCallback(() => {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      setExtractError("Voice input is not supported in your browser. Please use Chrome or Edge.");
      return;
    }

    if (isListening && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
      return;
    }

    const recognition = new SpeechRec();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    let finalTranscript = inputMessage;

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interimTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += (finalTranscript ? " " : "") + transcript;
        } else {
          interimTranscript += transcript;
        }
      }
      setInputMessage(finalTranscript + (interimTranscript ? " " + interimTranscript : ""));
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      console.error("Speech recognition error:", event.error);
      if (event.error !== "no-speech") {
        setExtractError(`Voice input error: ${event.error}`);
      }
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = recognition;
    recognition.start();
    setIsListening(true);
    setExtractError(null);
  }, [isListening, inputMessage]);

  const handleRemoveAttachment = useCallback(() => {
    setAttachedSource(null);
  }, []);

  const handleSendMessage = useCallback(async () => {
    // Prevent sending if there's already a completed response
    if (!inputMessage.trim() || isSendingMessage || hasCompletedResponse) return;

    // Stop voice recording if active
    if (isListening && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }

    const queryText = inputMessage.trim();
    setInputMessage("");
    setAttachedSource(null);
    setExtractError(null);

    try {
      // Add user message immediately
      const tempUserMessage: Message = {
        id: `temp-${Date.now()}-user`,
        type: "user",
        content: queryText,
        timestamp: new Date().toISOString(),
      };

      setCurrentMessages((prev) => [...prev, tempUserMessage]);

      // Send query to API. Always show 3 top results.
      const response = await sendQuery({ query: queryText, top_k: 3 });

      if (response) {
        // Replace temp message with real messages
        const newMessages = convertQueryResponseToMessages(
          queryText,
          response,
          `query-${Date.now()}`
        );
        setCurrentMessages(newMessages);
        setCurrentSourceDocuments(response.source_documents);
      } else {
        setCurrentMessages((prev) =>
          prev.filter((msg) => msg.id !== tempUserMessage.id)
        );
      }
    } catch (error) {
      console.error("Error sending message:", error);
      setCurrentMessages((prev) =>
        prev.filter((msg) => msg.id.startsWith("temp-"))
      );
    }
  }, [inputMessage, isSendingMessage, hasCompletedResponse, sendQuery, isListening]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    // Prevent sending if there's already a completed response
    if (e.key === "Enter" && !e.shiftKey && !hasCompletedResponse) {
      e.preventDefault();
      handleSendMessage();
    }
  }, [handleSendMessage, hasCompletedResponse]);

  const handleNewChat = useCallback(() => {
    setCurrentMessages([]);
    setCurrentSourceDocuments([]);
    setAttachedSource(null);
    setExtractError(null);
    setShowUrlInput(false);
    setUrlValue("");
    if (isListening && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  }, [isListening]);

  const isProcessing = isExtractingUrl || isExtractingImage;

  const metrics = useMemo(() => {
    const assistantMessages = currentMessages.filter(
      (message) => message.type === "assistant"
    ).length;
    return {
      messages: currentMessages.length,
      assistantMessages,
      sources: currentSourceDocuments.length,
    };
  }, [currentMessages, currentSourceDocuments]);

  return (
    <div className="relative min-h-screen overflow-hidden bg-background text-foreground">
      {/* Background effects */}
      <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -left-24 -top-20 h-80 w-80 rounded-full bg-primary/25 blur-3xl opacity-35" />
        <div className="absolute right-[-10rem] top-1/4 h-[26rem] w-[26rem] rounded-full bg-secondary/20 blur-3xl opacity-45" />
        <div className="absolute bottom-[-7rem] left-1/2 h-72 w-72 -translate-x-1/2 rounded-full bg-accent/20 blur-3xl opacity-35" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,#1c2331,transparent_55%)]" />
      </div>

      <div className="relative z-10 flex min-h-screen flex-col">
        <header className="sticky top-0 z-30 px-4 pt-4 lg:px-6">
          <div className="mx-auto flex w-full max-w-7xl items-center justify-between gap-6 rounded-2xl border border-border/40 bg-background/60 px-6 py-4 shadow-lg shadow-black/5 backdrop-blur-xl backdrop-saturate-150 supports-[backdrop-filter]:bg-background/40 lg:px-8">
            <div className="flex items-center gap-4">
              <div className="relative flex h-14 w-14 items-center justify-center">
                <MedicalCrossLogo size={56} particleCount={12} />
              </div>
              <div className="flex flex-col">
                <h1 className="font-display text-xl font-bold tracking-tight">EvidenceMD</h1>
                <p className="text-[11px] font-medium tracking-wide text-muted-foreground/80">
                  Medical Claim Verifier
                </p>
              </div>
            </div>
            <div className="hidden items-center gap-2.5 lg:flex">
              <div className="flex items-center gap-2 rounded-full border border-border/50 bg-card/50 px-3.5 py-1.5 shadow-sm backdrop-blur-sm">
                <div className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75"></span>
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-primary"></span>
                </div>
                <span className="text-[11px] font-medium tracking-wide text-foreground/90">Live Analysis</span>
              </div>
              <div className="flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3.5 py-1.5 shadow-sm backdrop-blur-sm">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                <span className="text-[11px] font-medium tracking-wide text-emerald-600 dark:text-emerald-400">Peer-reviewed</span>
              </div>
            </div>
            <div className="flex items-center gap-2.5">
              <Button
                onClick={handleNewChat}
                className="h-10 rounded-xl bg-gradient-to-r from-primary to-primary/90 px-5 text-[13px] font-semibold text-primary-foreground shadow-lg shadow-primary/25 transition-all hover:shadow-xl hover:shadow-primary/30 hover:scale-[1.02]"
                data-tutorial="new-session"
              >
                <Plus className="mr-2 h-4 w-4" />
                New Session
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-10 w-10 rounded-xl hover:bg-accent/50"
                title="Tutorial"
                onClick={() => setIsTutorialActive(true)}
              >
                <BookOpen className="h-[18px] w-[18px]" />
              </Button>
            </div>
          </div>
        </header>

        <div className="mx-auto flex w-full max-w-7xl flex-1 flex-col px-4 pb-6 pt-4 lg:px-6">
          <div className="grid flex-1 gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
            <section className="flex min-w-0 flex-col gap-4">
              <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-border/40 bg-background/60 shadow-lg shadow-black/5 backdrop-blur-xl backdrop-saturate-150" data-tutorial="conversation">
                <div className="border-b border-border/40 px-6 py-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/80">Verification</p>
                      <h2 className="font-display text-lg font-bold tracking-tight">Current Analysis</h2>
                    </div>
                    <div className="hidden items-center gap-2 rounded-full border border-border/50 bg-card/50 px-3.5 py-1.5 text-[11px] font-medium tracking-wide text-foreground/90 shadow-sm backdrop-blur-sm md:flex">
                      <div className="relative flex h-2 w-2">
                        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-75"></span>
                        <span className="relative inline-flex h-2 w-2 rounded-full bg-primary"></span>
                      </div>
                      Real-time analysis
                    </div>
                  </div>
                </div>

                <ScrollArea className="flex-1 px-4 pb-3 pt-4">
                  <div className="flex flex-col gap-5">
                    {showWelcomeState && (
                      <div className="rounded-xl border border-border/70 bg-background/60 p-6" data-tutorial="suggestions">
                        <div className="flex items-center gap-3">
                          <Stethoscope className="h-6 w-6 text-primary" />
                          <div>
                            <h3 className="font-display text-xl font-semibold">Claim Intake</h3>
                            <p className="text-[12px] text-muted-foreground">
                              Submit one medical claim for evidence-based verification.
                            </p>
                          </div>
                        </div>
                        <div className="mt-4 grid gap-2 sm:grid-cols-2">
                          {suggestionPills.map((suggestion) => (
                            <Button
                              key={suggestion}
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={() => handleSuggestionClick(suggestion)}
                              className="h-9 rounded-lg border-border/60 bg-background/70 text-[12px] text-foreground/90 transition hover:border-primary/50 hover:bg-primary/10"
                            >
                              <CheckCircle2 className="mr-2 h-4 w-4 text-primary" />
                              {suggestion}
                            </Button>
                          ))}
                        </div>
                      </div>
                    )}

                    {currentMessages.length === 0 && !showWelcomeState ? (
                      <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-border/70 bg-background/60 p-8 text-center text-[12px] text-muted-foreground">
                        <Bot className="h-8 w-8 text-muted-foreground" />
                        Run a claim verification to view the analysis report here.
                      </div>
                    ) : (
                      currentMessages.map((message) => (
                        <MessageItem key={message.id} message={message} />
                      ))
                    )}

                    {isSendingMessage && (
                      <div className="max-w-[92%] rounded-xl border border-border/70 bg-background/70 px-4 py-3">
                        <div className="flex items-center gap-2 text-[10px] uppercase tracking-wide text-muted-foreground/80">
                          <Shield className="h-3 w-3 text-primary" />
                          <span>Evidence Engine</span>
                        </div>
                        <div className="mt-2 flex items-center gap-2 text-[12px]">
                          <Loader2 className="h-4 w-4 animate-spin text-primary" />
                          Running verification...
                        </div>
                      </div>
                    )}
                  </div>
                </ScrollArea>

                <div className="border-t border-border/40 bg-background/60 px-6 py-4 backdrop-blur-xl backdrop-saturate-150">
                  {sendError && (
                    <div className="mb-3 rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2 text-[12px] text-red-200">
                      <div className="flex items-center gap-2 font-medium">
                        <AlertCircle className="h-4 w-4" />
                        Error sending message
                      </div>
                      <p className="mt-1 text-[11px] opacity-80">{sendError}</p>
                    </div>
                  )}

                  {hasCompletedResponse ? (
                    <div className="rounded-xl border border-primary/30 bg-primary/10 px-5 py-4 text-center backdrop-blur-sm">
                      <div className="flex flex-col items-center gap-3">
                        <div className="flex items-center gap-2 text-[13px] font-semibold text-foreground">
                          <Shield className="h-4 w-4 text-primary" />
                          <span>Analysis Complete</span>
                        </div>
                        <p className="text-[12px] text-muted-foreground/90 max-w-md">
                          Verification is complete. Submit a new claim to run a separate, independent analysis.
                        </p>
                        <Button
                          onClick={handleNewChat}
                          className="h-10 rounded-xl bg-gradient-to-r from-primary to-primary/90 px-5 text-[13px] font-semibold text-primary-foreground shadow-lg shadow-primary/25 transition-all hover:shadow-xl hover:shadow-primary/30 hover:scale-[1.02]"
                        >
                          <Plus className="mr-2 h-4 w-4" />
                          Analyze New Claim
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col gap-2">
                      {/* Extraction error */}
                      {extractError && (
                        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-[12px] text-amber-200">
                          <div className="flex items-center gap-2">
                            <AlertCircle className="h-3.5 w-3.5 flex-shrink-0" />
                            <span>{extractError}</span>
                            <button onClick={() => setExtractError(null)} className="ml-auto hover:text-amber-100"><X className="h-3 w-3" /></button>
                          </div>
                        </div>
                      )}

                      {/* Processing indicator */}
                      {isProcessing && (
                        <div className="rounded-lg border border-primary/30 bg-primary/10 px-3 py-2 text-[12px] text-primary">
                          <div className="flex items-center gap-2">
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            <span>{isExtractingUrl ? "Extracting text from URL..." : "Extracting text from image (OCR)..."}</span>
                          </div>
                        </div>
                      )}

                      {/* Attached source chip */}
                      {attachedSource && (
                        <div className="flex items-center gap-2">
                          <div className="flex items-center gap-2 rounded-full border border-primary/40 bg-primary/10 px-3 py-1.5 text-[11px] text-primary">
                            {attachedSource.type === "url" ? <Globe className="h-3 w-3" /> : <Image className="h-3 w-3" />}
                            <span className="max-w-[200px] truncate font-medium">{attachedSource.label}</span>
                            <button
                              onClick={handleRemoveAttachment}
                              className="ml-1 rounded-full p-0.5 hover:bg-primary/20 transition-colors"
                            >
                              <X className="h-2.5 w-2.5" />
                            </button>
                          </div>
                          <span className="text-[10px] text-muted-foreground">Text extracted — review below then send</span>
                        </div>
                      )}

                      {/* URL input field (inline) */}
                      {showUrlInput && (
                        <div className="flex gap-2 rounded-xl border border-primary/30 bg-primary/5 p-2">
                          <Input
                            placeholder="Paste article URL (e.g., https://bbc.com/health/article)..."
                            value={urlValue}
                            onChange={(e) => setUrlValue(e.target.value)}
                            onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); handleExtractUrl(); } }}
                            className="h-9 flex-1 rounded-lg border-border/40 bg-background/60 text-[12px]"
                            disabled={isExtractingUrl}
                            autoFocus
                          />
                          <Button
                            onClick={handleExtractUrl}
                            disabled={!urlValue.trim() || isExtractingUrl}
                            size="sm"
                            className="h-9 rounded-lg px-4 text-[11px]"
                          >
                            {isExtractingUrl ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Extract"}
                          </Button>
                          <Button
                            onClick={() => { setShowUrlInput(false); setUrlValue(""); }}
                            variant="ghost"
                            size="sm"
                            className="h-9 w-9 rounded-lg p-0"
                          >
                            <X className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      )}

                      {/* Main input row */}
                      <div className="flex flex-col gap-2 sm:flex-row items-end" data-tutorial="input">
                        <div className="relative flex-1 w-full">
                          <Textarea
                            placeholder={isListening ? "Listening... speak your claim" : "Enter a medical claim to verify..."}
                            value={inputMessage}
                            onChange={(e) => setInputMessage(e.target.value)}
                            onKeyDown={handleKeyDown}
                            onPaste={handlePaste}
                            className={`min-h-[48px] max-h-[300px] w-full resize-none rounded-xl border-border/40 bg-card/50 py-3 pr-16 text-[13px] shadow-sm backdrop-blur-sm transition-all focus:border-primary/60 focus:ring-2 focus:ring-primary/20 ${isListening ? "border-red-500/50 ring-2 ring-red-500/20" : ""}`}
                            disabled={isSendingMessage || isProcessing}
                          />
                          <div className="pointer-events-none absolute bottom-3 right-4 hidden items-center gap-2 text-[10px] text-muted-foreground sm:flex">
                            <span>Press</span>
                            <kbd className="rounded-md border border-border/60 bg-background/70 px-2 py-1 text-[9px] uppercase">
                              Enter
                            </kbd>
                          </div>
                        </div>
                        <Button
                          onClick={handleSendMessage}
                          className="h-12 rounded-xl bg-gradient-to-r from-primary to-primary/90 px-6 text-[13px] font-semibold text-primary-foreground shadow-lg shadow-primary/25 transition-all hover:shadow-xl hover:shadow-primary/30 hover:scale-[1.02]"
                          disabled={!inputMessage.trim() || isSendingMessage || isProcessing}
                        >
                          {isSendingMessage ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <>
                              <Send className="mr-2 h-4 w-4" />
                              Run Verification
                            </>
                          )}
                        </Button>
                      </div>

                      {/* Multi-modal toolbar */}
                      <div className="flex items-center gap-1 pt-1" data-tutorial="multimodal">
                        {/* Hidden file input */}
                        <input
                          ref={fileInputRef}
                          type="file"
                          accept="image/png,image/jpeg,image/jpg,image/webp,image/bmp,image/tiff"
                          onChange={handleImageUpload}
                          className="hidden"
                        />

                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => { setShowUrlInput(!showUrlInput); setExtractError(null); }}
                          disabled={isSendingMessage || isProcessing}
                          className={`h-8 gap-1.5 rounded-lg px-2.5 text-[11px] font-medium transition-all ${showUrlInput
                            ? "bg-primary/15 text-primary border border-primary/30"
                            : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                            }`}
                          title="Paste an article URL to extract and analyze its text"
                        >
                          <Link className="h-3.5 w-3.5" />
                          <span className="hidden sm:inline">Link</span>
                        </Button>

                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => fileInputRef.current?.click()}
                          disabled={isSendingMessage || isProcessing}
                          className="h-8 gap-1.5 rounded-lg px-2.5 text-[11px] font-medium text-muted-foreground hover:text-foreground hover:bg-accent/50 transition-all"
                          title="Upload a screenshot or image to extract text via OCR"
                        >
                          <Image className="h-3.5 w-3.5" />
                          <span className="hidden sm:inline">Image</span>
                        </Button>

                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={handleToggleVoice}
                          disabled={isSendingMessage || isProcessing}
                          className={`h-8 gap-1.5 rounded-lg px-2.5 text-[11px] font-medium transition-all ${isListening
                            ? "bg-red-500/15 text-red-400 border border-red-500/30"
                            : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
                            }`}
                          title={isListening ? "Stop listening" : "Speak your claim (uses browser speech recognition)"}
                        >
                          {isListening ? (
                            <>
                              <span className="relative flex h-2 w-2">
                                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75"></span>
                                <span className="relative inline-flex h-2 w-2 rounded-full bg-red-500"></span>
                              </span>
                              <MicOff className="h-3.5 w-3.5" />
                              <span className="hidden sm:inline">Stop</span>
                            </>
                          ) : (
                            <>
                              <Mic className="h-3.5 w-3.5" />
                              <span className="hidden sm:inline">Voice</span>
                            </>
                          )}
                        </Button>

                        <div className="ml-auto text-[10px] text-muted-foreground/60">
                          Paste a link, upload a screenshot, or speak
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Persistent disclaimer */}
                  <div className="mt-3 flex items-start gap-2 rounded-lg border border-border/40 bg-muted/30 px-3 py-2.5 text-[11px] sm:text-[12px] text-muted-foreground">
                    <AlertCircle className="h-4 w-4 flex-shrink-0 text-red-500 mt-0.5" />
                    <p className="leading-relaxed">
                      <span className="font-semibold text-foreground">EvidenceMD</span> can be inaccurate; please double check its responses and sources it analyzes.
                    </p>
                  </div>
                </div>
              </div>
            </section>

            <aside className="flex flex-col gap-4" data-tutorial="sources">
              <div className="rounded-2xl border border-border/40 bg-background/60 p-5 shadow-lg shadow-black/5 backdrop-blur-xl backdrop-saturate-150">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/80">Sources</p>
                    <h3 className="font-display text-lg font-bold tracking-tight">Evidence Stack</h3>
                  </div>
                  <span className="rounded-full border border-border/50 bg-card/50 px-3 py-1 text-[10px] font-medium uppercase tracking-wide text-foreground/90 shadow-sm backdrop-blur-sm">
                    {metrics.sources} docs
                  </span>
                </div>
              </div>

              <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-border/40 bg-background/60 shadow-lg shadow-black/5 backdrop-blur-xl backdrop-saturate-150">
                <ScrollArea className="flex-1 px-4 py-4">
                  {currentSourceDocuments.length === 0 ? (
                    <div className="rounded-xl border border-dashed border-border/70 bg-background/60 p-6 text-center text-[12px] text-muted-foreground">
                      {currentMessages.length === 0
                        ? "Source documents will appear here once you ask something."
                        : "No supporting documents were returned for this response."}
                    </div>
                  ) : (
                    <div className="flex flex-col gap-3">
                      {currentSourceDocuments.map((doc, index) => (
                        <SourceDocumentCard key={index} doc={doc} index={index} />
                      ))}
                    </div>
                  )}
                </ScrollArea>
              </div>
            </aside>
          </div>
        </div>
      </div>

      <Analytics />
      <SpeedInsights />
      
      <Tutorial
        isActive={isTutorialActive}
        onComplete={() => {
          setIsTutorialActive(false);
          localStorage.setItem("tutorialCompleted", "true");
          onTutorialEnd?.();
        }}
        onSkip={() => {
          setIsTutorialActive(false);
          localStorage.setItem("tutorialCompleted", "true");
          onTutorialEnd?.();
        }}
      />
    </div>
  );
}

export default App;