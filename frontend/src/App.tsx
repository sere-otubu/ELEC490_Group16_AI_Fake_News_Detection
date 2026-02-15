import { useState, useMemo, useCallback, memo, type ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Plus,
  Bot,
  User,
  Send,
  Loader2,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Download,
  FileText,
  BookOpen,
  Settings,
  Shield,
  CheckCircle2,
  Activity,
  Stethoscope,
  ExternalLink,
} from "lucide-react";
import {
  useQueryRAG,
  convertQueryResponseToMessages,
} from "@/hooks/useApi";
import type { Message, SourceDocument } from "@/types/api";

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

  const renderInlineBold = useCallback((text: string, keyPrefix: string) => {
    const parts = text.split(/\*\*(.*?)\*\*/g);
    return parts.map((part, index) =>
      index % 2 === 1 ? (
        <strong key={`${keyPrefix}-b-${index}`} className="font-semibold text-foreground">
          {part}
        </strong>
      ) : (
        <span key={`${keyPrefix}-t-${index}`}>{part}</span>
      )
    );
  }, []);

  const renderAssistantContent = useCallback((text: string) => {
    const lines = text.split(/\n+/);
    return lines.map((line, index) => {
      if (!line.trim()) {
        return <div key={`line-${index}`} className="h-2" />;
      }

      const match = line.match(/^\s*(Verdict|Reasoning|Confidence Score|Evidence)\s*[:\-]\s*/i);
      if (match) {
        const label = match[1];
        const rest = line.slice(match[0].length);
        const isVerdict = label.toLowerCase() === "verdict";
        return (
          <div key={`line-${index}`} className="flex flex-wrap items-baseline gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-primary">
              {label}
            </span>
            <span
              className={`text-[13px] ${
                isVerdict ? "font-semibold text-foreground" : "text-foreground/90"
              }`}
            >
              {renderInlineBold(rest, `inline-${index}`)}
            </span>
          </div>
        );
      }

      return (
        <p key={`line-${index}`} className="text-[13px] text-foreground/90">
          {renderInlineBold(line, `inline-${index}`)}
        </p>
      );
    });
  }, [renderInlineBold]);

  const getConfidenceBarClass = useCallback((score: number) => {
    if (score < 40) return "bg-destructive";
    if (score < 70) return "bg-amber-400";
    return "bg-emerald-400";
  }, []);

  const confidenceScore = useMemo(
    () => (message.type === "assistant" ? parseConfidenceScore(message.content) : null),
    [message.content, message.type]
  );

  return (
    <div
      className={`flex items-end gap-3 ${
        message.type === "user" ? "flex-row-reverse" : ""
      }`}
    >
      <Avatar className="h-8 w-8 border border-border/70 bg-background/80 text-primary">
        <AvatarFallback>
          {message.type === "assistant" ? (
            <Bot className="h-4 w-4" />
          ) : (
            <User className="h-4 w-4" />
          )}
        </AvatarFallback>
      </Avatar>
      <div
        className={`flex max-w-[78%] flex-col gap-2 ${
          message.type === "user"
            ? "items-end text-right"
            : ""
        }`}
      >
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-wide text-muted-foreground/80">
          {message.type === "assistant" ? (
            <>
              <Shield className="h-3 w-3 text-primary" />
              <span>Medical Verifier</span>
            </>
          ) : (
            <>
              <User className="h-3 w-3 text-secondary" />
              <span>You</span>
            </>
          )}
        </div>
        <div
          className={`rounded-xl border px-3 py-2 text-[13px] leading-relaxed shadow-sm transition ${
            message.type === "user"
              ? "border-primary/70 bg-primary text-primary-foreground shadow-primary/20"
              : "border-border/70 bg-background/70"
          }`}
        >
          <div className="whitespace-pre-wrap text-[13px] md:text-[14px]">
            {message.type === "assistant"
              ? renderAssistantContent(message.content)
              : (message.content as ReactNode)}
          </div>
          {message.type === "assistant" && confidenceScore !== null && (
            <div className="mt-3 space-y-2">
              <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
                <span>Confidence</span>
                <span className="font-mono text-foreground">
                  {confidenceScore.toFixed(0)}%
                </span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-muted/70">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${getConfidenceBarClass(confidenceScore)}`}
                  style={{ width: `${confidenceScore}%` }}
                />
              </div>
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
    /confidence\s*score\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*(%?)/i,
    /confidence\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*(%?)/i,
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

function App() {
  const [inputMessage, setInputMessage] = useState("");
  const [currentMessages, setCurrentMessages] = useState<Message[]>([]);
  const [currentSourceDocuments, setCurrentSourceDocuments] = useState<
    SourceDocument[]
  >([]);

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

  const handleSendMessage = useCallback(async () => {
    // Prevent sending if there's already a completed response
    if (!inputMessage.trim() || isSendingMessage || hasCompletedResponse) return;

    const queryText = inputMessage.trim();
    setInputMessage("");

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
  }, [inputMessage, isSendingMessage, hasCompletedResponse, sendQuery]);

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
  }, []);

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
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#8a9bb812_1px,transparent_1px),linear-gradient(to_bottom,#8a9bb812_1px,transparent_1px)] bg-[size:20px_20px]" />
      </div>

      <div className="relative z-10 flex min-h-screen flex-col">
        <header className="sticky top-0 z-30 border-b border-border/70 bg-card/85 backdrop-blur">
          <div className="mx-auto flex w-full max-w-7xl items-center justify-between gap-4 px-4 py-3 lg:px-6">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/15 text-primary">
                <Shield className="h-5 w-5" />
              </div>
              <div>
                <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
                  Medical Claim Verifier
                </p>
                <h1 className="font-display text-lg font-semibold">Evidence Console</h1>
              </div>
            </div>
            <div className="hidden items-center gap-2 rounded-full border border-border/60 bg-background/60 px-3 py-1 text-[11px] uppercase tracking-wide text-muted-foreground md:flex">
              <Activity className="h-3.5 w-3.5 text-primary" />
              Analysis Session
            </div>
            <div className="flex items-center gap-2">
              <Button
                onClick={handleNewChat}
                className="h-9 rounded-full bg-primary/90 px-4 text-[12px] font-semibold text-primary-foreground shadow-sm shadow-primary/30 transition hover:bg-primary"
              >
                <Plus className="mr-2 h-4 w-4" />
                New Session
              </Button>
              <Button variant="ghost" size="icon" className="h-9 w-9" title="Settings">
                <Settings className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </header>

        <div className="mx-auto flex w-full max-w-7xl flex-1 flex-col px-4 pb-6 pt-4 lg:px-6">
          <div className="grid flex-1 gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
            <section className="flex min-w-0 flex-col gap-4">
              <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-border/70 bg-card/90">
                <div className="border-b border-border/70 px-4 py-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Conversation</p>
                      <h2 className="font-display text-lg font-semibold">Current Session</h2>
                    </div>
                    <div className="hidden items-center gap-2 text-[11px] text-muted-foreground md:flex">
                      <Activity className="h-3.5 w-3.5 text-accent" />
                      Real-time analysis
                    </div>
                  </div>
                </div>

                <ScrollArea className="flex-1 px-4 pb-3 pt-4">
                  <div className="flex flex-col gap-5">
                    {showWelcomeState && (
                      <div className="rounded-xl border border-border/70 bg-background/60 p-6">
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
                        Ask a question to see the conversation flow here.
                      </div>
                    ) : (
                      currentMessages.map((message) => (
                        <MessageItem key={message.id} message={message} />
                      ))
                    )}

                    {isSendingMessage && (
                      <div className="flex items-end gap-3">
                        <Avatar className="h-8 w-8 border border-border/70 bg-background/80 text-primary">
                          <AvatarFallback>
                            <Bot className="h-4 w-4" />
                          </AvatarFallback>
                        </Avatar>
                        <div className="flex max-w-[78%] flex-col gap-2">
                          <div className="flex items-center gap-2 text-[10px] uppercase tracking-wide text-muted-foreground/80">
                            <Shield className="h-3 w-3 text-primary" />
                            <span>Medical Verifier</span>
                          </div>
                          <div className="rounded-xl border border-border/70 bg-background/70 px-3 py-2">
                            <div className="flex items-center gap-2 text-[12px]">
                              <Loader2 className="h-4 w-4 animate-spin text-primary" />
                              Thinking...
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </ScrollArea>

                <div className="border-t border-border/70 bg-card/95 px-4 py-3">
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
                    <div className="rounded-xl border border-primary/30 bg-primary/10 px-4 py-3 text-center">
                      <div className="flex flex-col items-center gap-3">
                        <div className="flex items-center gap-2 text-[12px] font-semibold text-foreground">
                          <Shield className="h-4 w-4 text-primary" />
                          <span>Response Complete</span>
                        </div>
                        <p className="text-[12px] text-muted-foreground max-w-md">
                          This is a fact-checker, not a chatbot. Each query is independent and doesn't maintain context from previous messages.
                        </p>
                        <Button
                          onClick={handleNewChat}
                          className="h-10 rounded-full bg-primary/90 px-5 text-[12px] font-semibold text-primary-foreground shadow-sm shadow-primary/30 transition hover:bg-primary"
                        >
                          <Plus className="mr-2 h-4 w-4" />
                          Start New Fact-Check Session
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col gap-3 sm:flex-row">
                      <div className="relative flex-1">
                        <Input
                          placeholder="Enter a medical claim to verify (e.g., 'Long-term use of ibuprofen causes autism')..."
                          value={inputMessage}
                          onChange={(e) => setInputMessage(e.target.value)}
                          onKeyDown={handleKeyDown}
                          className="h-11 rounded-full border-border/60 bg-background/80 pr-16 text-[12px] shadow-inner shadow-black/20 transition-all focus:border-primary/60 focus:ring-2 focus:ring-primary/20"
                          disabled={isSendingMessage}
                        />
                        <div className="pointer-events-none absolute inset-y-0 right-4 hidden items-center gap-2 text-[10px] text-muted-foreground sm:flex">
                          <span>Press</span>
                          <kbd className="rounded-md border border-border/60 bg-background/70 px-2 py-1 text-[9px] uppercase">
                            Enter
                          </kbd>
                        </div>
                      </div>
                      <Button
                        onClick={handleSendMessage}
                        className="h-11 rounded-full bg-primary/90 px-6 text-[12px] font-semibold text-primary-foreground shadow-sm shadow-primary/30 transition hover:bg-primary"
                        disabled={!inputMessage.trim() || isSendingMessage}
                      >
                        {isSendingMessage ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <>
                            <Send className="mr-2 h-4 w-4" />
                            Send
                          </>
                        )}
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            </section>

            <aside className="flex flex-col gap-4">
              <div className="rounded-xl border border-border/70 bg-card/90 p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Sources</p>
                    <h3 className="font-display text-lg font-semibold">Evidence Stack</h3>
                  </div>
                  <span className="rounded-full border border-border/60 px-2 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                    {metrics.sources} docs
                  </span>
                </div>
              </div>

              <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-border/70 bg-card/90">
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
    </div>
  );
}

export default App;