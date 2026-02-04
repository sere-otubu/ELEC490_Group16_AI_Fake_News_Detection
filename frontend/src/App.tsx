import { useState, useMemo, useCallback, memo } from "react";
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
  Lightbulb,
  BookOpen,
  Menu,
  X,
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
    <div className="group relative overflow-hidden rounded-2xl border border-border/60 bg-card/70 px-6 py-5 backdrop-blur transition-all duration-300 hover:border-primary/60 hover:bg-card/80">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/60 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
      <div className="flex flex-wrap items-start justify-between gap-4 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/15 text-xs font-semibold text-primary">
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
                  className="text-sm font-semibold leading-tight hover:underline hover:text-primary transition-colors"
                >
                  {doc.metadata.file_name}
                </a>
              ) : (
                <span className="text-sm font-semibold leading-tight">
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
        <div className="flex flex-col items-end gap-2 text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <Badge
              variant={getRelevanceBadgeVariant(doc.score)}
              className="text-[11px] uppercase tracking-wide"
            >
              {getRelevanceLabel(doc.score)}
            </Badge>
            
            {/* Dynamic Button: Visit Source vs Save PDF */}
            <Button
              variant="ghost"
              size="sm"
              onClick={handleAction}
              className="h-7 px-2 text-[11px] hover:bg-primary/10"
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
              <div className="relative h-1.5 w-16 overflow-hidden rounded-full bg-muted/60">
                <div
                  className="absolute inset-y-0 left-0 rounded-full bg-primary transition-all duration-500"
                  style={{ width: `${Math.min(doc.score * 100, 100)}%` }}
                />
              </div>
              <span className="font-mono text-[11px] text-primary/80">
                {doc.score.toFixed(3)}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="relative text-sm leading-relaxed text-foreground/90">
        {doc.content.length > 200 ? (
          <>
            <p className="transition-all duration-300">
              {isExpanded ? doc.content : `${doc.content.substring(0, 200)}...`}
            </p>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="mt-3 h-7 px-2 text-[11px] text-primary hover:text-primary/80 hover:bg-primary/10"
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

  return (
    <div
      className={`flex items-end gap-3 ${
        message.type === "user" ? "flex-row-reverse" : ""
      }`}
    >
      <Avatar className="h-9 w-9 border border-border/60 bg-background/80 text-primary">
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
        <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground/80">
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
          className={`rounded-2xl border px-4 py-3 text-sm leading-relaxed shadow-sm transition ${
            message.type === "user"
              ? "border-primary/70 bg-primary text-primary-foreground shadow-primary/20"
              : "border-border/60 bg-background/65 backdrop-blur"
          }`}
        >
          <p className="whitespace-pre-wrap text-sm md:text-[15px]">
            {message.content}
          </p>
        </div>
        <span className="text-[11px] text-muted-foreground/80">
          {formatTimestamp(message.timestamp)}
        </span>
      </div>
    </div>
  );
});

MessageItem.displayName = "MessageItem";

function App() {
  const [inputMessage, setInputMessage] = useState("");
  const [currentMessages, setCurrentMessages] = useState<Message[]>([]);
  const [currentSourceDocuments, setCurrentSourceDocuments] = useState<
    SourceDocument[]
  >([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

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

      // Send query to API
      const response = await sendQuery({ query: queryText, top_k: 2 });

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
    setIsSidebarOpen(false); 
  }, []);

  const toggleSidebar = useCallback(() => {
    setIsSidebarOpen((prev) => !prev);
  }, []);

  return (
    <div className="relative min-h-screen overflow-hidden bg-background text-foreground">
      {/* Background effects */}
      <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -left-24 -top-32 h-96 w-96 rounded-full bg-primary/20 blur-3xl opacity-40" />
        <div className="absolute right-[-8rem] top-1/3 h-[28rem] w-[28rem] rounded-full bg-secondary/15 blur-3xl opacity-50" />
        <div className="absolute bottom-[-6rem] left-1/2 h-80 w-80 -translate-x-1/2 rounded-full bg-accent/20 blur-3xl opacity-40" />
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]" />
      </div>

      <div className="relative z-10 flex min-h-screen w-full">
        {/* Mobile Sidebar Overlay */}
        {isSidebarOpen && (
          <div
            className="fixed inset-0 z-40 bg-black/50 lg:hidden"
            onClick={() => setIsSidebarOpen(false)}
          />
        )}

        {/* Sidebar */}
        <aside
          className={`fixed left-0 top-0 z-50 h-full flex-col border-r border-border/60 bg-card/95 backdrop-blur transition-all duration-300 lg:relative lg:z-auto ${
            isSidebarOpen
              ? "w-[320px] translate-x-0"
              : "-translate-x-full lg:translate-x-0 lg:w-16"
          }`}
        >
          {/* Collapsed State */}
          {!isSidebarOpen && (
            <div className="hidden lg:flex flex-col items-center h-full py-4 gap-4 w-16 shrink-0">
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleSidebar}
                className="h-10 w-10"
              >
                <Menu className="h-5 w-5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleNewChat}
                className="h-10 w-10"
                title="New conversation"
              >
                <Plus className="h-5 w-5" />
              </Button>
              <div className="flex-1" />
              <Button
                variant="ghost"
                size="icon"
                className="h-10 w-10"
                title="Settings"
              >
                <Settings className="h-5 w-5" />
              </Button>
            </div>
          )}

          {/* Expanded State */}
          {isSidebarOpen && (
          <div className="flex h-full w-[320px] flex-col gap-6 p-6 overflow-hidden">
            <div className="flex items-center justify-between shrink-0">
              <div className="flex-1">
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={toggleSidebar}
                      className="h-9 w-9 lg:flex"
                    >
                      <Menu className="h-5 w-5" />
                    </Button>
                    <div className="flex items-center gap-2">
                      <Shield className="h-6 w-6 text-primary" />
                      <h1 className="text-2xl font-semibold leading-tight">
                        Medical Info
                      </h1>
                    </div>
                  </div>
                </div>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsSidebarOpen(false)}
                className="h-9 w-9 lg:hidden"
              >
                <X className="h-5 w-5" />
              </Button>
            </div>

            <div className="space-y-5 shrink-0">
              <Button
                className="h-11 w-full rounded-2xl bg-primary/90 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/30 transition hover:bg-primary"
                onClick={handleNewChat}
              >
                <Plus className="mr-2 h-4 w-4" />
                Start new conversation
              </Button>
            </div>

            {/* Sidebar Content */}
            <div className="flex-1 min-h-0 overflow-hidden">
              <div className="rounded-2xl border border-border/50 bg-background/45 p-5 backdrop-blur-md">
                <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                  <Lightbulb className="h-4 w-4 text-primary" />
                  Try asking about
                </div>
                <ul className="mt-3 space-y-2 text-xs text-muted-foreground">
                {suggestionPills.map((suggestion) => (
                  <li key={`tip-${suggestion}`}>
                    <button
                      type="button"
                      onClick={() => handleSuggestionClick(suggestion)}
                      disabled={hasCompletedResponse}
                      className="flex w-full items-center gap-2 rounded-xl border border-transparent bg-transparent px-2 py-1 text-left transition hover:border-primary/40 hover:bg-primary/10 hover:text-foreground disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:border-transparent disabled:hover:bg-transparent"
                    >
                      <span className="mt-0.5 h-1.5 w-1.5 rounded-full bg-primary/60" />
                      {suggestion}
                    </button>
                  </li>
                ))}
                </ul>
              </div>
            </div>
          </div>
          )}
        </aside>

        {/* Main Content */}
        <main
          className={`flex-1 overflow-hidden rounded-3xl border border-border/60 bg-card/60 shadow-[0_25px_65px_-45px_rgba(0,0,0,0.85)] backdrop-blur transition-all duration-300 mx-4 my-6 lg:mx-6 lg:my-6 ${
            !isSidebarOpen ? "lg:mx-auto lg:max-w-5xl" : ""
          }`}
        >
          <div className="flex h-full flex-col">
            <div className="border-b border-border/50 bg-card/70 px-4 py-4 md:px-6 md:py-6 lg:px-10 lg:py-8">
              <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="space-y-3">
                    <span className="inline-flex items-center gap-2 rounded-full border border-primary/40 bg-primary/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-wide text-primary">
                      <Activity className="h-3.5 w-3.5" />
                      Analysis Session
                    </span>
                    <div>
                      <h2 className="text-xl font-semibold tracking-tight md:text-2xl lg:text-3xl">
                        Current Session
                      </h2>
                      <p className="mt-2 max-w-xl text-sm text-muted-foreground">
                        Verify medical claims, check treatment accuracy, or get evidence-based information.
                      </p>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-2 lg:hidden">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={handleNewChat}
                      className="h-9 w-9"
                      title="New conversation"
                    >
                      <Plus className="h-5 w-5" />
                    </Button>
                  </div>
                </div>
              </div>
              {showWelcomeState && (
                <div className="mt-8 rounded-3xl border border-primary/20 bg-gradient-to-br from-card/80 to-card/40 p-8 text-center backdrop-blur md:p-12">
                  <div className="mx-auto max-w-2xl space-y-8">
                    <div className="inline-flex items-center gap-2 rounded-full border border-primary/40 bg-primary/10 px-4 py-1.5 text-xs font-semibold uppercase tracking-wide text-primary">
                      <Shield className="h-4 w-4" />
                      <span>Evidence-Based Verification</span>
                    </div>
                    <div className="space-y-4">
                      <div className="flex items-center justify-center gap-3">
                        <Stethoscope className="h-8 w-8 text-primary md:h-10 md:w-10" />
                        <h3 className="text-3xl font-bold md:text-4xl bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text text-transparent">
                          Medical Claim Verifier
                        </h3>
                      </div>
                      <p className="text-base text-muted-foreground md:text-lg leading-relaxed max-w-xl mx-auto">
                        Verify medical claims with confidence. Our AI-powered system analyzes 
                        claims against peer-reviewed medical literature to provide accurate, 
                        evidence-based assessments.
                      </p>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-4">
                      {suggestionPills.map((suggestion) => (
                        <Button
                          key={suggestion}
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => handleSuggestionClick(suggestion)}
                          className="rounded-xl border-border/60 bg-background/70 text-sm text-foreground/90 transition-all hover:border-primary/50 hover:bg-primary/10 hover:text-foreground hover:shadow-md hover:scale-[1.02]"
                        >
                          <CheckCircle2 className="mr-2 h-4 w-4 text-primary" />
                          {suggestion}
                        </Button>
                      ))}
                    </div>
                    <div className="flex items-center justify-center gap-6 pt-4 text-xs text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <Activity className="h-4 w-4 text-accent" />
                        <span>Real-time Analysis</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <BookOpen className="h-4 w-4 text-primary" />
                        <span>Peer-Reviewed Sources</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Shield className="h-4 w-4 text-accent" />
                        <span>Verified Accuracy</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="flex-1 overflow-hidden">
              <ScrollArea className="h-full px-6 py-6 md:px-10 md:py-8">
                <div className="mx-auto flex max-w-4xl flex-col gap-8">
                  <div className="flex flex-col gap-6">
                    {currentMessages.length === 0 ? (
                      !showWelcomeState && (
                        <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-border/60 bg-background/50 p-10 text-center text-sm text-muted-foreground">
                          <Bot className="h-10 w-10 text-muted-foreground" />
                          Ask a question to see the conversation flow here.
                        </div>
                      )
                    ) : (
                      currentMessages.map((message) => (
                        <MessageItem key={message.id} message={message} />
                      ))
                    )}

                    {isSendingMessage && (
                      <div className="flex items-end gap-3">
                        <Avatar className="h-9 w-9 border border-border/60 bg-background/80 text-primary">
                          <AvatarFallback>
                            <Bot className="h-4 w-4" />
                          </AvatarFallback>
                        </Avatar>
                        <div className="flex max-w-[78%] flex-col gap-2">
                          <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground/80">
                            <Shield className="h-3 w-3 text-primary" />
                            <span>Medical Verifier</span>
                          </div>
                          <div className="rounded-2xl border border-border/60 bg-background/65 px-4 py-3 backdrop-blur">
                            <div className="flex items-center gap-2 text-sm">
                              <Loader2 className="h-4 w-4 animate-spin text-primary" />
                              Thinking...
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  <div>
                    <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-foreground">
                      <BookOpen className="h-5 w-5 text-primary" />
                      Source documents
                    </div>
                    {currentSourceDocuments.length === 0 ? (
                      <div className="rounded-2xl border border-dashed border-border/70 bg-background/40 p-8 text-center text-sm text-muted-foreground">
                        {currentMessages.length === 0
                          ? "Source documents will appear here once you ask something."
                          : "No supporting documents were returned for this response."}
                      </div>
                    ) : (
                      <div className="grid gap-4 lg:grid-cols-2">
                        {currentSourceDocuments.map((doc, index) => (
                          <SourceDocumentCard
                            key={index}
                            doc={doc}
                            index={index}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </ScrollArea>
            </div>

            <div className="border-t border-border/60 bg-card/70 px-6 py-6 md:px-10">
              <div className="mx-auto flex max-w-4xl flex-col gap-4">
                {sendError && (
                  <div className="rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                    <div className="flex items-center gap-2 font-medium">
                      <AlertCircle className="h-4 w-4" />
                      Error sending message
                    </div>
                    <p className="mt-1 text-xs opacity-80">{sendError}</p>
                  </div>
                )}
                
                {hasCompletedResponse ? (
                  <div className="rounded-2xl border border-primary/30 bg-primary/10 px-6 py-5 text-center">
                    <div className="flex flex-col items-center gap-4">
                      <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                        <Shield className="h-5 w-5 text-primary" />
                        <span>Response Complete</span>
                      </div>
                      <p className="text-sm text-muted-foreground max-w-md">
                        This is a fact-checker, not a chatbot. Each query is independent and doesn't maintain context from previous messages.
                      </p>
                      <Button
                        onClick={handleNewChat}
                        className="h-11 rounded-2xl bg-primary/90 px-6 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/30 transition hover:bg-primary"
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
                        className="h-12 rounded-2xl border-border/60 bg-background/80 pr-16 text-sm shadow-inner shadow-black/20 backdrop-blur transition-all focus:border-primary/60 focus:ring-2 focus:ring-primary/20"
                        disabled={isSendingMessage}
                      />
                      <div className="pointer-events-none absolute inset-y-0 right-4 hidden items-center gap-2 text-[11px] text-muted-foreground sm:flex">
                        <span>Press</span>
                        <kbd className="rounded-md border border-border/60 bg-background/70 px-2 py-1 text-[10px] uppercase">
                          Enter
                        </kbd>
                      </div>
                    </div>
                    <Button
                      onClick={handleSendMessage}
                      className="h-12 rounded-2xl bg-primary/90 px-6 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/30 transition hover:bg-primary"
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
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;