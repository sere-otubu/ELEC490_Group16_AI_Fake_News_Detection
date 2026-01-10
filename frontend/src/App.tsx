import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Plus, Bot, User, Send, Loader2, AlertCircle, ChevronDown, ChevronUp,
  Download, FileText, ShieldCheck, History, Activity, Microscope,
  PanelLeft, X
} from "lucide-react";
import {
  useQueryHistory, useQuery, useQueryRAG,
  convertQueryResponseToMessages, convertQueryHistoryToChats,
} from "@/hooks/useApi";
import type { Message, SourceDocument } from "@/types/api";

interface SourceDocumentCardProps {
  doc: SourceDocument;
  index: number;
}

const SourceDocumentCard = ({ doc, index }: SourceDocumentCardProps) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const docPosition = index + 1;

  const handleDownload = () => {
    alert(`Downloading ${doc.metadata.file_name}...`);
  };

  const getRelevanceBadgeVariant = (score: number) => {
    if (score >= 0.8) return "default";
    if (score >= 0.6) return "secondary";
    return "outline";
  };

  const getRelevanceLabel = (score: number) => {
    if (score >= 0.8) return "Strong Evidence";
    if (score >= 0.6) return "Related Study";
    return "Loose Mention";
  };

  return (
    <div className="group relative overflow-hidden rounded-2xl border border-border/60 bg-card px-6 py-5 shadow-sm transition-all duration-300 hover:border-primary/60 hover:shadow-md">
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/60 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />
      <div className="flex flex-wrap items-start justify-between gap-4 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-xs font-semibold text-primary">
            Ref {docPosition}
          </div>
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-foreground">
              <FileText className="h-4 w-4 text-primary" />
              <span className="text-sm font-semibold leading-tight">{doc.metadata.file_name}</span>
            </div>
            {doc.metadata.page && (
              <span className="text-xs text-muted-foreground">Page {doc.metadata.page}</span>
            )}
          </div>
        </div>
        <div className="flex flex-col items-end gap-2 text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <Badge variant={getRelevanceBadgeVariant(doc.score)} className="text-[11px] uppercase tracking-wide">
              {getRelevanceLabel(doc.score)}
            </Badge>
            <Button variant="ghost" size="sm" onClick={handleDownload} className="h-7 px-2 text-[11px] hover:bg-primary/10">
              <Download className="mr-1 h-3 w-3" /> PDF
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <span>Confidence</span>
            <div className="flex items-center gap-2">
              <div className="relative h-1.5 w-16 overflow-hidden rounded-full bg-muted">
                <div className="absolute inset-y-0 left-0 rounded-full bg-primary transition-all duration-500" style={{ width: `${Math.min(doc.score * 100, 100)}%` }} />
              </div>
              <span className="font-mono text-[11px] text-primary/80">{(doc.score * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      </div>
      <div className="relative text-sm leading-relaxed text-foreground/80">
        {doc.content.length > 200 ? (
          <>
            <p className="transition-all duration-300">{isExpanded ? doc.content : `${doc.content.substring(0, 200)}...`}</p>
            <Button variant="ghost" size="sm" onClick={() => setIsExpanded(!isExpanded)} className="mt-3 h-7 px-2 text-[11px] text-primary hover:text-primary/80 hover:bg-primary/10">
              {isExpanded ? <><ChevronUp className="mr-1 h-3 w-3" /> Show less</> : <><ChevronDown className="mr-1 h-3 w-3" /> Read more</>}
            </Button>
          </>
        ) : (<p>{doc.content}</p>)}
      </div>
    </div>
  );
};

function App() {
  const [inputMessage, setInputMessage] = useState("");
  const [activeChat, setActiveChat] = useState<string | null>(null);
  const [currentMessages, setCurrentMessages] = useState<Message[]>([]);
  const [currentSourceDocuments, setCurrentSourceDocuments] = useState<SourceDocument[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const { queryHistory, isLoading: isLoadingHistory, isError: historyError, mutate: refreshHistory } = useQueryHistory(20, 0);
  const { query: selectedQuery, isLoading: isLoadingQuery } = useQuery(activeChat);
  const { sendQuery, isLoading: isSendingMessage, error: sendError } = useQueryRAG();

  const chatList = queryHistory ? convertQueryHistoryToChats(queryHistory) : [];
  const activeChatMeta = activeChat ? chatList.find((chat) => chat.id === activeChat) : null;

  const suggestionPills = [
    "Does garlic cure the flu?",
    "Is 'alkaline water' scientifically proven?",
    "Vaccine efficacy data for new variants",
    "Side effects of long-term ibuprofen use"
  ];
  
  const showWelcomeState = currentMessages.length === 0 && !isSendingMessage && !isLoadingQuery;
  const showHistoryEmptyState = chatList.length === 0 && !isLoadingHistory && !historyError;

  const formatTimestamp = (timestamp: string) => new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  useEffect(() => {
    if (selectedQuery) {
      const messages: Message[] = [
        { id: `${selectedQuery.query_history.id}-user`, type: "user", content: selectedQuery.query_history.query, timestamp: selectedQuery.query_history.created_at },
        {
          id: `${selectedQuery.query_history.id}-assistant`, type: "assistant", content: selectedQuery.query_history.chat_response, timestamp: selectedQuery.query_history.created_at,
          source_documents: selectedQuery.source_documents.map((doc) => ({
            content: doc.content_preview, score: doc.similarity_score, metadata: doc.document_metadata || { file_name: "Unknown", page: undefined, source: undefined },
          })),
        },
      ];
      setCurrentMessages(messages);
      setCurrentSourceDocuments(selectedQuery.source_documents.map((doc) => ({
        content: doc.content_preview, score: doc.similarity_score, metadata: doc.document_metadata || { file_name: "Unknown", page: undefined, source: undefined },
      })));
    } else {
      setCurrentMessages([]);
      setCurrentSourceDocuments([]);
    }
  }, [selectedQuery]);

  const handleSuggestionClick = (value: string) => setInputMessage(value);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isSendingMessage) return;
    const queryText = inputMessage.trim();
    setInputMessage("");

    try {
      const tempUserMessage: Message = { id: `temp-${Date.now()}-user`, type: "user", content: queryText, timestamp: new Date().toISOString() };
      setCurrentMessages((prev) => [...prev, tempUserMessage]);
      setActiveChat(null);

      const response = await sendQuery({ query: queryText, top_k: 2 });
      if (response) {
        const newMessages = convertQueryResponseToMessages(queryText, response, `query-${Date.now()}`);
        setCurrentMessages(newMessages);
        setCurrentSourceDocuments(response.source_documents);
        refreshHistory();
      } else {
        setCurrentMessages((prev) => prev.filter((msg) => msg.id !== tempUserMessage.id));
      }
    } catch (error) {
      console.error("Error sending message:", error);
      setCurrentMessages((prev) => prev.filter((msg) => msg.id.startsWith("temp-")));
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleNewChat = () => { setActiveChat(null); setCurrentMessages([]); setCurrentSourceDocuments([]); };
  const handleSelectChat = (chatId: string) => setActiveChat(chatId);

  return (
    <div className="relative min-h-screen overflow-hidden bg-background text-foreground">
      {/* Background Ambience */}
      <div aria-hidden="true" className="pointer-events-none absolute inset-0">
        <div className="absolute -left-24 -top-32 h-96 w-96 rounded-full bg-blue-100/50 blur-3xl opacity-30" />
        <div className="absolute right-[-6rem] top-1/3 h-80 w-80 rounded-full bg-cyan-100/50 blur-3xl opacity-30" />
      </div>

      <div className="relative z-10 mx-auto flex min-h-screen w-full max-w-[1600px] flex-col gap-6 px-4 py-6 lg:flex-row lg:px-12 lg:py-10">
        
        {/* Main Chat Area */}
        <main className="order-1 flex-1 overflow-hidden rounded-3xl border border-border bg-card/80 shadow-xl backdrop-blur-sm">
          <div className="flex h-full flex-col">
            <div className="border-b border-border/50 bg-card/50 px-6 py-6 md:px-10 md:py-8">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      className="h-8 w-8 text-muted-foreground hover:text-primary" 
                      onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                      title={isSidebarOpen ? "Collapse Menu" : "Expand Menu"}
                    >
                      <PanelLeft className="h-5 w-5" />
                    </Button>
                    <span className="inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-3 py-1 text-[11px] font-semibold uppercase tracking-wide text-primary">
                      <ShieldCheck className="h-3.5 w-3.5" /> Trusted Source Verification
                    </span>
                  </div>
                  <div>
                    <h2 className="text-2xl font-semibold tracking-tight md:text-3xl">{activeChatMeta ? activeChatMeta.name : "New Fact Check"}</h2>
                    <p className="mt-2 max-w-xl text-sm text-muted-foreground">{activeChatMeta ? "Reviewing analysis results." : "Input a medical claim or health rumor to verify it against trusted clinical journals."}</p>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  {activeChatMeta && <Badge className="rounded-full bg-green-100 text-[11px] font-medium text-green-700 hover:bg-green-200">Verified {new Date(activeChatMeta.created_at).toLocaleDateString()}</Badge>}
                </div>
              </div>
              {showWelcomeState && (
                <div className="mt-8 rounded-3xl border border-border/60 bg-white/50 p-8 text-center backdrop-blur md:p-10">
                  <div className="mx-auto max-w-2xl space-y-6">
                    <div className="inline-flex items-center gap-2 rounded-full border border-accent/40 bg-accent/20 px-4 py-1 text-[11px] font-semibold uppercase tracking-wide text-accent-foreground">
                      <Microscope className="h-4 w-4" /> Evidence-Based AI
                    </div>
                    <div className="space-y-3">
                      <h3 className="text-2xl font-semibold md:text-3xl">Combat Misinformation with Science</h3>
                      <p className="text-sm text-muted-foreground md:text-base">Enter any health claim, social media rumor, or treatment query. We analyze thousands of peer-reviewed papers to give you the facts.</p>
                    </div>
                    <div className="flex flex-wrap justify-center gap-3">
                      {suggestionPills.map((suggestion) => (
                        <Button key={suggestion} type="button" variant="outline" size="sm" onClick={() => handleSuggestionClick(suggestion)} className="rounded-full border-border/60 bg-white/80 text-xs text-muted-foreground transition hover:border-primary/50 hover:bg-primary/5 hover:text-foreground">
                          {suggestion}
                        </Button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="flex-1 overflow-hidden">
              <ScrollArea className="h-full px-6 py-6 md:px-10 md:py-8">
                <div className="mx-auto flex max-w-4xl flex-col gap-8">
                  <div className="flex flex-col gap-6">
                    {isLoadingQuery && activeChat ? (
                      <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-border/60 bg-white/50 p-10 text-center text-sm text-muted-foreground"><Loader2 className="h-6 w-6 animate-spin text-primary" /> Retrieving analysis...</div>
                    ) : currentMessages.length === 0 ? (
                      !showWelcomeState && <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-border/60 bg-white/50 p-10 text-center text-sm text-muted-foreground"><Activity className="h-10 w-10 text-muted-foreground/50" /> Awaiting claim input...</div>
                    ) : (
                      currentMessages.map((message) => (
                        <div key={message.id} className={`flex items-end gap-3 ${message.type === "user" ? "flex-row-reverse" : ""}`}>
                          <Avatar className="h-9 w-9 border border-border bg-white text-primary"><AvatarFallback>{message.type === "assistant" ? <ShieldCheck className="h-4 w-4" /> : <User className="h-4 w-4" />}</AvatarFallback></Avatar>
                          <div className={`flex max-w-[78%] flex-col gap-2 ${message.type === "user" ? "items-end text-right" : ""}`}>
                            <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground/80">{message.type === "assistant" ? "Medical Fact Checker" : "Claim"}</div>
                            <div className={`rounded-2xl border px-4 py-3 text-sm leading-relaxed shadow-sm transition ${message.type === "user" ? "border-primary/20 bg-primary text-primary-foreground shadow-md" : "border-border/60 bg-white/80 backdrop-blur"}`}>
                              <p className="whitespace-pre-wrap text-sm md:text-[15px]">{message.content}</p>
                            </div>
                            <span className="text-[11px] text-muted-foreground/80">{formatTimestamp(message.timestamp)}</span>
                          </div>
                        </div>
                      ))
                    )}
                    {isSendingMessage && (
                      <div className="flex items-end gap-3">
                        <Avatar className="h-9 w-9 border border-border bg-white text-primary"><AvatarFallback><ShieldCheck className="h-4 w-4" /></AvatarFallback></Avatar>
                        <div className="flex max-w-[78%] flex-col gap-2"><div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground/80">Medical Fact Checker</div><div className="rounded-2xl border border-border/60 bg-white/80 px-4 py-3 backdrop-blur"><div className="flex items-center gap-2 text-sm"><Loader2 className="h-4 w-4 animate-spin text-primary" /> Analyzing clinical databases...</div></div></div>
                      </div>
                    )}
                  </div>
                  <div>
                    <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-foreground"><Microscope className="h-5 w-5 text-primary" /> Clinical Evidence</div>
                    {currentSourceDocuments.length === 0 ? (
                      <div className="rounded-2xl border border-dashed border-border/70 bg-white/40 p-8 text-center text-sm text-muted-foreground">{currentMessages.length === 0 ? "Relevant medical papers will appear here." : "No specific medical papers found."}</div>
                    ) : (
                      <div className="grid gap-4 lg:grid-cols-2">{currentSourceDocuments.map((doc, index) => <SourceDocumentCard key={index} doc={doc} index={index} />)}</div>
                    )}
                  </div>
                </div>
              </ScrollArea>
            </div>

            <div className="border-t border-border/60 bg-card/70 px-6 py-6 md:px-10">
              <div className="mx-auto flex max-w-4xl flex-col gap-4">
                {sendError && (
                  <div className="rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-600"><div className="flex items-center gap-2 font-medium"><AlertCircle className="h-4 w-4" /> Error analyzing claim</div><p className="mt-1 text-xs opacity-80">{sendError}</p></div>
                )}
                <div className="flex flex-col gap-3 sm:flex-row">
                  <div className="relative flex-1">
                    <Input placeholder="E.g., Does ginger tea help with nausea?" value={inputMessage} onChange={(e) => setInputMessage(e.target.value)} onKeyDown={handleKeyDown} className="h-12 rounded-2xl border-border bg-white pr-16 text-sm shadow-sm transition-all focus:border-primary/50 focus:shadow-md" disabled={isSendingMessage} />
                    <div className="pointer-events-none absolute inset-y-0 right-4 hidden items-center gap-2 text-[11px] text-muted-foreground sm:flex"><span>Press</span><kbd className="rounded-md border border-border bg-gray-50 px-2 py-1 text-[10px] uppercase">Enter</kbd></div>
                  </div>
                  <Button onClick={handleSendMessage} className="h-12 rounded-2xl bg-primary px-6 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition hover:bg-primary/90" disabled={!inputMessage.trim() || isSendingMessage}>
                    {isSendingMessage ? <Loader2 className="h-4 w-4 animate-spin" /> : <><Send className="mr-2 h-4 w-4" /> Analyze</>}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </main>

        {/* Collapsible Sidebar */}
        {isSidebarOpen && (
          <aside className="order-2 flex w-full animate-in slide-in-from-right-10 fade-in duration-300 flex-col gap-6 rounded-3xl border border-border bg-card/80 p-6 shadow-xl backdrop-blur-sm lg:order-none lg:w-[320px]">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h1 className="text-2xl font-semibold leading-tight tracking-tight text-primary">MedFact Check</h1>
                <p className="text-xs text-muted-foreground">Clinical verification engine</p>
              </div>
              <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setIsSidebarOpen(false)}>
                <X className="h-4 w-4" />
              </Button>
            </div>
            
            <Button className="h-11 w-full rounded-2xl bg-primary px-6 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/20 transition hover:bg-primary/90" onClick={handleNewChat}>
              <Plus className="mr-2 h-4 w-4" /> Check New Claim
            </Button>

            <div className="rounded-2xl border border-border/60 bg-white/50 p-5">
              <div className="flex items-center gap-2 text-sm font-semibold text-foreground"><Activity className="h-4 w-4 text-primary" /> Trending Rumors</div>
              <ul className="mt-3 space-y-2 text-xs text-muted-foreground">
                {suggestionPills.map((suggestion) => (
                  <li key={`tip-${suggestion}`}><button type="button" onClick={() => handleSuggestionClick(suggestion)} className="flex w-full items-center gap-2 rounded-xl border border-transparent bg-transparent px-2 py-1 text-left transition hover:border-primary/40 hover:bg-primary/5 hover:text-foreground"><span className="mt-0.5 h-1.5 w-1.5 rounded-full bg-primary/60" />{suggestion}</button></li>
                ))}
              </ul>
            </div>
            
            <div className="flex-1 overflow-hidden">
              <div className="mb-3 flex items-center justify-between text-sm font-semibold text-muted-foreground"><div className="flex items-center gap-2 text-foreground"><History className="h-4 w-4 text-primary" /> Recent Checks</div><span className="text-[11px] uppercase tracking-wide text-muted-foreground/80">{chatList.length} saved</span></div>
              <ScrollArea className="h-full pr-1">
                <div className="space-y-2">
                  {isLoadingHistory ? (
                    <div className="flex items-center justify-center rounded-2xl border border-border/50 bg-white/50 p-6 text-sm text-muted-foreground"><Loader2 className="mr-2 h-4 w-4 animate-spin text-primary" /> Loading history...</div>
                  ) : historyError ? (
                    <div className="flex items-center justify-center rounded-2xl border border-red-500/40 bg-red-500/10 p-6 text-sm text-red-600"><AlertCircle className="mr-2 h-4 w-4" /> Failed to load history</div>
                  ) : showHistoryEmptyState ? (
                    <div className="rounded-2xl border border-dashed border-border/60 bg-white/40 p-6 text-center text-sm text-muted-foreground">No history yet.</div>
                  ) : (
                    chatList.map((chat) => (
                      <button key={chat.id} onClick={() => handleSelectChat(chat.id)} className={`w-full rounded-2xl border px-4 py-4 text-left transition-all ${activeChat === chat.id ? "border-primary/40 bg-primary/5 text-foreground shadow-sm" : "border-transparent bg-white/40 text-muted-foreground hover:border-primary/20 hover:bg-white/80 hover:text-foreground"}`}>
                        <div className="flex items-start justify-between gap-3"><div className="min-w-0"><p className="truncate text-sm font-semibold leading-tight">{chat.name}</p><p className="mt-1 text-xs text-muted-foreground">{new Date(chat.created_at).toLocaleDateString()}</p></div>{!chat.success && <Badge variant="outline" className="rounded-full border-red-400/60 bg-transparent text-[11px] text-red-400">Error</Badge>}</div>
                      </button>
                    ))
                  )}
                </div>
              </ScrollArea>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}

export default App;