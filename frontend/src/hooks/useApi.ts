import { useState } from "react";
import type { 
  Message, 
  QueryHistoryListResponse, 
  QueryResponse, 
  QueryDetailResponse,
  QueryRequest
} from "@/types/api";

// MOCK DATA
const MOCK_HISTORY_ITEM = {
  id: "chat-1",
  query: "What is a strip foul?",
  chat_response: "A strip foul occurs when a defensive player knocks the disc out of the hands of an offensive player...",
  created_at: new Date().toISOString(),
  success: true,
  top_k: 2,
  response_time_ms: 120,
  source_document_count: 2,
  error_message: null
};

const MOCK_HISTORY_LIST: QueryHistoryListResponse = {
  items: [MOCK_HISTORY_ITEM],
  total_count: 1,
  limit: 20,
  offset: 0
};

export const useQueryHistory = (limit = 10, offset = 0) => {
  return {
    queryHistory: MOCK_HISTORY_LIST,
    isLoading: false,
    isError: null,
    mutate: () => {},
  };
};

export const useQuery = (queryId: string | null) => {
  // Explicitly cast this object to QueryDetailResponse
  const mockData: QueryDetailResponse | null = queryId ? {
    query_history: MOCK_HISTORY_ITEM,
    source_documents: [
      {
        id: "doc-1",
        content_preview: "Section 12.3: Strip Fouls. No defensive player may touch the disc...",
        similarity_score: 0.89,
        document_metadata: { file_name: "WFDF_Rules_2021.pdf", page: 12 },
        created_at: new Date().toISOString()
      }
    ]
  } : null;

  return {
    query: mockData,
    isLoading: false,
    isError: null,
  };
};

export const useQueryRAG = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Update signature to accept QueryRequest (which includes top_k)
  const sendQuery = async (request: QueryRequest) => {
    setIsLoading(true);
    setError(null);
    
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    setIsLoading(false);
    
    const response: QueryResponse = {
      chat_response: `This is a simulated AI response to: "${request.query}". \n\nIn a real app, this would come from the backend RAG system.`,
      source_documents: [
        {
          content: "Section 12.3: Strip Fouls. No defensive player may touch the disc while it is in possession of the thrower.",
          score: 0.89,
          metadata: { file_name: "WFDF_Rules_2021.pdf", page: 12 }
        }
      ]
    };
    return response;
  };

  return { sendQuery, isLoading, error };
};

export const convertQueryResponseToMessages = (
  userQuery: string,
  response: QueryResponse,
  baseMessageId: string
): Message[] => {
  const timestamp = new Date().toISOString();
  return [
    { id: `${baseMessageId}-user`, type: "user", content: userQuery, timestamp },
    { id: `${baseMessageId}-assistant`, type: "assistant", content: response.chat_response, timestamp, source_documents: response.source_documents },
  ];
};

export const convertQueryHistoryToChats = (queryHistory: QueryHistoryListResponse) => {
  return queryHistory.items.map((item) => ({
    id: item.id,
    name: item.query,
    created_at: item.created_at,
    success: item.success,
  }));
};