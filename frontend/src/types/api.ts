export interface DocumentMetadata {
  file_name: string;
  page?: number;
  source?: string;
}

// Used for live RAG responses
export interface SourceDocument {
  content: string;
  score: number;
  metadata: DocumentMetadata;
}

// Used for Historical responses (Database structure)
export interface SourceDocumentHistoryResponse {
  id: string;
  content_preview: string;
  similarity_score: number;
  document_metadata: DocumentMetadata | null;
  created_at: string;
}

export interface QueryRequest {
  query: string;
  top_k?: number; // <--- This fixes the 'top_k' error
}

export interface QueryResponse {
  chat_response: string;
  source_documents: SourceDocument[];
}

export interface QueryHistoryResponse {
  id: string;
  query: string;
  chat_response: string;
  top_k: number;
  response_time_ms: number | null;
  source_document_count: number;
  created_at: string;
  success: boolean;
  error_message: string | null;
}

export interface QueryHistoryListResponse {
  items: QueryHistoryResponse[];
  total_count: number;
  limit: number;
  offset: number;
}

export interface QueryDetailResponse {
  query_history: QueryHistoryResponse;
  source_documents: SourceDocumentHistoryResponse[]; // <--- explicitly typed array
}

// Frontend-specific types
export interface Message {
  id: string;
  type: "user" | "assistant";
  content: string;
  timestamp: string;
  source_documents?: SourceDocument[];
}