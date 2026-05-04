// api.ts — Typed client for all ResearchRAG backend endpoints.
// Every function throws an Error with a descriptive message on failure.

const BASE = '/api'

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      detail = body.detail || body.message || JSON.stringify(body)
    } catch { /* ignore */ }
    throw new Error(detail)
  }
  return res.json()
}

// ── Types ────────────────────────────────────────────────────────────────────

export type PipelineStage =
  | 'pending' | 'downloading' | 'downloaded'
  | 'processing' | 'processed'
  | 'failed_download' | 'failed_processing'

export type ChatMode = 'standard' | 'study' | 'technical'

export interface PaperPreview {
  paper_id:      string
  source:        string
  title:         string
  abstract:      string
  authors:       string[]
  url:           string
  has_full_text: boolean
  published:     string
  journal:       string
  categories:    string[]
  doi:           string
}

export interface Paper {
  paper_id:       string
  source:         string
  title:          string | null
  abstract:       string | null
  authors:        string[]
  url:            string | null
  pipeline_stage: PipelineStage
  chunk_count:    number
  error_message:  string | null
  topic:          string | null
  created_at:     string | null
  processed_at:   string | null
}

export interface Session {
  session_id:     string
  paper_id:       string
  topic:          string | null
  level:          string
  mode:           ChatMode
  title:          string | null
  last_active_at: string | null
  created_at:     string | null
}

export interface Message {
  id:         number
  role:       'user' | 'assistant'
  content:    string
  level:      string
  created_at: string | null
}

export interface SessionDetail extends Session {
  messages: Message[]
}

export interface SocialItem {
  id:           number
  platform:     string
  content_type: string
  content:      string
  hashtags:     string[]
  created_at:   string | null
}

export interface ModelOption {
  id:          string
  name:        string
  provider:    string
  description: string
  recommended: boolean
  active:      boolean
}

export interface ModelsResponse {
  provider:     string
  active_model: string
  models:       ModelOption[]
}

// ── Papers ───────────────────────────────────────────────────────────────────

export interface SearchParams {
  topic:      string
  limit:      number
  source:     'arxiv' | 'pubmed' | 'both'
  sort_by?:   'date' | 'relevance'
  date_from?: string
  date_to?:   string
  category?:  string
  keyword?:   string
}

export async function searchPapers(
  params: SearchParams
): Promise<{ papers: PaperPreview[]; message: string }> {
  return request('/papers/search', { method: 'POST', body: JSON.stringify(params) })
}

export async function processPapers(
  paperIds: string[]
): Promise<{ papers: Paper[]; message: string }> {
  return request('/papers/process', {
    method: 'POST',
    body: JSON.stringify({ paper_ids: paperIds }),
  })
}

export async function uploadPaper(
  file: File,
  topic: string
): Promise<{ paper: Paper; message: string }> {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('topic', topic)
  const res = await fetch(`${BASE}/papers/upload`, { method: 'POST', body: fd })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try { const b = await res.json(); detail = b.detail || detail } catch { /* ignore */ }
    throw new Error(detail)
  }
  return res.json()
}

export async function listPapers(params?: {
  stage?: string; source?: string; limit?: number; offset?: number
}): Promise<{ papers: Paper[] }> {
  const q = new URLSearchParams()
  if (params?.stage)  q.set('stage',  params.stage)
  if (params?.source) q.set('source', params.source)
  if (params?.limit)  q.set('limit',  String(params.limit))
  if (params?.offset) q.set('offset', String(params.offset))
  return request(`/papers${q.toString() ? '?' + q : ''}`)
}

export async function deletePaper(
  paperId: string
): Promise<{ success: boolean; message: string }> {
  return request(`/papers/${paperId}`, { method: 'DELETE' })
}

// ── Chat ─────────────────────────────────────────────────────────────────────

export async function listSessions(): Promise<{ sessions: Session[] }> {
  return request('/chat/sessions')
}

export async function createSession(params: {
  paper_id: string
  topic?:   string
  level?:   string
  mode:     ChatMode
}): Promise<{ session: SessionDetail }> {
  return request('/chat/sessions', {
    method: 'POST',
    body: JSON.stringify(params),
  })
}

export async function getSession(
  sessionId: string
): Promise<{ session: SessionDetail }> {
  return request(`/chat/sessions/${sessionId}`)
}

export async function sendMessage(
  sessionId: string,
  message: string,
  level: string
): Promise<{ session_id: string; response: string; level: string }> {
  return request(`/chat/sessions/${sessionId}/message`, {
    method: 'POST',
    body: JSON.stringify({ message, level }),
  })
}

// ── Study cache ───────────────────────────────────────────────────────────────

export interface StudyCacheStatus {
  paper_id:   string
  outline:    { content: { summary: string; sections: { index: number; title: string; description: string }[] }; created_at: string } | null
  sections:   { section_title: string; content: string; level: string | null; created_at: string }[]
  flashcards: { cards: { front: string; back: string }[]; created_at: string } | null
}

export async function getStudyCacheStatus(paperId: string): Promise<StudyCacheStatus> {
  return request(`/study/${paperId}/cache`)
}

export async function bustStudyCache(paperId: string): Promise<{ success: boolean; deleted: number }> {
  return request(`/study/${paperId}/cache`, { method: 'DELETE' })
}

// ── Technical cache ───────────────────────────────────────────────────────────

export interface TechnicalCachedSection {
  section_key:   string
  section_label: string
  content:       string
}

export type TechnicalAnalyzeResponse =
  {
    cached:     true
    paper_id:   string
    sections:   TechnicalCachedSection[]
  }
 | {
    cached:     false
    queue_key:  string
    paper_id:   string
    sections:   { key: string; label: string }[]
    message:    string
  }

export async function bustTechnicalCache(paperId: string): Promise<{ success: boolean; deleted: number }> {
  return request(`/technical/${paperId}/cache`, { method: 'DELETE' })
}

export async function updateLevel(
  sessionId: string,
  level: string
): Promise<{ success: boolean }> {
  return request(`/chat/sessions/${sessionId}/level`, {
    method: 'PATCH',
    body: JSON.stringify({ level }),
  })
}

export async function deleteSession(
  sessionId: string
): Promise<{ success: boolean; message: string }> {
  return request(`/chat/sessions/${sessionId}`, { method: 'DELETE' })
}

// ── Models ───────────────────────────────────────────────────────────────────

export async function getModels(): Promise<ModelsResponse> {
  return request('/models')
}

export async function selectModel(
  modelId: string
): Promise<{ success: boolean; active_model: string; message: string }> {
  return request('/models/select', {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId }),
  })
}

// ── Generate ─────────────────────────────────────────────────────────────────

export async function generateContent(params: {
  paper_id: string
  platform: 'twitter' | 'linkedin' | 'carousel'
  description: string
  color_scheme?: string
}): Promise<{ queue_key: string; paper_id: string; platform: string; message: string }> {
  return request('/generate', {
    method: 'POST',
    body: JSON.stringify({ color_scheme: 'light', ...params }),
  })
}

export async function generationHistory(
  paperId: string,
  platform?: string
): Promise<{ paper_id: string; items: SocialItem[] }> {
  const q = platform ? `?platform=${platform}` : ''
  return request(`/generate/history/${paperId}${q}`)
}

export async function exportCarousel(
  contentId: number
): Promise<{ success: boolean; file_path: string; filename: string; download_url: string }> {
  return request(`/generate/${contentId}/export`, { method: 'POST' })
}

export async function getShareLinks(
  contentId: number
): Promise<{ linkedin_url: string; twitter_url: string }> {
  return request(`/generate/${contentId}/share`)
}