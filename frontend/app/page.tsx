'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import {
  searchPapers, processPapers, uploadPaper, listSessions, getSession, listPapers,
  sendMessage, updateLevel, generateContent, generationHistory,
  exportCarousel, getShareLinks, getModels, selectModel,
  type Paper, type PaperPreview, type Session, type SessionDetail,
  type Message, type SocialItem, type ModelOption,
} from './lib/api'

// ── Types ────────────────────────────────────────────────────────────────────

type View = 'search' | 'selection' | 'processing' | 'chat'
type Level = 'beginner' | 'intermediate' | 'advanced'
type Platform = 'twitter' | 'linkedin' | 'carousel'

interface PaperStatus extends Paper {
  sseStage?: string
  sseMessage?: string
}

const BACKEND = 'http://localhost:8000'

// ── arXiv category options (full names → codes) ───────────────────────────────
const ARXIV_CATEGORIES = [
  { label: 'All Categories',                   value: '' },
  { label: '── Computer Science ──',           value: '', disabled: true },
  { label: 'Artificial Intelligence',          value: 'cs.AI' },
  { label: 'Machine Learning',                 value: 'cs.LG' },
  { label: 'Computation and Language (NLP)',   value: 'cs.CL' },
  { label: 'Computer Vision',                  value: 'cs.CV' },
  { label: 'Robotics',                         value: 'cs.RO' },
  { label: 'Human-Computer Interaction',       value: 'cs.HC' },
  { label: 'Information Retrieval',            value: 'cs.IR' },
  { label: 'Neural and Evolutionary Computing',value: 'cs.NE' },
  { label: '── Biology & Medicine ──',         value: '', disabled: true },
  { label: 'Biomolecules',                     value: 'q-bio.BM' },
  { label: 'Genomics',                         value: 'q-bio.GN' },
  { label: 'Neurons and Cognition',            value: 'q-bio.NC' },
  { label: 'Quantitative Methods',             value: 'q-bio.QM' },
  { label: '── Statistics & Mathematics ──',   value: '', disabled: true },
  { label: 'Machine Learning (Statistics)',    value: 'stat.ML' },
  { label: 'Statistics Theory',               value: 'stat.TH' },
  { label: 'Mathematics General',             value: 'math.GM' },
  { label: '── Physics & Engineering ──',      value: '', disabled: true },
  { label: 'Physics General',                 value: 'physics.gen-ph' },
  { label: 'Electrical Engineering',          value: 'eess.SP' },
  { label: 'Systems and Control',             value: 'eess.SY' },
  { label: '── Economics ──',                  value: '', disabled: true },
  { label: 'Economics',                       value: 'econ.GN' },
]

// ── Helpers ──────────────────────────────────────────────────────────────────

function stageLabel(stage: string): string {
  const map: Record<string, string> = {
    pending: 'Pending', downloading: 'Downloading…', downloaded: 'Downloaded',
    processing: 'Processing…', processed: 'Ready',
    failed_download: 'Download failed', failed_processing: 'Processing failed',
  }
  return map[stage] ?? stage
}

function stageClass(stage: string): string {
  if (stage === 'processed') return 'badge-done'
  if (stage.startsWith('failed')) return 'badge-error'
  if (stage === 'pending') return 'badge-pending'
  return 'badge-active'
}

function isTerminal(stage: string) {
  return stage === 'processed' || stage.startsWith('failed')
}

function truncateAbstract(text: string, sentences = 2): string {
  if (!text) return ''
  const parts = text.match(/[^.!?]+[.!?]+/g) || []
  return parts.slice(0, sentences).join(' ').trim() || text.slice(0, 200)
}

// ── Spinner ──────────────────────────────────────────────────────────────────

function Spinner({ size = '' }: { size?: string }) {
  return <span className={`spinner ${size}`} />
}

// ── Icons ────────────────────────────────────────────────────────────────────

function IconSearch() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
    </svg>
  )
}
function IconUpload() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  )
}
function IconSend() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
    </svg>
  )
}
function IconShare() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/>
    </svg>
  )
}
function IconDownload() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
  )
}
function IconChevron({ open }: { open: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
      style={{ transition: 'transform 0.2s', transform: open ? 'rotate(180deg)' : 'rotate(0deg)' }}>
      <polyline points="6 9 12 15 18 9"/>
    </svg>
  )
}
function IconSparkle() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5z"/>
    </svg>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// VIEW 1: Search / Upload
// ─────────────────────────────────────────────────────────────────────────────

function SearchView({ onResults }: {
  onResults: (papers: PaperPreview[]) => void
}) {
  const [topic, setTopic]       = useState('')
  const [source, setSource]     = useState<'both' | 'arxiv' | 'pubmed'>('both')
  const [limit, setLimit]       = useState(10)
  const [file, setFile]         = useState<File | null>(null)
  const [mode, setMode]         = useState<'search' | 'upload'>('search')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  // Refine filters
  const [showRefine, setShowRefine] = useState(false)
  const [sortBy, setSortBy]         = useState<'date' | 'relevance'>('date')
  const [dateFrom, setDateFrom]     = useState('')
  const [dateTo, setDateTo]         = useState('')
  const [category, setCategory]     = useState('')
  const [keyword, setKeyword]       = useState('')

  // Active filter summary for collapsed state
  const activeFilters = [
    sortBy !== 'date' ? 'By relevance' : '',
    category ? ARXIV_CATEGORIES.find(c => c.value === category)?.label ?? '' : '',
    dateFrom && dateTo ? `${dateFrom} – ${dateTo}` : dateFrom ? `From ${dateFrom}` : dateTo ? `To ${dateTo}` : '',
    keyword ? `"${keyword}"` : '',
  ].filter(Boolean)

  // Model picker
  const [models, setModels]               = useState<ModelOption[]>([])
  const [activeModel, setActiveModel]     = useState('')
  const [modelSwitching, setModelSwitching] = useState(false)
  const [isOpenRouter, setIsOpenRouter]   = useState(false)

  useEffect(() => {
    getModels().then(res => {
      if (res.provider === 'openrouter') {
        setIsOpenRouter(true); setModels(res.models); setActiveModel(res.active_model)
      }
    }).catch(() => {})
  }, [])

  async function handleModelSelect(modelId: string) {
    if (modelId === activeModel) return
    setModelSwitching(true)
    try { await selectModel(modelId); setActiveModel(modelId) }
    catch (err: unknown) { setError(err instanceof Error ? err.message : 'Failed to switch model.') }
    finally { setModelSwitching(false) }
  }

  const topicError = topic.trim().length === 0 && topic.length > 0
    ? 'Topic cannot be empty.'
    : topic.length > 200 ? 'Topic must be 200 characters or fewer.' : ''

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault()
    if (!topic.trim()) { setError('Please enter a topic.'); return }
    setError(''); setLoading(true)
    try {
      const result = await searchPapers({
        topic: topic.trim(), limit, source, sort_by: sortBy,
        ...(dateFrom ? { date_from: dateFrom } : {}),
        ...(dateTo   ? { date_to: dateTo }     : {}),
        ...(category ? { category }            : {}),
        ...(keyword  ? { keyword }             : {}),
      })
      onResults(result.papers)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Search failed.')
    } finally { setLoading(false) }
  }

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault()
    if (!file) { setError('Please select a PDF file.'); return }
    if (!topic.trim()) { setError('Please enter a topic.'); return }
    setError(''); setLoading(true)
    try {
      // Upload goes directly to processing — no selection step needed
      const result = await uploadPaper(file, topic.trim())
      // Wrap as PaperPreview so App can route to processing directly
      onResults([{
        paper_id: result.paper.paper_id, source: 'local',
        title: result.paper.title ?? file.name, abstract: result.paper.abstract ?? '',
        authors: result.paper.authors ?? [], url: result.paper.url ?? '',
        has_pdf: true, published: '', journal: '', categories: [], doi: '',
      }])
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Upload failed.')
    } finally { setLoading(false) }
  }

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (!f) return
    if (!f.name.endsWith('.pdf')) { setError('Only PDF files are accepted.'); return }
    if (f.size > 50 * 1024 * 1024) { setError('File must be under 50 MB.'); return }
    setError(''); setFile(f)
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '24px' }}>
      <div style={{ width: '100%', maxWidth: '560px' }} className="fade-in">

        <div style={{ marginBottom: '32px', textAlign: 'center' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent)', display: 'inline-block' }} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--accent)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>ResearchRAG</span>
          </div>
          <h1 style={{ fontSize: '1.75rem', fontWeight: 600, color: 'var(--text)', marginBottom: '8px' }}>Research, understood.</h1>
          <p style={{ color: 'var(--text-2)', fontSize: '0.9375rem' }}>Search papers, review them, then choose which ones to explore.</p>
        </div>

        {/* Model picker */}
        {isOpenRouter && models.length > 0 && (
          <div className="card" style={{ padding: '16px', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
              <p style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>AI Model</p>
              {modelSwitching && <Spinner size="spinner-sm" />}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {models.map(m => (
                <button key={m.id} onClick={() => handleModelSelect(m.id)} disabled={modelSwitching}
                  style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '12px', padding: '10px 12px', borderRadius: 'var(--radius)', border: `1px solid ${activeModel === m.id ? 'var(--accent)' : 'var(--border)'}`, background: activeModel === m.id ? 'var(--accent-glow)' : 'var(--bg)', cursor: modelSwitching ? 'not-allowed' : 'pointer', textAlign: 'left', transition: 'all 0.15s', opacity: modelSwitching ? 0.6 : 1 }}>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '2px' }}>
                      <span style={{ fontSize: '0.875rem', fontWeight: 500, color: activeModel === m.id ? 'var(--accent)' : 'var(--text)' }}>{m.name}</span>
                      {m.recommended && <span style={{ fontSize: '0.6875rem', background: 'var(--accent-glow)', color: 'var(--accent)', padding: '1px 6px', borderRadius: '10px', fontWeight: 500 }}>recommended</span>}
                    </div>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-3)', lineHeight: 1.4 }}>{m.description}</p>
                  </div>
                  {activeModel === m.id && <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--accent)', flexShrink: 0, marginTop: '4px' }} />}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Mode toggle */}
        <div style={{ display: 'flex', background: 'var(--bg-2)', borderRadius: 'var(--radius)', padding: '3px', marginBottom: '20px', border: '1px solid var(--border)' }}>
          {(['search', 'upload'] as const).map(m => (
            <button key={m} onClick={() => { setMode(m); setError('') }}
              style={{ flex: 1, padding: '7px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: '0.875rem', fontWeight: 500, transition: 'all 0.15s', background: mode === m ? 'var(--bg-3)' : 'transparent', color: mode === m ? 'var(--text)' : 'var(--text-2)' }}>
              {m === 'search' ? 'Search papers' : 'Upload PDF'}
            </button>
          ))}
        </div>

        <div className="card" style={{ padding: '24px' }}>
          <form onSubmit={mode === 'search' ? handleSearch : handleUpload}>

            <div style={{ marginBottom: '16px' }}>
              <label className="label">{mode === 'search' ? 'Topic or keywords' : 'Topic label'}</label>
              <input className={`input ${topicError ? 'error' : ''}`} value={topic} onChange={e => { setTopic(e.target.value); setError('') }} placeholder={mode === 'search' ? 'e.g. large language models, RAG' : 'e.g. transformer architecture'} maxLength={200} />
              {topicError && <p style={{ color: 'var(--error)', fontSize: '0.8125rem', marginTop: '4px' }}>{topicError}</p>}
            </div>

            {mode === 'search' && (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '16px' }}>
                  <div>
                    <label className="label">Source</label>
                    <select className="select" value={source} onChange={e => setSource(e.target.value as typeof source)}>
                      <option value="both">arXiv + PubMed</option>
                      <option value="arxiv">arXiv only</option>
                      <option value="pubmed">PubMed only</option>
                    </select>
                  </div>
                  <div>
                    <label className="label">Number of results</label>
                    <select className="select" value={limit} onChange={e => setLimit(Number(e.target.value))}>
                      {[5, 10, 20, 30, 50].map(n => <option key={n} value={n}>{n} papers</option>)}
                    </select>
                  </div>
                </div>

                {/* Refine results — collapsible */}
                <div style={{ marginBottom: '20px', border: '1px solid var(--border)', borderRadius: 'var(--radius)', overflow: 'hidden' }}>
                  <button type="button" onClick={() => setShowRefine(o => !o)}
                    style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: 'var(--bg-3)', border: 'none', cursor: 'pointer', color: 'var(--text-2)', fontFamily: 'var(--font-sans)', fontSize: '0.8125rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontWeight: 500 }}>Refine results</span>
                      {activeFilters.length > 0 && !showRefine && (
                        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--accent)' }}>
                          · {activeFilters.join(' · ')}
                        </span>
                      )}
                    </div>
                    <IconChevron open={showRefine} />
                  </button>

                  {showRefine && (
                    <div style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      {/* Sort by */}
                      <div>
                        <label className="label">Sort by</label>
                        <select className="select" value={sortBy} onChange={e => setSortBy(e.target.value as typeof sortBy)}>
                          <option value="date">Most recent</option>
                          <option value="relevance">Most relevant</option>
                        </select>
                      </div>

                      {/* arXiv category — only shown when source includes arxiv */}
                      {source !== 'pubmed' && (
                        <div>
                          <label className="label">arXiv category</label>
                          <select className="select" value={category} onChange={e => setCategory(e.target.value)}>
                            {ARXIV_CATEGORIES.map((c, i) => (
                              <option key={i} value={c.value} disabled={c.disabled as boolean | undefined}>
                                {c.label}
                              </option>
                            ))}
                          </select>
                        </div>
                      )}

                      {/* Date range */}
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                        <div>
                          <label className="label">From date</label>
                          <input className="input" type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)} />
                        </div>
                        <div>
                          <label className="label">To date</label>
                          <input className="input" type="date" value={dateTo} onChange={e => setDateTo(e.target.value)} />
                        </div>
                      </div>

                      {/* Must-include keyword */}
                      <div>
                        <label className="label">Must include keyword</label>
                        <input className="input" value={keyword} onChange={e => setKeyword(e.target.value)} placeholder="e.g. interpretability, fine-tuning" />
                      </div>

                      {/* Clear filters */}
                      {activeFilters.length > 0 && (
                        <button type="button" className="btn btn-ghost btn-sm"
                          onClick={() => { setSortBy('date'); setCategory(''); setDateFrom(''); setDateTo(''); setKeyword('') }}>
                          Clear filters
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </>
            )}

            {mode === 'upload' && (
              <div style={{ marginBottom: '20px' }}>
                <label className="label">PDF file</label>
                <div onClick={() => fileRef.current?.click()}
                  style={{ border: `2px dashed ${file ? 'var(--accent)' : 'var(--border)'}`, borderRadius: 'var(--radius)', padding: '24px', textAlign: 'center', cursor: 'pointer', background: file ? 'var(--accent-glow)' : 'var(--bg)', transition: 'all 0.15s' }}>
                  <div style={{ color: file ? 'var(--accent)' : 'var(--text-3)', marginBottom: '4px' }}><IconUpload /></div>
                  <p style={{ fontSize: '0.875rem', color: file ? 'var(--text)' : 'var(--text-2)' }}>{file ? file.name : 'Click to select a PDF'}</p>
                  {file && <p style={{ fontSize: '0.75rem', color: 'var(--text-3)', marginTop: '2px' }}>{(file.size / 1024 / 1024).toFixed(1)} MB</p>}
                </div>
                <input ref={fileRef} type="file" accept=".pdf" style={{ display: 'none' }} onChange={handleFile} />
              </div>
            )}

            {error && <div className="notice notice-error" style={{ marginBottom: '16px' }}>{error}</div>}

            <button type="submit" className="btn btn-primary btn-full btn-lg" disabled={loading}>
              {loading ? <><Spinner />{mode === 'search' ? 'Searching…' : 'Uploading…'}</> : <>{mode === 'search' ? <IconSearch /> : <IconUpload />}{mode === 'search' ? 'Search papers' : 'Upload & process'}</>}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// VIEW 2: Paper Selection
// ─────────────────────────────────────────────────────────────────────────────

// Source badge colors
const SOURCE_COLORS: Record<string, { bg: string; color: string; label: string }> = {
  arxiv:  { bg: 'rgba(180,120,255,0.15)', color: '#c084fc', label: 'arXiv' },
  pubmed: { bg: 'rgba(59,130,246,0.15)',  color: '#60a5fa', label: 'PubMed' },
  local:  { bg: 'rgba(16,185,129,0.15)',  color: 'var(--accent)', label: 'Local' },
}

function PaperSelectionCard({
  paper,
  selected,
  onToggle,
}: {
  paper: PaperPreview
  selected: boolean
  onToggle: () => void
}) {
  const [summary, setSummary]       = useState<string | null>(null)
  const [summaryLoading, setSummaryLoading] = useState(false)
  const [showFullAbstract, setShowFullAbstract] = useState(false)
  const srcStyle = SOURCE_COLORS[paper.source] ?? SOURCE_COLORS.local
  const canProcess = paper.has_pdf || paper.source !== 'pubmed'

  // Generate AI summary on mount (staggered by parent)
  useEffect(() => {
    if (!paper.abstract) { setSummary(truncateAbstract(paper.abstract || '', 2)); return }
    setSummaryLoading(true)
    fetch('/api/generate/followup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        context: `Summarize this research paper abstract in 2-3 plain sentences for a non-specialist:\n\n${paper.abstract}`,
      }),
    })
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(data => {
        // The followup endpoint returns questions; we're repurposing it for summaries.
        // The LLM returns a JSON array — take the joined text as the summary.
        const qs: string[] = data.questions ?? []
        if (qs.length > 0) {
          setSummary(qs.join(' '))
        } else {
          setSummary(truncateAbstract(paper.abstract, 2))
        }
      })
      .catch(() => setSummary(truncateAbstract(paper.abstract, 2)))
      .finally(() => setSummaryLoading(false))
  }, [paper.paper_id])

  return (
    <div className="card fade-in" style={{
      marginBottom: '12px',
      border: `1px solid ${selected ? 'var(--accent)' : 'var(--border)'}`,
      background: selected ? 'rgba(16,185,129,0.04)' : 'var(--bg-2)',
      transition: 'all 0.15s',
    }}>
      <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-start' }}>

        {/* Select toggle */}
        <div style={{ flexShrink: 0, paddingTop: '2px' }}>
          <button onClick={onToggle}
            disabled={!canProcess}
            title={!canProcess ? 'Abstract only — chat unavailable' : undefined}
            style={{
              width: '22px', height: '22px', borderRadius: '6px',
              border: `2px solid ${selected ? 'var(--accent)' : 'var(--border)'}`,
              background: selected ? 'var(--accent)' : 'transparent',
              cursor: canProcess ? 'pointer' : 'not-allowed',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              flexShrink: 0, transition: 'all 0.15s',
              opacity: canProcess ? 1 : 0.4,
            }}>
            {selected && (
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M2 6l3 3 5-5" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            )}
          </button>
        </div>

        {/* Content */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* Header row */}
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '6px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.7rem', fontWeight: 600, padding: '2px 8px', borderRadius: '12px', background: srcStyle.bg, color: srcStyle.color, fontFamily: 'var(--font-mono)', flexShrink: 0 }}>
              {srcStyle.label}
            </span>
            {!paper.has_pdf && paper.source === 'pubmed' && (
              <span style={{ fontSize: '0.7rem', padding: '2px 8px', borderRadius: '12px', background: 'var(--warn-bg)', color: 'var(--warn)', fontFamily: 'var(--font-mono)', flexShrink: 0 }}>
                Abstract only
              </span>
            )}
            {paper.published && (
              <span style={{ fontSize: '0.75rem', color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
                {paper.published.slice(0, 10)}
              </span>
            )}
          </div>

          {/* Title */}
          <p style={{ fontWeight: 600, fontSize: '0.9375rem', color: 'var(--text)', lineHeight: 1.4, marginBottom: '6px' }}>
            {paper.title || paper.paper_id}
          </p>

          {/* Authors */}
          {paper.authors.length > 0 && (
            <p style={{ fontSize: '0.8125rem', color: 'var(--text-3)', marginBottom: '8px' }}>
              {paper.authors.slice(0, 3).join(', ')}{paper.authors.length > 3 ? ' et al.' : ''}
            </p>
          )}

          {/* AI summary */}
          <div style={{ fontSize: '0.875rem', color: 'var(--text-2)', lineHeight: 1.6, marginBottom: '8px' }}>
            {summaryLoading ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Spinner size="spinner-sm" />
                <span style={{ color: 'var(--text-3)', fontSize: '0.8125rem' }}>Summarising…</span>
              </div>
            ) : (
              <span>{summary ?? truncateAbstract(paper.abstract, 2)}</span>
            )}
          </div>

          {/* Show original abstract toggle */}
          {paper.abstract && (
            <div>
              <button type="button" onClick={() => setShowFullAbstract(o => !o)}
                style={{ fontSize: '0.8125rem', color: 'var(--accent)', background: 'none', border: 'none', cursor: 'pointer', padding: 0, textDecoration: 'underline' }}>
                {showFullAbstract ? 'Hide abstract' : 'Show original abstract'}
              </button>
              {showFullAbstract && (
                <p style={{ fontSize: '0.8125rem', color: 'var(--text-2)', lineHeight: 1.6, marginTop: '8px', padding: '10px 12px', background: 'var(--bg)', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}>
                  {paper.abstract}
                </p>
              )}
            </div>
          )}

          {/* Journal / DOI */}
          {(paper.journal || paper.doi) && (
            <p style={{ fontSize: '0.75rem', color: 'var(--text-3)', marginTop: '6px', fontFamily: 'var(--font-mono)' }}>
              {paper.journal}{paper.journal && paper.doi ? ' · ' : ''}{paper.doi ? `DOI: ${paper.doi}` : ''}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

function SelectionView({ papers, onSelect, onBack }: {
  papers:   PaperPreview[]
  onSelect: (selected: PaperPreview[]) => void
  onBack:   () => void
}) {
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')

  function toggle(paperId: string) {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(paperId)) next.delete(paperId)
      else next.add(paperId)
      return next
    })
  }

  function selectAll() {
    setSelected(new Set(papers.filter(p => p.has_pdf || p.source !== 'pubmed').map(p => p.paper_id)))
  }

  function deselectAll() { setSelected(new Set()) }

  async function handleProcess() {
    if (selected.size === 0) return
    setError(''); setLoading(true)
    try {
      await processPapers(Array.from(selected))
      onSelect(papers.filter(p => selected.has(p.paper_id)))
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to start processing.')
      setLoading(false)
    }
  }

  const processable = papers.filter(p => p.has_pdf || p.source !== 'pubmed')

  return (
    <div style={{ minHeight: '100vh', padding: '32px 24px' }}>
      <div style={{ maxWidth: '760px', margin: '0 auto' }}>

        {/* Header */}
        <div style={{ marginBottom: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '4px' }}>
                Select papers to process
              </h2>
              <p style={{ color: 'var(--text-3)', fontSize: '0.875rem' }}>
                {papers.length} result{papers.length !== 1 ? 's' : ''} found · {selected.size} selected
              </p>
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
              <button className="btn btn-ghost btn-sm" onClick={onBack}>← New search</button>
              <button className="btn btn-ghost btn-sm" onClick={selected.size === processable.length ? deselectAll : selectAll}>
                {selected.size === processable.length ? 'Deselect all' : 'Select all'}
              </button>
              <button className="btn btn-primary" onClick={handleProcess}
                disabled={selected.size === 0 || loading}
                style={{ minWidth: '140px' }}>
                {loading
                  ? <><Spinner />Starting…</>
                  : selected.size === 0
                  ? 'Select papers'
                  : `Process ${selected.size} paper${selected.size !== 1 ? 's' : ''}`
                }
              </button>
            </div>
          </div>

          {/* Progress bar for selection count */}
          <div style={{ height: '3px', background: 'var(--border)', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{ height: '100%', background: 'var(--accent)', borderRadius: '2px', width: `${processable.length ? (selected.size / processable.length) * 100 : 0}%`, transition: 'width 0.3s ease' }} />
          </div>
        </div>

        {error && <div className="notice notice-error" style={{ marginBottom: '16px' }}>{error}</div>}

        {/* Paper cards — summaries generated staggered 400ms apart */}
        {papers.map((paper, i) => (
          <div key={paper.paper_id} style={{ animationDelay: `${i * 60}ms` }}>
            <PaperSelectionCard
              paper={paper}
              selected={selected.has(paper.paper_id)}
              onToggle={() => toggle(paper.paper_id)}
            />
          </div>
        ))}

        {papers.length === 0 && (
          <div style={{ textAlign: 'center', padding: '48px', color: 'var(--text-3)' }}>
            No papers found. Try a different topic or adjust your filters.
          </div>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// VIEW 3: Processing
// ─────────────────────────────────────────────────────────────────────────────

function PaperCard({ paper }: { paper: PaperStatus }) {
  const stage    = paper.sseStage ?? paper.pipeline_stage
  const isActive = !isTerminal(stage)
  return (
    <div className="card fade-in" style={{ marginBottom: '10px' }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '12px' }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-3)', background: 'var(--bg-3)', padding: '1px 6px', borderRadius: '4px', textTransform: 'uppercase' }}>{paper.source}</span>
            {isActive && <span className="pulse-dot" />}
          </div>
          <p style={{ fontWeight: 500, fontSize: '0.9375rem', color: 'var(--text)', lineHeight: 1.4, marginBottom: '4px', overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }}>
            {paper.title ?? paper.paper_id}
          </p>
          {paper.sseMessage && <p style={{ fontSize: '0.8125rem', color: 'var(--text-3)', marginTop: '2px' }}>{paper.sseMessage}</p>}
          {stage.startsWith('failed') && paper.error_message && <p style={{ fontSize: '0.8125rem', color: 'var(--error)', marginTop: '4px' }}>{paper.error_message}</p>}
        </div>
        <span className={`badge ${stageClass(stage)}`} style={{ flexShrink: 0 }}>
          {isActive && <Spinner size="spinner-sm" />}
          {stageLabel(stage)}
        </span>
      </div>
    </div>
  )
}

function ProcessingView({ papers, onDone }: {
  papers: Paper[]
  onDone: (processedPapers: Paper[]) => void
}) {
  const [statuses, setStatuses] = useState<Record<string, PaperStatus>>(
    () => Object.fromEntries(papers.map(p => [p.paper_id, { ...p }]))
  )
  const esRefs  = useRef<Record<string, EventSource>>({})
  const doneRef = useRef(false)

  const update = useCallback((paperId: string, patch: Partial<PaperStatus>) => {
    setStatuses(prev => ({ ...prev, [paperId]: { ...prev[paperId], ...patch } }))
  }, [])

  useEffect(() => {
    papers.forEach(paper => {
      const es = new EventSource(`${BACKEND}/api/papers/${paper.paper_id}/progress`)
      esRefs.current[paper.paper_id] = es

      es.addEventListener('progress', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data)
          update(paper.paper_id, { sseStage: data.stage, sseMessage: data.message ?? '' })
        } catch { /* ignore */ }
      })

      es.addEventListener('done', (e: MessageEvent) => {
        try {
          const data = JSON.parse(e.data)
          update(paper.paper_id, {
            sseStage:       data.success ? 'processed' : 'failed_processing',
            sseMessage:     data.message ?? (data.success ? 'Ready' : data.error ?? 'Failed'),
            pipeline_stage: data.success ? 'processed' : 'failed_processing',
            chunk_count:    data.chunk_count ?? paper.chunk_count,
            error_message:  data.success ? null : (data.error ?? 'Processing failed'),
          })
        } catch { /* ignore */ }
        es.close()
        delete esRefs.current[paper.paper_id]
      })

      es.onerror = () => {
        update(paper.paper_id, { sseStage: paper.pipeline_stage })
        es.close()
        delete esRefs.current[paper.paper_id]
      }
    })

    return () => {
      Object.values(esRefs.current).forEach(es => es.close())
      esRefs.current = {}
    }
  }, [papers, update])

  useEffect(() => {
    if (doneRef.current) return
    const all = Object.values(statuses)
    if (all.length === 0) return
    if (all.every(p => isTerminal(p.sseStage ?? p.pipeline_stage))) {
      doneRef.current = true
      setTimeout(() => onDone(Object.values(statuses)), 1200)
    }
  }, [statuses, onDone])

  const list      = Object.values(statuses)
  const doneCount = list.filter(p => isTerminal(p.sseStage ?? p.pipeline_stage)).length
  const allDone   = doneCount === list.length

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '40px 24px' }}>
      <div style={{ width: '100%', maxWidth: '640px' }}>
        <div style={{ marginBottom: '28px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Processing papers</h2>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8125rem', color: 'var(--text-2)' }}>{doneCount}/{list.length}</span>
          </div>
          <div style={{ height: '3px', background: 'var(--border)', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{ height: '100%', background: 'var(--accent)', borderRadius: '2px', width: `${list.length ? (doneCount / list.length) * 100 : 0}%`, transition: 'width 0.4s ease' }} />
          </div>
        </div>

        {list.map(paper => <PaperCard key={paper.paper_id} paper={paper} />)}

        {allDone && (
          <div style={{ textAlign: 'center', marginTop: '24px' }} className="fade-in">
            <p style={{ color: 'var(--text-2)', marginBottom: '12px', fontSize: '0.9375rem' }}>
              {list.filter(p => (p.sseStage ?? p.pipeline_stage) === 'processed').length} paper(s) ready — opening chat…
            </p>
            <button className="btn btn-ghost" onClick={() => onDone(Object.values(statuses))}>Continue now</button>
          </div>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Generation Panel
// ─────────────────────────────────────────────────────────────────────────────

function GenerationPanel({ paperId }: { paperId: string }) {
  const [open, setOpen]               = useState(true)
  const [platform, setPlatform]       = useState<Platform>('twitter')
  const [style, setStyle]             = useState('educational')
  const [tone, setTone]               = useState('conversational')
  const [colorScheme, setColorScheme] = useState('light')
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState('')
  const [result, setResult]           = useState<SocialItem | null>(null)
  const [exporting, setExporting]     = useState(false)
  const [exportUrl, setExportUrl]     = useState('')
  const [shareLinks, setShareLinks]   = useState<{ linkedin_url: string; twitter_url: string } | null>(null)
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => { setResult(null); setError(''); setExportUrl(''); setShareLinks(null) }, [platform])
  useEffect(() => () => { esRef.current?.close() }, [])

  async function handleGenerate() {
    setLoading(true); setError(''); setResult(null); setExportUrl(''); setShareLinks(null)
    try {
      const res = await generateContent({ paper_id: paperId, platform, style, tone, color_scheme: colorScheme })
      const qk  = res.queue_key
      await new Promise<void>((resolve, reject) => {
        const es = new EventSource(`${BACKEND}/api/generate/${qk}/progress`)
        esRef.current = es
        es.addEventListener('completed', async () => {
          try {
            const hist = await generationHistory(paperId, platform)
            const latest = hist.items[0] ?? null
            setResult(latest)
            if (latest) { const links = await getShareLinks(latest.id); setShareLinks(links) }
          } catch { /* ignore */ }
        })
        es.addEventListener('done', (e: MessageEvent) => {
          try { const d = JSON.parse(e.data); if (!d.success && d.error) reject(new Error(d.error)); else resolve() }
          catch { resolve() }
          es.close()
        })
        es.addEventListener('failed', (e: MessageEvent) => {
          try { reject(new Error(JSON.parse(e.data).error ?? 'Generation failed')) }
          catch { reject(new Error('Generation failed')) }
          es.close()
        })
        es.onerror = () => { reject(new Error('SSE connection lost')); es.close() }
      })
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Generation failed.')
    } finally { setLoading(false) }
  }

  async function handleExport() {
    if (!result) return
    setExporting(true)
    try {
      const res = await exportCarousel(result.id)
      setExportUrl(`/api/generate/${result.id}/download?filename=${encodeURIComponent(res.filename)}`)
    } catch (err: unknown) { setError(err instanceof Error ? err.message : 'Export failed.') }
    finally { setExporting(false) }
  }

  function renderPreview() {
    if (!result) return null
    try {
      if (platform === 'twitter') {
        const tweets: string[] = JSON.parse(result.content)
        return (
          <div>
            {tweets.map((t, i) => (
              <div key={i} style={{ padding: '10px 12px', borderBottom: i < tweets.length - 1 ? '1px solid var(--border)' : 'none', fontSize: '0.875rem', lineHeight: 1.6 }}>
                <span style={{ color: 'var(--text-3)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem', marginRight: '8px' }}>{i + 1}</span>{t}
              </div>
            ))}
            {result.hashtags?.length > 0 && (
              <div style={{ padding: '8px 12px', borderTop: '1px solid var(--border)', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {result.hashtags.map(h => <span key={h} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--accent)', background: 'var(--accent-glow)', padding: '2px 8px', borderRadius: '12px' }}>{h}</span>)}
              </div>
            )}
          </div>
        )
      }
      if (platform === 'linkedin') {
        return (
          <div style={{ padding: '12px', fontSize: '0.875rem', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
            {result.content}
            {result.hashtags?.length > 0 && (
              <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                {result.hashtags.map(h => <span key={h} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--accent)' }}>{h}</span>)}
              </div>
            )}
          </div>
        )
      }
      if (platform === 'carousel') {
        const slides: Array<{ type: string; title: string; body: string }> = JSON.parse(result.content)
        return (
          <div>
            {slides.map((s, i) => (
              <div key={i} style={{ padding: '10px 12px', borderBottom: i < slides.length - 1 ? '1px solid var(--border)' : 'none' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: 'var(--text-3)', background: 'var(--bg-3)', padding: '1px 6px', borderRadius: '4px' }}>{s.type}</span>
                  <span style={{ fontWeight: 600, fontSize: '0.875rem' }}>{s.title}</span>
                </div>
                <p style={{ fontSize: '0.8125rem', color: 'var(--text-2)', lineHeight: 1.5 }}>{s.body}</p>
              </div>
            ))}
          </div>
        )
      }
    } catch { return <div style={{ padding: '12px', fontSize: '0.875rem' }}>{result.content}</div> }
  }

  return (
    <div style={{ borderTop: '1px solid var(--border)' }}>
      <button onClick={() => setOpen(o => !o)}
        style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '14px 16px', background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text)', borderBottom: open ? '1px solid var(--border)' : 'none' }}>
        <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-2)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Generate content</span>
        <IconChevron open={open} />
      </button>

      {open && (
        <div style={{ padding: '14px 16px' }}>
          <div style={{ display: 'flex', gap: '6px', marginBottom: '14px' }}>
            {(['twitter', 'linkedin', 'carousel'] as Platform[]).map(p => (
              <button key={p} onClick={() => setPlatform(p)}
                style={{ flex: 1, padding: '6px', borderRadius: 'var(--radius)', border: `1px solid ${platform === p ? 'var(--accent)' : 'var(--border)'}`, background: platform === p ? 'var(--accent-glow)' : 'transparent', color: platform === p ? 'var(--accent)' : 'var(--text-2)', fontFamily: 'var(--font-sans)', fontSize: '0.8125rem', cursor: 'pointer', transition: 'all 0.15s', fontWeight: 500 }}>
                {p.charAt(0).toUpperCase() + p.slice(1)}
              </button>
            ))}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '10px' }}>
            <div><label className="label">Style</label><input className="input" value={style} onChange={e => setStyle(e.target.value)} placeholder="e.g. educational" /></div>
            <div><label className="label">Tone</label><input className="input" value={tone} onChange={e => setTone(e.target.value)} placeholder="e.g. conversational" /></div>
          </div>
          {platform === 'carousel' && (
            <div style={{ marginBottom: '10px' }}>
              <label className="label">Color scheme</label>
              <select className="select" value={colorScheme} onChange={e => setColorScheme(e.target.value)}>
                <option value="light">Light</option><option value="dark">Dark</option><option value="bold">Bold</option>
              </select>
            </div>
          )}
          {error && <div className="notice notice-error" style={{ marginBottom: '10px' }}>{error}</div>}
          <button className="btn btn-primary btn-full" onClick={handleGenerate} disabled={loading} style={{ marginBottom: '12px' }}>
            {loading ? <><Spinner />Generating…</> : 'Generate'}
          </button>
          {result && (
            <div className="fade-in" style={{ border: '1px solid var(--border)', borderRadius: 'var(--radius)', overflow: 'hidden', marginBottom: '10px' }}>
              <div style={{ background: 'var(--bg-3)', padding: '8px 12px', borderBottom: '1px solid var(--border)' }}>
                <span style={{ fontSize: '0.8125rem', fontWeight: 500, color: 'var(--text-2)' }}>Preview</span>
              </div>
              <div style={{ background: 'var(--bg-2)', maxHeight: '260px', overflowY: 'auto' }}>{renderPreview()}</div>
            </div>
          )}
          {result && (
            <div className="fade-in" style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              {platform === 'carousel' && (
                <button className="btn btn-ghost btn-sm" onClick={handleExport} disabled={exporting}>
                  {exporting ? <><Spinner size="spinner-sm" />Exporting…</> : <><IconDownload />Export PDF</>}
                </button>
              )}
              {exportUrl && <a href={exportUrl} download className="btn btn-ghost btn-sm"><IconDownload />Download PDF</a>}
              {shareLinks && (
                <>
                  <a href={shareLinks.linkedin_url} target="_blank" rel="noopener noreferrer" className="btn btn-ghost btn-sm"><IconShare />LinkedIn</a>
                  <a href={shareLinks.twitter_url}  target="_blank" rel="noopener noreferrer" className="btn btn-ghost btn-sm"><IconShare />Twitter/X</a>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Follow-up suggestions
// ─────────────────────────────────────────────────────────────────────────────

function FollowUpSuggestions({ lastResponse, onSelect }: {
  lastResponse: string
  onSelect: (q: string) => void
}) {
  const [questions, setQuestions] = useState<string[]>([])
  const [loading, setLoading]     = useState(false)
  const prevResponse = useRef('')

  useEffect(() => {
    if (!lastResponse || lastResponse === prevResponse.current) return
    prevResponse.current = lastResponse
    setLoading(true); setQuestions([])
    fetch('/api/generate/followup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ context: lastResponse }),
    })
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(data => {
        if (Array.isArray(data.questions))
          setQuestions(data.questions.slice(0, 3).map(String).filter(Boolean))
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [lastResponse])

  if (!loading && questions.length === 0) return null

  return (
    <div className="fade-in" style={{ marginBottom: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
        <IconSparkle />
        <span style={{ fontSize: '0.75rem', color: 'var(--text-3)', fontWeight: 500 }}>Suggested follow-ups</span>
      </div>
      {loading ? (
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Spinner size="spinner-sm" /><span style={{ fontSize: '0.8125rem', color: 'var(--text-3)' }}>Thinking…</span>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {questions.map((q, i) => (
            <button key={i} onClick={() => onSelect(q)}
              style={{ textAlign: 'left', padding: '8px 12px', background: 'var(--bg-2)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', cursor: 'pointer', fontSize: '0.8125rem', color: 'var(--text-2)', lineHeight: 1.4, transition: 'all 0.15s' }}
              onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--accent)'; (e.currentTarget as HTMLButtonElement).style.color = 'var(--text)' }}
              onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--border)'; (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-2)' }}>
              {q}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// VIEW 4: Chat
// ─────────────────────────────────────────────────────────────────────────────

function ChatView({ initialPapers, onNewSearch }: { initialPapers: Paper[]; onNewSearch: () => void }) {
  const [sessions, setSessions]           = useState<Session[]>([])
  const [activeSession, setActiveSession] = useState<SessionDetail | null>(null)
  const [messages, setMessages]           = useState<Message[]>([])
  const [input, setInput]                 = useState('')
  const [level, setLevel]                 = useState<Level>('beginner')
  const [sending, setSending]             = useState(false)
  const [chatError, setChatError]         = useState('')
  const [sessError, setSessError]         = useState('')
  const [generateOpen, setGenerateOpen]   = useState(false)
  const [activePaper, setActivePaper]     = useState<Paper | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  const lastAssistantMessage = messages.filter(m => m.role === 'assistant').slice(-1)[0]?.content ?? ''

  useEffect(() => { loadSessions() }, [])

  async function loadSessions() {
    try {
      const res = await listSessions()
      setSessions(res.sessions)
      if (res.sessions.length > 0 && !activeSession) await openSession(res.sessions[0].session_id)
    } catch { /* non-critical */ }
  }

  async function openSession(sessionId: string) {
    setSessError('')
    try {
      const res = await getSession(sessionId)
      setActiveSession(res.session)
      setMessages(res.session.messages)
      setLevel((res.session.level as Level) ?? 'beginner')
      try {
        const papersRes = await listPapers({ stage: 'processed' })
        const current = papersRes.papers.find(p => p.paper_id === res.session.paper_id)
        setActivePaper(current ?? initialPapers.find(p => p.paper_id === res.session.paper_id) ?? null)
      } catch {
        setActivePaper(initialPapers.find(p => p.paper_id === res.session.paper_id) ?? null)
      }
    } catch (err: unknown) {
      setSessError(err instanceof Error ? err.message : 'Failed to load session.')
    }
  }

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  async function handleSend(e: React.FormEvent) {
    e.preventDefault()
    if (!input.trim() || !activeSession || sending) return
    await submitMessage(input.trim())
  }

  async function submitMessage(text: string) {
    if (!activeSession || sending) return
    setInput(''); setChatError(''); setSending(true)
    const tempMsg: Message = { id: Date.now(), role: 'user', content: text, level, created_at: null }
    setMessages(prev => [...prev, tempMsg])
    try {
      const res = await sendMessage(activeSession.session_id, text, level)
      const assistantMsg: Message = { id: Date.now() + 1, role: 'assistant', content: res.response, level, created_at: null }
      setMessages(prev => [...prev, assistantMsg])
    } catch (err: unknown) {
      setChatError(err instanceof Error ? err.message : 'Message failed.')
      setMessages(prev => prev.filter(m => m.id !== tempMsg.id))
      setInput(text)
    } finally { setSending(false) }
  }

  async function handleLevelChange(newLevel: Level) {
    setLevel(newLevel)
    if (activeSession) { try { await updateLevel(activeSession.session_id, newLevel) } catch { /* non-critical */ } }
  }

  const paperIsReady = activePaper?.pipeline_stage === 'processed' && (activePaper?.chunk_count ?? 0) > 0

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* Sidebar */}
      <div style={{ width: '260px', flexShrink: 0, background: 'var(--bg-2)', borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '16px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--accent)', display: 'inline-block', flexShrink: 0 }} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--accent)', letterSpacing: '0.08em' }}>RESEARCHRAG</span>
          </div>
          <button className="btn btn-ghost btn-full btn-sm" onClick={onNewSearch}>+ New search</button>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
          {sessError && <div className="notice notice-error" style={{ margin: '8px' }}>{sessError}</div>}
          {sessions.length === 0 && <p style={{ color: 'var(--text-3)', fontSize: '0.8125rem', padding: '12px 8px', textAlign: 'center' }}>No sessions yet</p>}
          {sessions.map(s => (
            <button key={s.session_id} onClick={() => openSession(s.session_id)}
              style={{ width: '100%', textAlign: 'left', padding: '10px', borderRadius: 'var(--radius)', border: 'none', cursor: 'pointer', background: activeSession?.session_id === s.session_id ? 'var(--bg-3)' : 'transparent', color: activeSession?.session_id === s.session_id ? 'var(--text)' : 'var(--text-2)', marginBottom: '2px', transition: 'all 0.12s' }}>
              <p style={{ fontSize: '0.875rem', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginBottom: '2px' }}>{s.topic ?? s.session_id}</p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{s.level}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Main */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>
        <div style={{ padding: '14px 20px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'var(--bg-2)', flexShrink: 0, gap: '12px' }}>
          <div style={{ minWidth: 0 }}>
            <p style={{ fontWeight: 600, fontSize: '0.9375rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {activeSession?.topic ?? activePaper?.title ?? 'Select a session'}
            </p>
            {activePaper && <p style={{ fontSize: '0.75rem', color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{activePaper.source} · {activePaper.chunk_count} chunks</p>}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
            <div style={{ display: 'flex', gap: '4px' }}>
              {(['beginner', 'intermediate', 'advanced'] as Level[]).map(l => (
                <button key={l} onClick={() => handleLevelChange(l)}
                  style={{ padding: '4px 10px', borderRadius: '20px', border: `1px solid ${level === l ? 'var(--accent)' : 'var(--border)'}`, background: level === l ? 'var(--accent-glow)' : 'transparent', color: level === l ? 'var(--accent)' : 'var(--text-3)', fontFamily: 'var(--font-sans)', fontSize: '0.75rem', cursor: 'pointer', transition: 'all 0.12s', fontWeight: 500 }}>
                  {l.charAt(0).toUpperCase() + l.slice(1)}
                </button>
              ))}
            </div>
            <button onClick={() => setGenerateOpen(o => !o)} className="btn btn-ghost btn-sm"
              style={{ borderColor: generateOpen ? 'var(--accent)' : undefined, color: generateOpen ? 'var(--accent)' : undefined }}>
              {generateOpen ? 'Hide generate' : 'Generate ✦'}
            </button>
          </div>
        </div>

        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>
            <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
              {!activeSession && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                  <p style={{ color: 'var(--text-3)', textAlign: 'center', fontSize: '0.9375rem' }}>
                    {sessions.length === 0 ? 'Waiting for papers to process…' : 'Select a session from the sidebar'}
                  </p>
                </div>
              )}
              {messages.map((msg, i) => (
                <div key={msg.id ?? i} className="fade-in" style={{ display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start', marginBottom: '14px' }}>
                  <div style={{ maxWidth: '72%', padding: '10px 14px', borderRadius: 'var(--radius-lg)', background: msg.role === 'user' ? 'var(--accent)' : 'var(--bg-3)', color: msg.role === 'user' ? '#fff' : 'var(--text)', fontSize: '0.9rem', lineHeight: 1.6, borderBottomRightRadius: msg.role === 'user' ? '4px' : undefined, borderBottomLeftRadius: msg.role === 'assistant' ? '4px' : undefined }}>
                    {msg.content}
                  </div>
                </div>
              ))}
              {sending && (
                <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: '14px' }}>
                  <div style={{ padding: '10px 14px', borderRadius: 'var(--radius-lg)', borderBottomLeftRadius: '4px', background: 'var(--bg-3)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <Spinner size="spinner-sm" /><span style={{ fontSize: '0.875rem', color: 'var(--text-3)' }}>Thinking…</span>
                  </div>
                </div>
              )}
              {activeSession && !sending && lastAssistantMessage && (
                <FollowUpSuggestions lastResponse={lastAssistantMessage} onSelect={q => submitMessage(q)} />
              )}
              <div ref={bottomRef} />
            </div>

            <div style={{ padding: '14px 20px', borderTop: '1px solid var(--border)', background: 'var(--bg-2)', flexShrink: 0 }}>
              {chatError && <div className="notice notice-error" style={{ marginBottom: '10px' }}>{chatError}</div>}
              <form onSubmit={handleSend} style={{ display: 'flex', gap: '8px' }}>
                <input className="input" value={input} onChange={e => setInput(e.target.value)} placeholder={activeSession ? 'Ask anything about this paper…' : 'Waiting for a session…'} disabled={!activeSession || sending} maxLength={2000} style={{ flex: 1 }} />
                <button type="submit" className="btn btn-primary" disabled={!activeSession || !input.trim() || sending} style={{ flexShrink: 0 }}><IconSend /></button>
              </form>
            </div>
          </div>

          {generateOpen && (
            <div className="fade-in" style={{ width: '300px', flexShrink: 0, borderLeft: '1px solid var(--border)', overflowY: 'auto', background: 'var(--bg)', display: 'flex', flexDirection: 'column' }}>
              {paperIsReady
                ? <GenerationPanel paperId={activePaper!.paper_id} />
                : (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '120px', padding: '20px' }}>
                    <p style={{ color: 'var(--text-3)', fontSize: '0.8125rem', textAlign: 'center', lineHeight: 1.6 }}>
                      Paper must finish processing before you can generate content.
                    </p>
                  </div>
                )
              }
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Root App
// ─────────────────────────────────────────────────────────────────────────────

export default function App() {
  const [view, setView]               = useState<View>('search')
  const [searchResults, setSearchResults] = useState<PaperPreview[]>([])
  const [papers, setPapers]           = useState<Paper[]>([])

  function handleSearchResults(results: PaperPreview[]) {
    setSearchResults(results)
    // Upload goes straight to processing (paper already saved by uploadPaper)
    if (results.length === 1 && results[0].source === 'local') {
      // Convert preview to Paper-like for ProcessingView
      setPapers([{
        paper_id: results[0].paper_id, source: 'local',
        title: results[0].title, abstract: results[0].abstract,
        authors: results[0].authors, url: results[0].url,
        pipeline_stage: 'pending', chunk_count: 0,
        error_message: null, topic: null, created_at: null, processed_at: null,
      }])
      setView('processing')
    } else {
      setView('selection')
    }
  }

  function handleSelection(selected: PaperPreview[]) {
    // processPapers already called — convert previews to Paper-like for ProcessingView
    setPapers(selected.map(p => ({
      paper_id: p.paper_id, source: p.source,
      title: p.title, abstract: p.abstract,
      authors: p.authors, url: p.url,
      pipeline_stage: 'pending' as const, chunk_count: 0,
      error_message: null, topic: null, created_at: null, processed_at: null,
    })))
    setView('processing')
  }

  function handleProcessingDone(processedPapers: Paper[]) {
    setPapers(processedPapers)
    setView('chat')
  }

  function handleNewSearch() {
    setSearchResults([])
    setView('search')
    // Note: papers and sessions persist in ChatView sidebar
  }

  return (
    <>
      {view === 'search'     && <SearchView onResults={handleSearchResults} />}
      {view === 'selection'  && <SelectionView papers={searchResults} onSelect={handleSelection} onBack={() => setView('search')} />}
      {view === 'processing' && <ProcessingView papers={papers} onDone={handleProcessingDone} />}
      {view === 'chat'       && <ChatView initialPapers={papers} onNewSearch={handleNewSearch} />}
    </>
  )
}