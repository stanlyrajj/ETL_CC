'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  searchPapers, processPapers, uploadPaper, listSessions, getSession,
  createSession, listPapers, sendMessage, updateLevel, generateContent,
  generationHistory, deleteSession, exportCarousel, getShareLinks,
  getModels, selectModel,
  type Paper, type PaperPreview, type Session, type SessionDetail,
  type Message, type SocialItem, type ModelOption, type ChatMode,
} from './lib/api'

// ── Types ─────────────────────────────────────────────────────────────────────
type View     = 'search' | 'selection' | 'processing' | 'chat'
type Level    = 'beginner' | 'intermediate' | 'advanced'
type Platform = 'twitter' | 'linkedin' | 'carousel'

interface PaperStatus extends Paper { sseStage?: string; sseMessage?: string }

const BACKEND = 'http://localhost:8000'

// ── Chat mode config ──────────────────────────────────────────────────────────
const CHAT_MODES: { id: ChatMode; label: string; description: string; color: string }[] = [
  { id: 'standard',  label: 'Chat',      description: 'Ask anything about this paper',  color: 'var(--accent)' },
  { id: 'study',     label: 'Study',     description: 'Structured lesson + flashcards', color: 'var(--study-color)' },
  { id: 'technical', label: 'Technical', description: 'System design & implementation', color: 'var(--technical-color)' },
]

// ── arXiv categories ──────────────────────────────────────────────────────────
const ARXIV_CATEGORIES = [
  { label: 'All Categories', value: '' },
  { label: '── Computer Science ──', value: '', disabled: true },
  { label: 'Artificial Intelligence', value: 'cs.AI' },
  { label: 'Machine Learning', value: 'cs.LG' },
  { label: 'Computation and Language (NLP)', value: 'cs.CL' },
  { label: 'Computer Vision', value: 'cs.CV' },
  { label: 'Robotics', value: 'cs.RO' },
  { label: 'Human-Computer Interaction', value: 'cs.HC' },
  { label: 'Information Retrieval', value: 'cs.IR' },
  { label: 'Neural and Evolutionary Computing', value: 'cs.NE' },
  { label: '── Biology & Medicine ──', value: '', disabled: true },
  { label: 'Biomolecules', value: 'q-bio.BM' },
  { label: 'Genomics', value: 'q-bio.GN' },
  { label: 'Neurons and Cognition', value: 'q-bio.NC' },
  { label: 'Quantitative Methods', value: 'q-bio.QM' },
  { label: '── Statistics & Mathematics ──', value: '', disabled: true },
  { label: 'Machine Learning (Statistics)', value: 'stat.ML' },
  { label: 'Statistics Theory', value: 'stat.TH' },
  { label: 'Mathematics General', value: 'math.GM' },
  { label: '── Physics & Engineering ──', value: '', disabled: true },
  { label: 'Physics General', value: 'physics.gen-ph' },
  { label: 'Electrical Engineering', value: 'eess.SP' },
  { label: 'Systems and Control', value: 'eess.SY' },
  { label: '── Economics ──', value: '', disabled: true },
  { label: 'Economics', value: 'econ.GN' },
]

// ── Helpers ───────────────────────────────────────────────────────────────────
function stageLabel(stage: string): string {
  const map: Record<string, string> = {
    pending: 'Pending', downloading: 'Downloading…', downloaded: 'Downloaded',
    processing: 'Processing…', processed: 'Ready',
    failed_download: 'Download failed', failed_processing: 'Processing failed',
  }
  return map[stage] ?? stage
}
function stageClass(stage: string): string {
  if (stage === 'processed')      return 'badge-done'
  if (stage.startsWith('failed')) return 'badge-error'
  if (stage === 'pending')        return 'badge-pending'
  return 'badge-active'
}
function isTerminal(stage: string) { return stage === 'processed' || stage.startsWith('failed') }
function truncateAbstract(text: string, sentences = 2): string {
  if (!text) return ''
  const parts = text.match(/[^.!?]+[.!?]+/g) || []
  return parts.slice(0, sentences).join(' ').trim() || text.slice(0, 200)
}
function relativeTime(iso: string | null): string {
  if (!iso) return ''
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000), hours = Math.floor(diff / 3600000), days = Math.floor(diff / 86400000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  if (hours < 24) return `${hours}h ago`
  if (days === 1) return 'Yesterday'
  if (days < 7) return `${days}d ago`
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}
type SessionGroup = { label: string; sessions: Session[] }
function groupSessions(sessions: Session[]): SessionGroup[] {
  const now = Date.now()
  const today: Session[] = [], yesterday: Session[] = [], week: Session[] = [], older: Session[] = []
  for (const s of sessions) {
    const days = (now - new Date(s.last_active_at ?? s.created_at ?? 0).getTime()) / 86400000
    if (days < 1) today.push(s); else if (days < 2) yesterday.push(s); else if (days < 7) week.push(s); else older.push(s)
  }
  const groups: SessionGroup[] = []
  if (today.length)     groups.push({ label: 'Today',     sessions: today })
  if (yesterday.length) groups.push({ label: 'Yesterday', sessions: yesterday })
  if (week.length)      groups.push({ label: 'This week', sessions: week })
  if (older.length)     groups.push({ label: 'Older',     sessions: older })
  return groups
}
const SOURCE_COLORS: Record<string, { bg: string; color: string; label: string }> = {
  arxiv:  { bg: 'rgba(180,120,255,0.15)', color: '#c084fc', label: 'arXiv' },
  pubmed: { bg: 'rgba(59,130,246,0.15)',  color: '#60a5fa', label: 'PubMed' },
  local:  { bg: 'rgba(16,185,129,0.15)',  color: 'var(--accent)', label: 'Local' },
}

// ── localStorage helpers ──────────────────────────────────────────────────────
function getStoredBool(key: string, fallback: boolean): boolean {
  try { const v = localStorage.getItem(key); return v === null ? fallback : v === 'true' } catch { return fallback }
}
function setStoredBool(key: string, value: boolean): void {
  try { localStorage.setItem(key, String(value)) } catch { /* ignore */ }
}

// ── Spinner & Icons ───────────────────────────────────────────────────────────
function Spinner({ size = '' }: { size?: string }) { return <span className={`spinner ${size}`} /> }
function IconSearch()   { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg> }
function IconUpload()   { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg> }
function IconSend()     { return <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg> }
function IconShare()    { return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/><polyline points="16 6 12 2 8 6"/><line x1="12" y1="2" x2="12" y2="15"/></svg> }
function IconDownload() { return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> }
function IconChevron({ open }: { open: boolean }) { return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transition: 'transform 0.2s', transform: open ? 'rotate(180deg)' : 'rotate(0deg)' }}><polyline points="6 9 12 15 18 9"/></svg> }
function IconSparkle()  { return <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5z"/></svg> }
function IconTrash()    { return <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/></svg> }

// ── MermaidBlock ──────────────────────────────────────────────────────────────
function MermaidBlock({ code }: { code: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [error, setError] = useState(false)
  useEffect(() => {
    if (!ref.current) return
    const el = ref.current
    import('mermaid').then(m => {
      const mermaid = m.default
      mermaid.initialize({ startOnLoad: false, theme: 'dark', themeVariables: { background: 'transparent', primaryColor: '#1e2333', primaryTextColor: '#e8eaf0', lineColor: '#3a4258', edgeLabelBackground: '#1e2333' } })
      const id = `mermaid-${Math.random().toString(36).slice(2)}`
      mermaid.render(id, code).then(({ svg }) => { el.innerHTML = svg }).catch(() => setError(true))
    }).catch(() => setError(true))
  }, [code])
  if (error) return <pre style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '12px', fontSize: '0.8125rem', color: 'var(--text-2)', overflowX: 'auto' }}>{code}</pre>
  return <div ref={ref} className="mermaid-block" />
}

function MarkdownWithMermaid({ content, className = 'md-body' }: { content: string; className?: string }) {
  const parts = content.split(/(```mermaid[\s\S]*?```)/g)
  return (
    <div className={className}>
      {parts.map((part, i) => {
        const m = part.match(/^```mermaid\n([\s\S]*?)\n?```$/)
        if (m) return <MermaidBlock key={i} code={m[1].trim()} />
        return part.trim() ? <ReactMarkdown key={i}>{part}</ReactMarkdown> : null
      })}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Study Panel — with outline + section caching (Optimization 3)
// ─────────────────────────────────────────────────────────────────────────────

interface StudySection { index: number; title: string; description: string }
interface StudyOutline { summary: string; sections: StudySection[] }
interface Flashcard    { front: string; back: string }
type StudyPhase = 'idle' | 'loading_outline' | 'outline' | 'teaching' | 'flashcards' | 'error'

// Module-level cache keyed by paperId — survives mode switches
// Cleared only when user explicitly clicks "Regenerate"
const _studyCache: Map<string, {
  outline:    StudyOutline
  sections:   { title: string; content: string }[]
  flashcards: Flashcard[]
  phase:      StudyPhase
}> = new Map()

function StudyPanel({ paperId }: { paperId: string }) {
  const cached = _studyCache.get(paperId)

  const [phase, setPhase]               = useState<StudyPhase>(cached?.phase ?? 'idle')
  const [outline, setOutline]           = useState<StudyOutline | null>(cached?.outline ?? null)
  const [sections, setSections]         = useState<{ title: string; content: string }[]>(cached?.sections ?? [])
  const [currentSection, setCurrentSection] = useState(cached?.sections.length ? cached.sections.length - 1 : 0)
  const [sectionLoading, setSectionLoading] = useState(false)
  const [flashcards, setFlashcards]     = useState<Flashcard[]>(cached?.flashcards ?? [])
  const [flashcardsLoading, setFlashcardsLoading] = useState(false)
  const [cardIndex, setCardIndex]       = useState(0)
  const [flipped, setFlipped]           = useState(false)
  const [error, setError]               = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  // Persist state to module cache on every meaningful change
  useEffect(() => {
    if (outline || sections.length > 0 || flashcards.length > 0) {
      _studyCache.set(paperId, { outline: outline!, sections, flashcards, phase })
    }
  }, [paperId, outline, sections, flashcards, phase])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [sections, phase])

  async function fetchOutline() {
    setPhase('loading_outline'); setError('')
    try {
      const res = await fetch(`/api/study/${paperId}/outline`, { method: 'POST' })
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail ?? 'Failed to generate outline') }
      const data = await res.json()
      setOutline(data.outline)
      setSections([])
      setFlashcards([])
      setCurrentSection(0)
      setPhase('outline')
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to generate outline.')
      setPhase('error')
    }
  }

  async function startTeaching() {
    if (!outline) return
    setPhase('teaching'); setSections([]); setCurrentSection(0)
    await loadSection(0)
  }

  async function loadSection(index: number) {
    if (!outline) return
    const section = outline.sections[index]
    setSectionLoading(true)
    try {
      const res = await fetch(`/api/study/${paperId}/section`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ section_title: section.title, section_description: section.description }),
      })
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail ?? 'Failed to load section') }
      const data = await res.json()
      setSections(prev => [...prev, { title: section.title, content: data.content }])
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load section.')
    } finally { setSectionLoading(false) }
  }

  // Pre-fetch next section in background while user reads current (Optimization 6)
  useEffect(() => {
    if (!outline || phase !== 'teaching' || sectionLoading) return
    const nextIndex = sections.length  // next section to load
    if (nextIndex >= outline.sections.length) return
    // Only pre-fetch if user is on the last loaded section
    if (currentSection < sections.length - 1) return
    // Don't pre-fetch if already loaded
    if (sections.length > currentSection + 1) return
    const timer = setTimeout(() => {
      loadSection(nextIndex)
    }, 800)  // small delay so it doesn't compete with current render
    return () => clearTimeout(timer)
  }, [sections.length, currentSection, phase, sectionLoading, outline])

  async function nextSection() {
    if (!outline) return
    const next = currentSection + 1
    if (next < outline.sections.length) {
      setCurrentSection(next)
      // Section may already be pre-fetched — only load if not yet available
      if (next >= sections.length) {
        await loadSection(next)
      }
    } else {
      await fetchFlashcards()
    }
  }

  async function fetchFlashcards() {
    setFlashcardsLoading(true)
    try {
      const res = await fetch(`/api/study/${paperId}/flashcards`, { method: 'POST' })
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail ?? 'Failed to generate flashcards') }
      const data = await res.json()
      setFlashcards(data.cards ?? [])
      setCardIndex(0); setFlipped(false)
      setPhase('flashcards')
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to generate flashcards.')
    } finally { setFlashcardsLoading(false) }
  }

  function clearCache() {
    _studyCache.delete(paperId)
    setPhase('idle'); setOutline(null); setSections([]); setFlashcards([]); setCurrentSection(0); setError('')
  }

  function prevCard() { setCardIndex(i => Math.max(0, i - 1)); setFlipped(false) }
  function nextCard() { setCardIndex(i => Math.min(flashcards.length - 1, i + 1)); setFlipped(false) }

  const totalSections = outline?.sections.length ?? 0
  const progress = totalSections > 0 ? (sections.length / totalSections) * 100 : 0

  if (phase === 'idle') return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', padding: '40px', gap: '16px' }}>
      <div style={{ textAlign: 'center', marginBottom: '8px' }}>
        <p style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--study-color)', marginBottom: '8px' }}>Study Mode</p>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-2)', maxWidth: '360px', lineHeight: 1.6 }}>The AI will analyze this paper and create a personalized learning plan for you. Review the plan, then learn step by step.</p>
      </div>
      <button className="btn btn-primary" onClick={fetchOutline} style={{ background: 'var(--study-color)', minWidth: '200px' }}>Generate Learning Plan</button>
    </div>
  )

  if (phase === 'loading_outline') return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '16px' }}>
      <Spinner size="spinner-lg" />
      <p style={{ color: 'var(--text-2)', fontSize: '0.9rem' }}>Analyzing paper and creating your learning plan…</p>
    </div>
  )

  if (phase === 'error') return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', padding: '40px', gap: '16px' }}>
      <div className="notice notice-error" style={{ maxWidth: '400px' }}>{error}</div>
      <button className="btn btn-ghost" onClick={clearCache}>Try again</button>
    </div>
  )

  if (phase === 'outline' && outline) return (
    <div style={{ padding: '24px', maxWidth: '680px', margin: '0 auto' }}>
      <div className="study-outline-card" style={{ marginBottom: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--study-color)', display: 'inline-block' }} />
          <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--study-color)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Your Learning Plan</span>
        </div>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-2)', lineHeight: 1.6, marginBottom: '20px' }}>{outline.summary}</p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {outline.sections.map((s, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '12px', padding: '12px', background: 'var(--bg-2)', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}>
              <div className="study-section-number" style={{ marginTop: '1px' }}>{i + 1}</div>
              <div>
                <p style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text)', marginBottom: '3px' }}>{s.title}</p>
                <p style={{ fontSize: '0.8125rem', color: 'var(--text-3)', lineHeight: 1.4 }}>{s.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
      <div style={{ display: 'flex', gap: '10px' }}>
        <button className="btn btn-primary btn-lg" onClick={startTeaching} style={{ background: 'var(--study-color)', flex: 1 }}>Begin Learning →</button>
        <button className="btn btn-ghost" onClick={clearCache}>Regenerate</button>
      </div>
    </div>
  )

  if (phase === 'teaching') {
    const isLastSection = currentSection === totalSections - 1
    const currentContent = sections[currentSection]

    return (
      <div style={{ padding: '24px', maxWidth: '720px', margin: '0 auto' }}>
        <div style={{ marginBottom: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
            <span style={{ fontSize: '0.8125rem', color: 'var(--study-color)', fontWeight: 500 }}>Section {currentSection + 1} of {totalSections}</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{Math.round(progress)}%</span>
          </div>
          <div className="study-progress-bar"><div className="study-progress-fill" style={{ width: `${progress}%` }} /></div>
        </div>

        {/* Show only the current section — not all sections at once */}
        {sectionLoading && !currentContent ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '40px 0', color: 'var(--text-3)', fontSize: '0.875rem', justifyContent: 'center' }}>
            <Spinner /><span>Loading section…</span>
          </div>
        ) : currentContent ? (
          <div className="study-section-block fade-in">
            <div className="study-section-header">
              <div className="study-section-number">{currentSection + 1}</div>
              <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, color: 'var(--text)' }}>{currentContent.title}</h3>
            </div>
            <MarkdownWithMermaid content={currentContent.content} />
          </div>
        ) : null}

        {error && <div className="notice notice-error" style={{ marginBottom: '12px' }}>{error}</div>}

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '16px' }}>
          <button className="btn btn-ghost btn-sm" onClick={() => { if (currentSection > 0) { setCurrentSection(s => s - 1) } }} disabled={currentSection === 0}>← Previous</button>
          {flashcardsLoading
            ? <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-3)', fontSize: '0.875rem' }}><Spinner size="spinner-sm" /><span>Generating flashcards…</span></div>
            : (
              <button className="btn btn-primary" onClick={nextSection}
                disabled={sectionLoading && !sections[currentSection + 1]}
                style={{ background: 'var(--study-color)' }}>
                {isLastSection ? 'Finish & Get Flashcards →' : 'Next Section →'}
              </button>
            )
          }
        </div>
        <div ref={bottomRef} />
      </div>
    )
  }

  if (phase === 'flashcards' && flashcards.length > 0) {
    const card = flashcards[cardIndex]
    return (
      <div style={{ padding: '24px', maxWidth: '600px', margin: '0 auto' }}>
        <div style={{ marginBottom: '20px', textAlign: 'center' }}>
          <p style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--study-color)', marginBottom: '4px' }}>Flashcards</p>
          <p style={{ fontSize: '0.8125rem', color: 'var(--text-3)' }}>Card {cardIndex + 1} of {flashcards.length} · Click card to reveal answer</p>
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', gap: '6px', marginBottom: '20px' }}>
          {flashcards.map((_, i) => (
            <button key={i} onClick={() => { setCardIndex(i); setFlipped(false) }}
              style={{ width: '8px', height: '8px', borderRadius: '50%', border: 'none', cursor: 'pointer', background: i === cardIndex ? 'var(--study-color)' : 'var(--border)', transition: 'background 0.15s', padding: 0 }} />
          ))}
        </div>
        <div className={`flashcard-scene${flipped ? ' flipped' : ''}`} onClick={() => setFlipped(f => !f)}>
          <div className="flashcard-inner">
            <div className="flashcard-front">
              <div className="flashcard-label">Question</div>
              <div className="flashcard-text">{card.front}</div>
              <div className="flashcard-hint">Click to reveal answer</div>
            </div>
            <div className="flashcard-back">
              <div className="flashcard-label">Answer</div>
              <div className="flashcard-text">{card.back}</div>
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '8px' }}>
          <button className="btn btn-ghost btn-sm" onClick={prevCard} disabled={cardIndex === 0}>← Previous</button>
          <button className="btn btn-ghost btn-sm" onClick={clearCache}>Restart</button>
          <button className="btn btn-ghost btn-sm" onClick={nextCard} disabled={cardIndex === flashcards.length - 1}>Next →</button>
        </div>
      </div>
    )
  }

  return null
}

// ─────────────────────────────────────────────────────────────────────────────
// Technical Panel — with section caching (Optimization 4)
// ─────────────────────────────────────────────────────────────────────────────

const TECHNICAL_SECTION_DEFS = [
  { key: 'overview',       label: 'Overview' },
  { key: 'concepts',       label: 'Core Concepts' },
  { key: 'architecture',   label: 'System Architecture' },
  { key: 'implementation', label: 'Implementation Details' },
  { key: 'scalability',    label: 'Scalability & Trade-offs' },
]

interface TechnicalSection { key: string; label: string; content: string }
type TechnicalPhase = 'idle' | 'analyzing' | 'done' | 'error'

// Module-level cache — survives mode switches, cleared on "Re-analyze"
const _technicalCache: Map<string, {
  sections: TechnicalSection[]
  phase:    TechnicalPhase
}> = new Map()

function TechnicalPanel({ paperId }: { paperId: string }) {
  const cached = _technicalCache.get(paperId)

  const [phase, setPhase]           = useState<TechnicalPhase>(cached?.phase ?? 'idle')
  const [sections, setSections]     = useState<TechnicalSection[]>(cached?.sections ?? [])
  const [activeKey, setActiveKey]   = useState(cached?.sections[0]?.key ?? 'overview')
  const [loadingKey, setLoadingKey] = useState<string | null>(null)
  const [error, setError]           = useState('')
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (sections.length > 0) {
      _technicalCache.set(paperId, { sections, phase })
    }
  }, [paperId, sections, phase])

  useEffect(() => () => { esRef.current?.close() }, [])

  async function startAnalysis() {
    setPhase('analyzing'); setSections([]); setError(''); setActiveKey('overview')
    _technicalCache.delete(paperId)
    try {
      const res = await fetch(`/api/technical/${paperId}/analyze`, { method: 'POST' })
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail ?? 'Failed to start analysis') }
      const data = await res.json()
      const qk   = data.queue_key
      setLoadingKey('overview')
      await new Promise<void>((resolve, reject) => {
        const es = new EventSource(`${BACKEND}/api/technical/${qk}/progress`)
        esRef.current = es
        es.addEventListener('section', (e: MessageEvent) => {
          try {
            const d = JSON.parse(e.data)
            setSections(prev => [...prev, { key: d.section_key, label: d.section_label, content: d.content }])
            setActiveKey(d.section_key)
            const nextIdx = d.section_index + 1
            setLoadingKey(nextIdx < TECHNICAL_SECTION_DEFS.length ? TECHNICAL_SECTION_DEFS[nextIdx].key : null)
          } catch { /* ignore */ }
        })
        es.addEventListener('section_failed', (e: MessageEvent) => {
          try {
            const d = JSON.parse(e.data)
            setSections(prev => [...prev, { key: d.section_key, label: d.section_label, content: `*This section could not be generated: ${d.error}*` }])
          } catch { /* ignore */ }
        })
        es.addEventListener('done', () => { resolve(); es.close() })
        es.onerror = () => { reject(new Error('SSE connection lost')); es.close() }
      })
      setPhase('done'); setLoadingKey(null)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Analysis failed.')
      setPhase('error'); setLoadingKey(null)
    }
  }

  function clearCache() { _technicalCache.delete(paperId); setPhase('idle'); setSections([]); setError('') }

  const activeSection = sections.find(s => s.key === activeKey) ?? sections[0] ?? null

  if (phase === 'idle') return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', padding: '40px', gap: '16px' }}>
      <div style={{ textAlign: 'center', marginBottom: '8px' }}>
        <p style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--technical-color)', marginBottom: '8px' }}>Technical Mode</p>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-2)', maxWidth: '380px', lineHeight: 1.6 }}>Get a comprehensive technical breakdown — architecture diagrams, implementation details, pseudocode, and engineering trade-offs.</p>
        <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '6px', alignItems: 'center' }}>
          {TECHNICAL_SECTION_DEFS.map((s, i) => (
            <div key={s.key} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8125rem', color: 'var(--text-3)' }}>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--technical-color)', fontSize: '0.75rem' }}>{i + 1}.</span>{s.label}
            </div>
          ))}
        </div>
      </div>
      <button className="btn btn-primary btn-lg" onClick={startAnalysis} style={{ background: 'var(--technical-color)', minWidth: '220px' }}>Analyze Paper</button>
    </div>
  )

  if (phase === 'error') return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', padding: '40px', gap: '16px' }}>
      <div className="notice notice-error" style={{ maxWidth: '400px' }}>{error}</div>
      <button className="btn btn-ghost" onClick={clearCache}>Try again</button>
    </div>
  )

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      <nav className="technical-nav">
        <p style={{ fontSize: '0.6875rem', fontWeight: 600, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.08em', padding: '0 16px 8px' }}>Sections</p>
        {TECHNICAL_SECTION_DEFS.map(def => {
          const done    = sections.some(s => s.key === def.key)
          const loading = loadingKey === def.key
          return (
            <button key={def.key}
              className={`technical-nav-item ${activeKey === def.key ? 'active' : ''} ${done ? 'done' : ''} ${loading ? 'loading' : ''}`}
              onClick={() => done && setActiveKey(def.key)} disabled={!done}
              style={{ cursor: done ? 'pointer' : 'default' }}>
              {loading && <Spinner size="spinner-sm" />}
              {done && !loading && <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--technical-color)', flexShrink: 0 }} />}
              {!done && !loading && <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--border)', flexShrink: 0 }} />}
              {def.label}
            </button>
          )
        })}
        {phase === 'done' && (
          <div style={{ padding: '12px 16px', marginTop: 'auto' }}>
            <button className="btn btn-ghost btn-sm btn-full" onClick={clearCache} style={{ fontSize: '0.75rem' }}>Re-analyze</button>
          </div>
        )}
        {phase === 'analyzing' && (
          <div style={{ padding: '12px 16px', marginTop: 'auto' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-3)', display: 'flex', alignItems: 'center', gap: '6px' }}><Spinner size="spinner-sm" />Analyzing…</div>
          </div>
        )}
      </nav>
      <div style={{ flex: 1, overflowY: 'auto', padding: '24px' }}>
        {sections.length === 0 && phase === 'analyzing' && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '200px', gap: '12px', color: 'var(--text-3)' }}>
            <Spinner /><span>Generating overview…</span>
          </div>
        )}
        {activeSection && (
          <div className="technical-section-block fade-in">
            <div className="technical-section-title">
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--technical-color)', flexShrink: 0 }} />
              {activeSection.label}
            </div>
            <MarkdownWithMermaid content={activeSection.content} />
          </div>
        )}
        {loadingKey && phase === 'analyzing' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '16px 0', color: 'var(--text-3)', fontSize: '0.875rem' }}>
            <Spinner size="spinner-sm" />
            <span>Generating {TECHNICAL_SECTION_DEFS.find(s => s.key === loadingKey)?.label ?? 'next section'}…</span>
          </div>
        )}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// VIEW 1: Search / Upload
// ─────────────────────────────────────────────────────────────────────────────

function SearchView({ onResults }: { onResults: (papers: PaperPreview[]) => void }) {
  const [topic, setTopic]     = useState('')
  const [source, setSource]   = useState<'both' | 'arxiv' | 'pubmed'>('both')
  const [limit, setLimit]     = useState(10)
  const [file, setFile]       = useState<File | null>(null)
  const [mode, setMode]       = useState<'search' | 'upload'>('search')
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  const [showRefine, setShowRefine] = useState(false)
  const [sortBy, setSortBy]         = useState<'date' | 'relevance'>('date')
  const [dateFrom, setDateFrom]     = useState('')
  const [dateTo, setDateTo]         = useState('')
  const [category, setCategory]     = useState('')
  const [keyword, setKeyword]       = useState('')

  const activeFilters = [
    sortBy !== 'date' ? 'By relevance' : '',
    category ? ARXIV_CATEGORIES.find(c => c.value === category)?.label ?? '' : '',
    dateFrom && dateTo ? `${dateFrom} – ${dateTo}` : dateFrom ? `From ${dateFrom}` : dateTo ? `To ${dateTo}` : '',
    keyword ? `"${keyword}"` : '',
  ].filter(Boolean)

  const [models, setModels]                 = useState<ModelOption[]>([])
  const [activeModel, setActiveModel]       = useState('')
  const [modelSwitching, setModelSwitching] = useState(false)
  const [isOpenRouter, setIsOpenRouter]     = useState(false)

  useEffect(() => {
    getModels().then(res => {
      if (res.provider === 'openrouter') { setIsOpenRouter(true); setModels(res.models); setActiveModel(res.active_model) }
    }).catch(() => {})
  }, [])

  async function handleModelSelect(modelId: string) {
    if (modelId === activeModel) return
    setModelSwitching(true)
    try { await selectModel(modelId); setActiveModel(modelId) }
    catch (err: unknown) { setError(err instanceof Error ? err.message : 'Failed to switch model.') }
    finally { setModelSwitching(false) }
  }

  const topicError = topic.trim().length === 0 && topic.length > 0 ? 'Topic cannot be empty.'
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
    } catch (err: unknown) { setError(err instanceof Error ? err.message : 'Search failed.') }
    finally { setLoading(false) }
  }

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault()
    if (!file) { setError('Please select a PDF file.'); return }
    if (!topic.trim()) { setError('Please enter a topic.'); return }
    setError(''); setLoading(true)
    try {
      const result = await uploadPaper(file, topic.trim())
      onResults([{
        paper_id: result.paper.paper_id, source: 'local',
        title: result.paper.title ?? file.name, abstract: result.paper.abstract ?? '',
        authors: result.paper.authors ?? [], url: result.paper.url ?? '',
        has_full_text: true, published: '', journal: '', categories: [], doi: '',
      }])
    } catch (err: unknown) { setError(err instanceof Error ? err.message : 'Upload failed.') }
    finally { setLoading(false) }
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
                  <div><label className="label">Source</label>
                    <select className="select" value={source} onChange={e => setSource(e.target.value as typeof source)}>
                      <option value="both">arXiv + PubMed</option><option value="arxiv">arXiv only</option><option value="pubmed">PubMed only</option>
                    </select>
                  </div>
                  <div><label className="label">Number of results</label>
                    <select className="select" value={limit} onChange={e => setLimit(Number(e.target.value))}>
                      {[5, 10, 20, 30, 50].map(n => <option key={n} value={n}>{n} papers</option>)}
                    </select>
                  </div>
                </div>
                <div style={{ marginBottom: '20px', border: '1px solid var(--border)', borderRadius: 'var(--radius)', overflow: 'hidden' }}>
                  <button type="button" onClick={() => setShowRefine(o => !o)}
                    style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: 'var(--bg-3)', border: 'none', cursor: 'pointer', color: 'var(--text-2)', fontFamily: 'var(--font-sans)', fontSize: '0.8125rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <span style={{ fontWeight: 500 }}>Refine results</span>
                      {activeFilters.length > 0 && !showRefine && <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--accent)' }}>· {activeFilters.join(' · ')}</span>}
                    </div>
                    <IconChevron open={showRefine} />
                  </button>
                  {showRefine && (
                    <div style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <div><label className="label">Sort by</label>
                        <select className="select" value={sortBy} onChange={e => setSortBy(e.target.value as typeof sortBy)}>
                          <option value="date">Most recent</option><option value="relevance">Most relevant</option>
                        </select>
                      </div>
                      {source !== 'pubmed' && (
                        <div><label className="label">arXiv category</label>
                          <select className="select" value={category} onChange={e => setCategory(e.target.value)}>
                            {ARXIV_CATEGORIES.map((c, i) => <option key={i} value={c.value} disabled={c.disabled as boolean | undefined}>{c.label}</option>)}
                          </select>
                        </div>
                      )}
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                        <div><label className="label">From date</label><input className="input" type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)} /></div>
                        <div><label className="label">To date</label><input className="input" type="date" value={dateTo} onChange={e => setDateTo(e.target.value)} /></div>
                      </div>
                      <div><label className="label">Must include keyword</label>
                        <input className="input" value={keyword} onChange={e => setKeyword(e.target.value)} placeholder="e.g. interpretability, fine-tuning" />
                      </div>
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
// VIEW 2: Paper Selection — Optimization 1: on-demand summary
// ─────────────────────────────────────────────────────────────────────────────

function PaperSelectionCard({ paper, selected, onToggle }: {
  paper: PaperPreview; selected: boolean; onToggle: () => void
}) {
  const [summary, setSummary]               = useState<string | null>(null)
  const [summaryLoading, setSummaryLoading] = useState(false)
  const [showFullAbstract, setShowFullAbstract] = useState(false)
  const srcStyle = SOURCE_COLORS[paper.source] ?? SOURCE_COLORS.local

  // OPTIMIZATION 1: Summary is no longer auto-fetched on mount.
  // It only runs when the user explicitly clicks "Summarize".
  async function handleSummarize() {
    if (!paper.abstract || summaryLoading || summary) return
    setSummaryLoading(true)
    try {
      const res = await fetch('/api/generate/followup', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          context: `Summarize this research paper abstract in 2-3 plain sentences for a non-specialist:\n\n${paper.abstract}`,
        }),
      })
      if (res.ok) {
        const data = await res.json()
        const qs: string[] = data.questions ?? []
        setSummary(qs.length > 0 ? qs.join(' ') : truncateAbstract(paper.abstract, 2))
      } else {
        setSummary(truncateAbstract(paper.abstract, 2))
      }
    } catch { setSummary(truncateAbstract(paper.abstract, 2)) }
    finally { setSummaryLoading(false) }
  }

  return (
    <div className="card fade-in" style={{ marginBottom: '12px', border: `1px solid ${selected ? 'var(--accent)' : 'var(--border)'}`, background: selected ? 'rgba(16,185,129,0.04)' : 'var(--bg-2)', transition: 'all 0.15s' }}>
      <div style={{ display: 'flex', gap: '14px', alignItems: 'flex-start' }}>
        <div style={{ flexShrink: 0, paddingTop: '2px' }}>
          <button onClick={onToggle}
            style={{ width: '22px', height: '22px', borderRadius: '6px', border: `2px solid ${selected ? 'var(--accent)' : 'var(--border)'}`, background: selected ? 'var(--accent)' : 'transparent', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, transition: 'all 0.15s' }}>
            {selected && <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2 6l3 3 5-5" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>}
          </button>
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', marginBottom: '6px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.7rem', fontWeight: 600, padding: '2px 8px', borderRadius: '12px', background: srcStyle.bg, color: srcStyle.color, fontFamily: 'var(--font-mono)', flexShrink: 0 }}>{srcStyle.label}</span>
            {!paper.has_full_text && <span style={{ fontSize: '0.7rem', padding: '2px 8px', borderRadius: '12px', background: 'var(--warn-bg)', color: 'var(--warn)', fontFamily: 'var(--font-mono)', flexShrink: 0 }}>Abstract only</span>}
            {paper.published && <span style={{ fontSize: '0.75rem', color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{paper.published.slice(0, 10)}</span>}
          </div>
          <p style={{ fontWeight: 600, fontSize: '0.9375rem', color: 'var(--text)', lineHeight: 1.4, marginBottom: '6px' }}>{paper.title || paper.paper_id}</p>
          {paper.authors.length > 0 && <p style={{ fontSize: '0.8125rem', color: 'var(--text-3)', marginBottom: '8px' }}>{paper.authors.slice(0, 3).join(', ')}{paper.authors.length > 3 ? ' et al.' : ''}</p>}

          {/* Abstract / summary section */}
          <div style={{ fontSize: '0.875rem', color: 'var(--text-2)', lineHeight: 1.6, marginBottom: '8px' }}>
            {summary ? (
              <span>{summary}</span>
            ) : (
              <span style={{ color: 'var(--text-3)' }}>{truncateAbstract(paper.abstract, 2)}</span>
            )}
          </div>

          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'center' }}>
            {/* On-demand summarize button — only shown when abstract exists and no summary yet */}
            {paper.abstract && !summary && (
              <button type="button" onClick={handleSummarize} disabled={summaryLoading}
                style={{ fontSize: '0.8125rem', color: 'var(--accent)', background: 'none', border: 'none', cursor: summaryLoading ? 'default' : 'pointer', padding: 0, display: 'flex', alignItems: 'center', gap: '4px' }}>
                {summaryLoading ? <><Spinner size="spinner-sm" />Summarising…</> : <>✦ AI Summary</>}
              </button>
            )}
            {paper.abstract && (
              <button type="button" onClick={() => setShowFullAbstract(o => !o)}
                style={{ fontSize: '0.8125rem', color: 'var(--text-3)', background: 'none', border: 'none', cursor: 'pointer', padding: 0, textDecoration: 'underline' }}>
                {showFullAbstract ? 'Hide abstract' : 'Show abstract'}
              </button>
            )}
          </div>

          {showFullAbstract && paper.abstract && (
            <p style={{ fontSize: '0.8125rem', color: 'var(--text-2)', lineHeight: 1.6, marginTop: '8px', padding: '10px 12px', background: 'var(--bg)', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}>{paper.abstract}</p>
          )}

          {(paper.journal || paper.doi) && <p style={{ fontSize: '0.75rem', color: 'var(--text-3)', marginTop: '6px', fontFamily: 'var(--font-mono)' }}>{paper.journal}{paper.journal && paper.doi ? ' · ' : ''}{paper.doi ? `DOI: ${paper.doi}` : ''}</p>}
        </div>
      </div>
    </div>
  )
}

function SelectionView({ papers, onSelect, onBack }: {
  papers: PaperPreview[]; onSelect: (selected: PaperPreview[]) => void; onBack: () => void
}) {
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')

  function toggle(paperId: string) {
    setSelected(prev => { const next = new Set(prev); if (next.has(paperId)) next.delete(paperId); else next.add(paperId); return next })
  }
  function selectAll()   { setSelected(new Set(papers.map(p => p.paper_id))) }
  function deselectAll() { setSelected(new Set()) }

  async function handleProcess() {
    if (selected.size === 0) return
    setError(''); setLoading(true)
    try { await processPapers(Array.from(selected)); onSelect(papers.filter(p => selected.has(p.paper_id))) }
    catch (err: unknown) { setError(err instanceof Error ? err.message : 'Failed to start processing.'); setLoading(false) }
  }

  return (
    <div style={{ minHeight: '100vh', padding: '32px 24px' }}>
      <div style={{ maxWidth: '760px', margin: '0 auto' }}>
        <div style={{ marginBottom: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <div>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '4px' }}>Select papers to process</h2>
              <p style={{ color: 'var(--text-3)', fontSize: '0.875rem' }}>{papers.length} result{papers.length !== 1 ? 's' : ''} · {selected.size} selected</p>
            </div>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
              <button className="btn btn-ghost btn-sm" onClick={onBack}>← New search</button>
              <button className="btn btn-ghost btn-sm" onClick={selected.size === papers.length ? deselectAll : selectAll}>{selected.size === papers.length ? 'Deselect all' : 'Select all'}</button>
              <button className="btn btn-primary" onClick={handleProcess} disabled={selected.size === 0 || loading} style={{ minWidth: '140px' }}>
                {loading ? <><Spinner />Starting…</> : selected.size === 0 ? 'Select papers' : `Process ${selected.size} paper${selected.size !== 1 ? 's' : ''}`}
              </button>
            </div>
          </div>
          <div style={{ height: '3px', background: 'var(--border)', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{ height: '100%', background: 'var(--accent)', borderRadius: '2px', width: `${papers.length ? (selected.size / papers.length) * 100 : 0}%`, transition: 'width 0.3s ease' }} />
          </div>
        </div>
        {error && <div className="notice notice-error" style={{ marginBottom: '16px' }}>{error}</div>}
        {papers.map(paper => <PaperSelectionCard key={paper.paper_id} paper={paper} selected={selected.has(paper.paper_id)} onToggle={() => toggle(paper.paper_id)} />)}
        {papers.length === 0 && <div style={{ textAlign: 'center', padding: '48px', color: 'var(--text-3)' }}>No papers found.</div>}
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
          <p style={{ fontWeight: 500, fontSize: '0.9375rem', color: 'var(--text)', lineHeight: 1.4, marginBottom: '4px', overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }}>{paper.title ?? paper.paper_id}</p>
          {paper.sseMessage && <p style={{ fontSize: '0.8125rem', color: 'var(--text-3)', marginTop: '2px' }}>{paper.sseMessage}</p>}
          {stage.startsWith('failed') && paper.error_message && <p style={{ fontSize: '0.8125rem', color: 'var(--error)', marginTop: '4px' }}>{paper.error_message}</p>}
        </div>
        <span className={`badge ${stageClass(stage)}`} style={{ flexShrink: 0 }}>{isActive && <Spinner size="spinner-sm" />}{stageLabel(stage)}</span>
      </div>
    </div>
  )
}

function ProcessingView({ papers, onDone }: { papers: Paper[]; onDone: (processedPapers: Paper[]) => void }) {
  const [statuses, setStatuses] = useState<Record<string, PaperStatus>>(() => Object.fromEntries(papers.map(p => [p.paper_id, { ...p }])))
  const esRefs  = useRef<Record<string, EventSource>>({})
  const doneRef = useRef(false)
  const update  = useCallback((paperId: string, patch: Partial<PaperStatus>) => {
    setStatuses(prev => ({ ...prev, [paperId]: { ...prev[paperId], ...patch } }))
  }, [])

  useEffect(() => {
    papers.forEach(paper => {
      const es = new EventSource(`${BACKEND}/api/papers/${paper.paper_id}/progress`)
      esRefs.current[paper.paper_id] = es
      es.addEventListener('progress', (e: MessageEvent) => { try { const d = JSON.parse(e.data); update(paper.paper_id, { sseStage: d.stage, sseMessage: d.message ?? '' }) } catch { /* ignore */ } })
      es.addEventListener('done', (e: MessageEvent) => {
        try { const d = JSON.parse(e.data); update(paper.paper_id, { sseStage: d.success ? 'processed' : 'failed_processing', sseMessage: d.message ?? (d.success ? 'Ready' : d.error ?? 'Failed'), pipeline_stage: d.success ? 'processed' : 'failed_processing', chunk_count: d.chunk_count ?? paper.chunk_count, error_message: d.success ? null : (d.error ?? 'Processing failed') }) } catch { /* ignore */ }
        es.close(); delete esRefs.current[paper.paper_id]
      })
      es.onerror = () => { update(paper.paper_id, { sseStage: paper.pipeline_stage }); es.close(); delete esRefs.current[paper.paper_id] }
    })
    return () => { Object.values(esRefs.current).forEach(es => es.close()); esRefs.current = {} }
  }, [papers, update])

  useEffect(() => {
    if (doneRef.current) return
    const all = Object.values(statuses)
    if (all.length === 0) return
    if (all.every(p => isTerminal(p.sseStage ?? p.pipeline_stage))) { doneRef.current = true; setTimeout(() => onDone(Object.values(statuses)), 1200) }
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
            <p style={{ color: 'var(--text-2)', marginBottom: '12px', fontSize: '0.9375rem' }}>{list.filter(p => (p.sseStage ?? p.pipeline_stage) === 'processed').length} paper(s) ready — opening chat…</p>
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
  const [description, setDescription] = useState('')
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

  const placeholders: Record<Platform, string> = {
    twitter:  'e.g. "Punchy breakdown for ML engineers, heavy on the data, 1-2 emojis per tweet"',
    linkedin: 'e.g. "Thought-leadership post for tech executives, professional tone, no emojis"',
    carousel: 'e.g. "Visual summary for a non-technical audience, keep it simple and engaging"',
  }

  async function handleGenerate() {
    setLoading(true); setError(''); setResult(null); setExportUrl(''); setShareLinks(null)
    const desc = description.trim() || 'Educational and accessible, suitable for a general professional audience.'
    try {
      const res = await generateContent({
        paper_id: paperId, platform,
        description: desc,
        color_scheme: colorScheme,
      })
      const qk = res.queue_key
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
          try { const d = JSON.parse(e.data); if (!d.success && d.error) reject(new Error(d.error)); else resolve() } catch { resolve() }
          es.close()
        })
        es.addEventListener('failed', (e: MessageEvent) => {
          try { reject(new Error(JSON.parse(e.data).error ?? 'Generation failed')) } catch { reject(new Error('Generation failed')) }
          es.close()
        })
        es.onerror = () => { reject(new Error('SSE connection lost')); es.close() }
      })
    } catch (err: unknown) { setError(err instanceof Error ? err.message : 'Generation failed.') }
    finally { setLoading(false) }
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
        // Try to parse inferred_attributes from the stored content
        // LinkedIn content is stored as plain text, so we display it directly
        return (
          <div>
            <div style={{ padding: '12px', fontSize: '0.875rem', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
              {result.content}
            </div>
            {result.hashtags?.length > 0 && (
              <div style={{ padding: '8px 12px', borderTop: '1px solid var(--border)', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
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

          {/* Platform selector */}
          <div style={{ display: 'flex', gap: '6px', marginBottom: '14px' }}>
            {(['twitter', 'linkedin', 'carousel'] as Platform[]).map(p => (
              <button key={p} onClick={() => setPlatform(p)}
                style={{ flex: 1, padding: '6px', borderRadius: 'var(--radius)', border: `1px solid ${platform === p ? 'var(--accent)' : 'var(--border)'}`, background: platform === p ? 'var(--accent-glow)' : 'transparent', color: platform === p ? 'var(--accent)' : 'var(--text-2)', fontFamily: 'var(--font-sans)', fontSize: '0.8125rem', cursor: 'pointer', transition: 'all 0.15s', fontWeight: 500 }}>
                {p.charAt(0).toUpperCase() + p.slice(1)}
              </button>
            ))}
          </div>

          {/* Content brief */}
          <div style={{ marginBottom: '10px' }}>
            <label className="label">
              Content brief
              <span style={{ fontWeight: 400, color: 'var(--text-3)', marginLeft: '6px' }}>— optional</span>
            </label>
            <textarea
              className="textarea"
              value={description}
              onChange={e => setDescription(e.target.value)}
              placeholder={placeholders[platform]}
              maxLength={500}
              style={{ minHeight: '72px', resize: 'vertical', fontSize: '0.8125rem' }}
            />
            <p style={{ fontSize: '0.75rem', color: 'var(--text-3)', marginTop: '4px', lineHeight: 1.4 }}>
              Describe your audience, tone, and style in plain English. The AI infers everything else.
            </p>
          </div>

          {/* Color scheme — carousel only */}
          {platform === 'carousel' && (
            <div style={{ marginBottom: '10px' }}>
              <label className="label">Color scheme</label>
              <select className="select" value={colorScheme} onChange={e => setColorScheme(e.target.value)}>
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="bold">Bold</option>
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
              <div style={{ background: 'var(--bg-2)', maxHeight: '260px', overflowY: 'auto' }}>
                {renderPreview()}
              </div>
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
// Follow-up suggestions — Optimization 2: opt-in toggle
// ─────────────────────────────────────────────────────────────────────────────

function FollowUpSuggestions({ lastResponse, onSelect, enabled }: {
  lastResponse: string; onSelect: (q: string) => void; enabled: boolean
}) {
  const [questions, setQuestions] = useState<string[]>([])
  const [loading, setLoading]     = useState(false)
  const prevResponse = useRef('')

  useEffect(() => {
    // OPTIMIZATION 2: Only fire if suggestions are enabled AND response changed
    if (!enabled || !lastResponse || lastResponse === prevResponse.current) return
    prevResponse.current = lastResponse
    setLoading(true); setQuestions([])
    fetch('/api/generate/followup', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ context: lastResponse }) })
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(data => { if (Array.isArray(data.questions)) setQuestions(data.questions.slice(0, 3).map(String).filter(Boolean)) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [lastResponse, enabled])

  // Clear questions when disabled
  useEffect(() => { if (!enabled) setQuestions([]) }, [enabled])

  if (!enabled || (!loading && questions.length === 0)) return null

  return (
    <div className="fade-in" style={{ marginBottom: '16px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px' }}>
        <IconSparkle />
        <span style={{ fontSize: '0.75rem', color: 'var(--text-3)', fontWeight: 500 }}>Suggested follow-ups</span>
      </div>
      {loading
        ? <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}><Spinner size="spinner-sm" /><span style={{ fontSize: '0.8125rem', color: 'var(--text-3)' }}>Thinking…</span></div>
        : <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>{questions.map((q, i) => <button key={i} onClick={() => onSelect(q)} style={{ textAlign: 'left', padding: '8px 12px', background: 'var(--bg-2)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', cursor: 'pointer', fontSize: '0.8125rem', color: 'var(--text-2)', lineHeight: 1.4, transition: 'all 0.15s' }} onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--accent)'; (e.currentTarget as HTMLButtonElement).style.color = 'var(--text)' }} onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--border)'; (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-2)' }}>{q}</button>)}</div>
      }
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Session Sidebar
// ─────────────────────────────────────────────────────────────────────────────

function SessionSidebar({ sessions, activeSessionId, onSelect, onDelete, onNewSearch }: {
  sessions: Session[]; activeSessionId: string | null
  onSelect: (sessionId: string) => void
  onDelete: (sessionId: string) => void
  onNewSearch: () => void
}) {
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [deleting, setDeleting]           = useState<string | null>(null)
  const MODE_COLORS: Record<ChatMode, string> = { standard: 'var(--accent)', study: 'var(--study-color)', technical: 'var(--technical-color)' }
  const groups = groupSessions(sessions)

  async function handleDelete(sessionId: string) {
    setDeleting(sessionId)
    try { await deleteSession(sessionId); onDelete(sessionId) } catch { /* non-critical */ }
    finally { setDeleting(null); setConfirmDelete(null) }
  }

  return (
    <div style={{ width: '260px', flexShrink: 0, background: 'var(--bg-2)', borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ padding: '16px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
          <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--accent)', display: 'inline-block', flexShrink: 0 }} />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--accent)', letterSpacing: '0.08em' }}>RESEARCHRAG</span>
        </div>
        <button className="btn btn-ghost btn-full btn-sm" onClick={onNewSearch}>+ New search</button>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
        {sessions.length === 0 && <p style={{ color: 'var(--text-3)', fontSize: '0.8125rem', padding: '16px 8px', textAlign: 'center', lineHeight: 1.6 }}>No sessions yet.<br />Search for a paper to get started.</p>}
        {groups.map(group => (
          <div key={group.label}>
            <p style={{ fontSize: '0.6875rem', fontWeight: 600, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.08em', padding: '8px 8px 4px' }}>{group.label}</p>
            {group.sessions.map(s => {
              const isActive   = s.session_id === activeSessionId
              const isConfirm  = confirmDelete === s.session_id
              const isDeleting = deleting === s.session_id
              const modeColor  = MODE_COLORS[s.mode] ?? 'var(--accent)'
              return (
                <div key={s.session_id} style={{ position: 'relative', borderRadius: 'var(--radius)', marginBottom: '2px', background: isActive ? 'var(--bg-3)' : 'transparent', border: `1px solid ${isActive ? 'var(--border-2)' : 'transparent'}`, transition: 'all 0.12s' }}
                  onMouseEnter={e => { if (!isActive) (e.currentTarget as HTMLDivElement).style.background = 'var(--bg-3)' }}
                  onMouseLeave={e => { if (!isActive) (e.currentTarget as HTMLDivElement).style.background = 'transparent' }}>
                  <button onClick={() => onSelect(s.session_id)} style={{ width: '100%', textAlign: 'left', padding: '9px 32px 9px 10px', background: 'transparent', border: 'none', cursor: 'pointer', borderRadius: 'var(--radius)' }}>
                    <p style={{ fontSize: '0.875rem', fontWeight: isActive ? 600 : 400, color: isActive ? 'var(--text)' : 'var(--text-2)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginBottom: '3px', lineHeight: 1.3 }}>{s.title || s.topic || s.session_id}</p>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ fontSize: '0.6rem', fontFamily: 'var(--font-mono)', fontWeight: 600, color: modeColor, background: `${modeColor}22`, padding: '1px 5px', borderRadius: '8px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{s.mode}</span>
                      <span style={{ fontSize: '0.6875rem', color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{relativeTime(s.last_active_at)}</span>
                    </div>
                  </button>
                  {!isConfirm && (
                    <button onClick={e => { e.stopPropagation(); setConfirmDelete(s.session_id) }} title="Delete session"
                      style={{ position: 'absolute', top: '50%', right: '6px', transform: 'translateY(-50%)', width: '22px', height: '22px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-3)', borderRadius: '4px', opacity: isActive ? 1 : 0, transition: 'opacity 0.15s, color 0.15s' }}
                      onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.color = 'var(--error)'; (e.currentTarget as HTMLButtonElement).style.opacity = '1' }}
                      onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-3)'; if (!isActive) (e.currentTarget as HTMLButtonElement).style.opacity = '0' }}>
                      <IconTrash />
                    </button>
                  )}
                  {isConfirm && (
                    <div className="fade-in" style={{ position: 'absolute', inset: 0, background: 'var(--bg-3)', borderRadius: 'var(--radius)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 10px', gap: '6px', border: '1px solid var(--error-bg)' }}>
                      <span style={{ fontSize: '0.8125rem', color: 'var(--text-2)', whiteSpace: 'nowrap' }}>Delete?</span>
                      <div style={{ display: 'flex', gap: '4px' }}>
                        <button onClick={e => { e.stopPropagation(); handleDelete(s.session_id) }} disabled={isDeleting} style={{ fontSize: '0.75rem', padding: '3px 8px', borderRadius: '4px', border: 'none', background: 'var(--error)', color: '#fff', cursor: 'pointer', fontFamily: 'var(--font-sans)' }}>{isDeleting ? '…' : 'Yes'}</button>
                        <button onClick={e => { e.stopPropagation(); setConfirmDelete(null) }} style={{ fontSize: '0.75rem', padding: '3px 8px', borderRadius: '4px', border: '1px solid var(--border)', background: 'transparent', color: 'var(--text-2)', cursor: 'pointer', fontFamily: 'var(--font-sans)' }}>No</button>
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        ))}
      </div>
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
  const [chatMode, setChatMode]           = useState<ChatMode>('standard')
  const [sending, setSending]             = useState(false)
  const [chatError, setChatError]         = useState('')
  const [sessError, setSessError]         = useState('')
  const [generateOpen, setGenerateOpen]   = useState(false)
  const [activePaper, setActivePaper]     = useState<Paper | null>(null)
  const [modeSwitching, setModeSwitching] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  // OPTIMIZATION 2: Follow-up suggestions opt-in toggle, persisted to localStorage
  const [suggestionsEnabled, setSuggestionsEnabled] = useState(() =>
    getStoredBool('researchrag_suggestions_enabled', false)
  )
  function toggleSuggestions() {
    setSuggestionsEnabled(v => {
      const next = !v
      setStoredBool('researchrag_suggestions_enabled', next)
      return next
    })
  }

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
      setChatMode(res.session.mode ?? 'standard')
      try {
        const papersRes = await listPapers({ stage: 'processed' })
        const current = papersRes.papers.find(p => p.paper_id === res.session.paper_id)
        setActivePaper(current ?? initialPapers.find(p => p.paper_id === res.session.paper_id) ?? null)
      } catch { setActivePaper(initialPapers.find(p => p.paper_id === res.session.paper_id) ?? null) }
    } catch (err: unknown) { setSessError(err instanceof Error ? err.message : 'Failed to load session.') }
  }

  function handleSessionDeleted(sessionId: string) {
    setSessions(prev => prev.filter(s => s.session_id !== sessionId))
    if (activeSession?.session_id === sessionId) { setActiveSession(null); setMessages([]); setActivePaper(null) }
  }

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  async function handleModeSwitch(newMode: ChatMode) {
    if (!activePaper || newMode === chatMode || modeSwitching) return
    setModeSwitching(true); setSessError('')
    try {
      const existing = sessions.find(s => s.paper_id === activePaper.paper_id && s.mode === newMode)
      if (existing) {
        await openSession(existing.session_id)
      } else {
        const res = await createSession({
          paper_id: activePaper.paper_id,
          topic: activeSession?.topic ?? activePaper.title ?? activePaper.paper_id,
          level, mode: newMode,
        })
        setSessions(prev => [res.session as unknown as Session, ...prev])
        await openSession(res.session.session_id)
      }
      setChatMode(newMode)
    } catch (err: unknown) { setSessError(err instanceof Error ? err.message : 'Failed to switch mode.') }
    finally { setModeSwitching(false) }
  }

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
      setSessions(prev => prev.map(s => s.session_id === activeSession.session_id ? { ...s, last_active_at: new Date().toISOString() } : s))
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

  const paperIsReady    = activePaper?.pipeline_stage === 'processed' && (activePaper?.chunk_count ?? 0) > 0
  const activeModeConfig = CHAT_MODES.find(m => m.id === chatMode) ?? CHAT_MODES[0]

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <SessionSidebar sessions={sessions} activeSessionId={activeSession?.session_id ?? null}
        onSelect={openSession} onDelete={handleSessionDeleted} onNewSearch={onNewSearch} />

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>

        {/* Header */}
        <div style={{ padding: '12px 20px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'var(--bg-2)', flexShrink: 0, gap: '12px' }}>
          <div style={{ minWidth: 0 }}>
            <p style={{ fontWeight: 600, fontSize: '0.9375rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {activeSession?.title ?? activeSession?.topic ?? activePaper?.title ?? 'Select a session'}
            </p>
            {activePaper && <p style={{ fontSize: '0.75rem', color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>{activePaper.source} · {activePaper.chunk_count} chunks</p>}
            {sessError && <p style={{ fontSize: '0.8125rem', color: 'var(--error)' }}>{sessError}</p>}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexShrink: 0 }}>

            {/* Chat mode selector */}
            <div style={{ display: 'flex', background: 'var(--bg-3)', borderRadius: 'var(--radius)', padding: '3px', border: '1px solid var(--border)', gap: '2px' }}>
              {CHAT_MODES.map(m => {
                const isActive = chatMode === m.id
                return (
                  <button key={m.id} onClick={() => handleModeSwitch(m.id)}
                    disabled={modeSwitching || !activePaper} title={m.description}
                    style={{ padding: '5px 12px', borderRadius: '6px', border: 'none', cursor: modeSwitching || !activePaper ? 'not-allowed' : 'pointer', fontFamily: 'var(--font-sans)', fontSize: '0.8125rem', fontWeight: 500, transition: 'all 0.15s', background: isActive ? m.color + '22' : 'transparent', color: isActive ? m.color : 'var(--text-3)', opacity: modeSwitching ? 0.6 : 1 }}>
                    {modeSwitching && isActive ? <Spinner size="spinner-sm" /> : m.label}
                  </button>
                )
              })}
            </div>

            {/* Level selector — Chat mode only */}
            {chatMode === 'standard' && (
              <div style={{ display: 'flex', gap: '4px' }}>
                {(['beginner', 'intermediate', 'advanced'] as Level[]).map(l => (
                  <button key={l} onClick={() => handleLevelChange(l)}
                    style={{ padding: '4px 10px', borderRadius: '20px', border: `1px solid ${level === l ? 'var(--accent)' : 'var(--border)'}`, background: level === l ? 'var(--accent-glow)' : 'transparent', color: level === l ? 'var(--accent)' : 'var(--text-3)', fontFamily: 'var(--font-sans)', fontSize: '0.75rem', cursor: 'pointer', transition: 'all 0.12s', fontWeight: 500 }}>
                    {l.charAt(0).toUpperCase() + l.slice(1)}
                  </button>
                ))}
              </div>
            )}

            {/* Suggestions toggle — Chat mode only */}
            {chatMode === 'standard' && (
              <button onClick={toggleSuggestions} title={suggestionsEnabled ? 'Disable follow-up suggestions' : 'Enable follow-up suggestions'}
                style={{ padding: '4px 8px', borderRadius: 'var(--radius)', border: `1px solid ${suggestionsEnabled ? 'var(--accent)' : 'var(--border)'}`, background: suggestionsEnabled ? 'var(--accent-glow)' : 'transparent', color: suggestionsEnabled ? 'var(--accent)' : 'var(--text-3)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.75rem', transition: 'all 0.15s' }}>
                <IconSparkle />
                {suggestionsEnabled ? 'Suggestions on' : 'Suggestions'}
              </button>
            )}

            <button onClick={() => setGenerateOpen(o => !o)} className="btn btn-ghost btn-sm"
              style={{ borderColor: generateOpen ? 'var(--accent)' : undefined, color: generateOpen ? 'var(--accent)' : undefined }}>
              {generateOpen ? 'Hide generate' : 'Generate ✦'}
            </button>
          </div>
        </div>

        {/* Body */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0 }}>

            {/* Mode banner */}
            {activeSession && chatMode !== 'standard' && (
              <div style={{ padding: '8px 20px', background: `${activeModeConfig.color}11`, borderBottom: `1px solid ${activeModeConfig.color}33`, display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: activeModeConfig.color, flexShrink: 0 }} />
                <span style={{ fontSize: '0.8125rem', color: activeModeConfig.color, fontWeight: 500 }}>{activeModeConfig.label} mode</span>
                <span style={{ fontSize: '0.8125rem', color: 'var(--text-3)' }}>— {activeModeConfig.description}</span>
              </div>
            )}

            {/* Study panel */}
            {chatMode === 'study' && activePaper && (
              <div style={{ flex: 1, overflowY: 'auto' }}><StudyPanel paperId={activePaper.paper_id} /></div>
            )}

            {/* Technical panel */}
            {chatMode === 'technical' && activePaper && (
              <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                <TechnicalPanel paperId={activePaper.paper_id} />
              </div>
            )}

            {/* Standard chat */}
            {chatMode === 'standard' && (
              <>
                <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
                  {!activeSession && (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                      <p style={{ color: 'var(--text-3)', textAlign: 'center', fontSize: '0.9375rem' }}>
                        {sessions.length === 0 ? 'Waiting for papers to process…' : 'Select a session from the sidebar'}
                      </p>
                    </div>
                  )}
                  {messages.map((msg, i) => (
                    <div key={msg.id ?? i} className="fade-in"
                      style={{ display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start', marginBottom: '14px' }}>
                      <div style={{ maxWidth: '72%', padding: '10px 14px', borderRadius: 'var(--radius-lg)', background: msg.role === 'user' ? 'var(--accent)' : 'var(--bg-3)', color: msg.role === 'user' ? '#fff' : 'var(--text)', fontSize: '0.9rem', lineHeight: 1.6, borderBottomRightRadius: msg.role === 'user' ? '4px' : undefined, borderBottomLeftRadius: msg.role === 'assistant' ? '4px' : undefined }}>
                        {msg.role === 'assistant'
                          ? <MarkdownWithMermaid content={msg.content} />
                          : msg.content
                        }
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
                    <FollowUpSuggestions
                      lastResponse={lastAssistantMessage}
                      onSelect={q => submitMessage(q)}
                      enabled={suggestionsEnabled}
                    />
                  )}
                  <div ref={bottomRef} />
                </div>

                <div style={{ padding: '14px 20px', borderTop: '1px solid var(--border)', background: 'var(--bg-2)', flexShrink: 0 }}>
                  {chatError && <div className="notice notice-error" style={{ marginBottom: '10px' }}>{chatError}</div>}
                  <form onSubmit={handleSend} style={{ display: 'flex', gap: '8px' }}>
                    <input className="input" value={input} onChange={e => setInput(e.target.value)}
                      placeholder={!activeSession ? 'Waiting for a session…' : 'Ask anything about this paper…'}
                      disabled={!activeSession || sending} maxLength={2000} style={{ flex: 1 }} />
                    <button type="submit" className="btn btn-primary"
                      disabled={!activeSession || !input.trim() || sending} style={{ flexShrink: 0 }}>
                      <IconSend />
                    </button>
                  </form>
                </div>
              </>
            )}

            {(chatMode === 'study' || chatMode === 'technical') && !activePaper && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
                <p style={{ color: 'var(--text-3)', textAlign: 'center', fontSize: '0.9375rem' }}>Select a session from the sidebar to begin.</p>
              </div>
            )}
          </div>

          {/* Generate panel */}
          {generateOpen && (
            <div className="fade-in" style={{ width: '300px', flexShrink: 0, borderLeft: '1px solid var(--border)', overflowY: 'auto', background: 'var(--bg)', display: 'flex', flexDirection: 'column' }}>
              {paperIsReady
                ? <GenerationPanel paperId={activePaper!.paper_id} />
                : <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '120px', padding: '20px' }}><p style={{ color: 'var(--text-3)', fontSize: '0.8125rem', textAlign: 'center', lineHeight: 1.6 }}>Paper must finish processing before you can generate content.</p></div>
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
  const [view, setView]                   = useState<View>('search')
  const [searchResults, setSearchResults] = useState<PaperPreview[]>([])
  const [papers, setPapers]               = useState<Paper[]>([])

  function handleSearchResults(results: PaperPreview[]) {
    setSearchResults(results)
    if (results.length === 1 && results[0].source === 'local') {
      setPapers([{ paper_id: results[0].paper_id, source: 'local', title: results[0].title, abstract: results[0].abstract, authors: results[0].authors, url: results[0].url, pipeline_stage: 'pending', chunk_count: 0, error_message: null, topic: null, created_at: null, processed_at: null }])
      setView('processing')
    } else { setView('selection') }
  }

  function handleSelection(selected: PaperPreview[]) {
    setPapers(selected.map(p => ({ paper_id: p.paper_id, source: p.source, title: p.title, abstract: p.abstract, authors: p.authors, url: p.url, pipeline_stage: 'pending' as const, chunk_count: 0, error_message: null, topic: null, created_at: null, processed_at: null })))
    setView('processing')
  }

  return (
    <>
      {view === 'search'     && <SearchView onResults={handleSearchResults} />}
      {view === 'selection'  && <SelectionView papers={searchResults} onSelect={handleSelection} onBack={() => setView('search')} />}
      {view === 'processing' && <ProcessingView papers={papers} onDone={p => { setPapers(p); setView('chat') }} />}
      {view === 'chat'       && <ChatView initialPapers={papers} onNewSearch={() => { setSearchResults([]); setView('search') }} />}
    </>
  )
}
