"use client";

import { useState, useEffect, useRef } from "react";

// ── Types ──────────────────────────────────────────────────────────────────────

interface Paper {
  paper_id: string;
  title: string;
  authors: string[];
  abstract: string;
  source: "arxiv" | "pubmed";
  pipeline_stage: string;
  chunk_count: number;
  url?: string;
  cached?: boolean;
  download_error?: string;
  processing_error?: string;
}

interface Session {
  session_id: string;
  paper_id: string;
  topic: string;
  title: string;
  level: string;
  last_active_at: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  level?: string;
}

interface SearchFilters {
  topic: string;
  source: "both" | "arxiv" | "pubmed";
  limit: number;
  date_from: string;
}

type AppView = "search" | "loading" | "chat";
type Level = "beginner" | "intermediate" | "advanced";
type Platform = "twitter" | "linkedin" | "carousel";

// ── Constants ──────────────────────────────────────────────────────────────────

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const LEVELS: Record<Level, { icon: string; label: string; desc: string }> = {
  beginner:     { icon: "◎", label: "Beginner",     desc: "Plain language & analogies" },
  intermediate: { icon: "◈", label: "Intermediate", desc: "Field concepts explained"   },
  advanced:     { icon: "◆", label: "Advanced",     desc: "Methodology & nuance"       },
};

const CAROUSEL_PRESETS = [
  { id: "emerald_dark",  label: "Dark Emerald",  colors: ["#0A1628", "#00C896", "#E8F5F0"] },
  { id: "emerald_light", label: "Light Emerald", colors: ["#F0FAF6", "#00916E", "#0A2E1E"] },
  { id: "emerald_bold",  label: "Bold Neon",     colors: ["#011A10", "#00FF9D", "#FFFFFF"] },
];

// ── Helpers ────────────────────────────────────────────────────────────────────

function reltime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

// ── Generate Modal ─────────────────────────────────────────────────────────────

function GenerateModal({ paper, onClose }: { paper: Paper; onClose: () => void }) {
  const [platform, setPlatform] = useState<Platform>("linkedin");
  const [style, setStyle]       = useState("educational");
  const [preset, setPreset]     = useState("emerald_dark");
  const [generating, setGenerating] = useState(false);
  const [result, setResult]     = useState<any>(null);
  const [error, setError]       = useState("");
  const [copied, setCopied]     = useState(false);

  async function generate() {
    setGenerating(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${API}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          paper_id: paper.paper_id,
          platform,
          content_style: style,
          carousel_style: platform === "carousel" ? { preset } : undefined,
        }),
      });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.detail || "Generation failed");
      }
      const { queue_key } = await res.json();

      const es = new EventSource(`${API}/api/generate/${queue_key}/progress`);
      es.addEventListener("completed", (e: MessageEvent) => {
        setResult(JSON.parse(e.data).content);
        setGenerating(false);
        es.close();
      });
      es.addEventListener("failed", (e: MessageEvent) => {
        setError(JSON.parse(e.data).error || "Generation failed");
        setGenerating(false);
        es.close();
      });
      es.addEventListener("done", () => es.close());
    } catch (e: any) {
      setError(e.message);
      setGenerating(false);
    }
  }

  function copyAll() {
    let text = "";
    if (platform === "twitter" && result?.tweets) text = result.tweets.join("\n\n");
    else if (platform === "linkedin" && result?.content)
      text = `${result.content}\n\n${(result.hashtags || []).join(" ")}`;
    if (text) {
      navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }

  return (
    <div className="modal-overlay" onClick={e => {
      if ((e.target as HTMLElement).classList.contains("modal-overlay")) onClose();
    }}>
      <div className="modal-panel">
        <div className="modal-hdr">
          <div>
            <div className="modal-eyebrow">Generate Content</div>
            <div className="modal-ptitle">{paper.title || paper.paper_id}</div>
          </div>
          <button className="icon-btn" onClick={onClose}>✕</button>
        </div>

        {/* Platform selector */}
        <div className="modal-sec">
          <div className="modal-sec-label">Platform</div>
          <div className="pill-row">
            {(["linkedin","twitter","carousel"] as Platform[]).map(p => (
              <button key={p} className={`pill ${platform === p ? "active" : ""}`}
                onClick={() => setPlatform(p)}>
                {p === "twitter" ? "𝕏 Twitter Thread"
                  : p === "linkedin" ? "in LinkedIn Post"
                  : "▦ Carousel"}
              </button>
            ))}
          </div>
        </div>

        {/* Style selector */}
        <div className="modal-sec">
          <div className="modal-sec-label">Content Style</div>
          <div className="pill-row">
            {["educational","professional","minimal","bold"].map(s => (
              <button key={s} className={`pill sm ${style === s ? "active" : ""}`}
                onClick={() => setStyle(s)}>
                {s.charAt(0).toUpperCase() + s.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Carousel preset */}
        {platform === "carousel" && (
          <div className="modal-sec">
            <div className="modal-sec-label">Slide Style</div>
            <div className="pill-row">
              {CAROUSEL_PRESETS.map(cp => (
                <button key={cp.id} className={`preset-pill ${preset === cp.id ? "active" : ""}`}
                  onClick={() => setPreset(cp.id)}>
                  <div className="preset-swatches">
                    {cp.colors.map((c,i) => (
                      <div key={i} style={{ background: c, width:12, height:12, borderRadius:3, border:"1px solid rgba(255,255,255,0.1)" }} />
                    ))}
                  </div>
                  {cp.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {error && <div className="modal-error">{error}</div>}

        {/* Result */}
        {result && (
          <div className="modal-result">
            {platform === "twitter" && result.tweets && (
              <>
                <div className="tweet-list">
                  {result.tweets.map((t: string, i: number) => (
                    <div key={i} className="tweet-card">
                      <p className="tweet-text">{t}</p>
                      <span className="tweet-chars">{t.length}/280</span>
                    </div>
                  ))}
                </div>
                {result.hashtags?.length > 0 && (
                  <div className="tag-row">
                    {result.hashtags.map((h: string) => <span key={h} className="tag">{h}</span>)}
                  </div>
                )}
                <button className="copy-all-btn" onClick={copyAll}>
                  {copied ? "✓ Copied" : "Copy all tweets"}
                </button>
              </>
            )}
            {platform === "linkedin" && result.content && (
              <>
                <div className="li-text">{result.content}</div>
                {result.hashtags?.length > 0 && (
                  <div className="tag-row">
                    {result.hashtags.map((h: string) => <span key={h} className="tag">{h}</span>)}
                  </div>
                )}
                <button className="copy-all-btn" onClick={copyAll}>
                  {copied ? "✓ Copied" : "Copy post"}
                </button>
              </>
            )}
            {platform === "carousel" && (
              <div className="carousel-done">
                <span className="carousel-done-icon">▦</span>
                <span>{result.slides?.length || 0} slides generated</span>
              </div>
            )}
          </div>
        )}

        <button className="generate-btn" onClick={generate} disabled={generating}>
          {generating
            ? <><span className="spinner" /> Generating with Gemini…</>
            : `Generate ${platform.charAt(0).toUpperCase() + platform.slice(1)}`}
        </button>
      </div>
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────────

export default function App() {
  const [view, setView]                 = useState<AppView>("search");
  const [sessions, setSessions]         = useState<Session[]>([]);
  const [activeSession, setActiveSession] = useState<Session | null>(null);
  const [messages, setMessages]         = useState<Message[]>([]);
  const [activePaper, setActivePaper]   = useState<Paper | null>(null);
  const [sidebarOpen, setSidebarOpen]   = useState(true);
  const [generateOpen, setGenerateOpen] = useState(false);

  const [filters, setFilters] = useState<SearchFilters>({
    topic: "", source: "both", limit: 5, date_from: "",
  });
  const [searching, setSearching]       = useState(false);
  const [searchResults, setSearchResults] = useState<Paper[]>([]);
  const [searchError, setSearchError]   = useState("");
  const [paperStages, setPaperStages]   = useState<Record<string, string>>({});

  const [input, setInput]     = useState("");
  const [level, setLevel]     = useState<Level>("beginner");
  const [sending, setSending] = useState(false);
  const [chatError, setChatError] = useState("");

  const bottomRef  = useRef<HTMLDivElement>(null);
  const inputRef   = useRef<HTMLTextAreaElement>(null);
  const esRefs     = useRef<Record<string, EventSource>>({});

  useEffect(() => { fetchSessions(); }, []);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function fetchSessions() {
    try {
      const d = await fetch(`${API}/api/chat/sessions`).then(r => r.json());
      setSessions(d.sessions || []);
    } catch {}
  }

  // ── Search ────────────────────────────────────────────────────────────────

  async function handleSearch() {
    const t = filters.topic.trim();
    if (t.length < 3) { setSearchError("Enter at least 3 characters."); return; }
    setSearching(true);
    setSearchError("");
    setView("loading");
    setSearchResults([]);
    setPaperStages({});

    try {
      const res = await fetch(`${API}/api/papers/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: t,
          limit: filters.limit,
          source: filters.source,
          date_from: filters.date_from || undefined,
        }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || "Search failed");
      const data = await res.json();
      const papers: Paper[] = data.papers || [];

      if (!papers.length) {
        setSearchError("No papers found. Try a different topic.");
        setView("search");
        setSearching(false);
        return;
      }

      setSearchResults(papers);

      for (const p of papers) {
        if (p.cached || p.pipeline_stage === "processed") {
          setPaperStages(s => ({ ...s, [p.paper_id]: "processed" }));
        } else {
          setPaperStages(s => ({ ...s, [p.paper_id]: "pending" }));
          listenSSE(p.paper_id);
        }
      }
    } catch (e: any) {
      setSearchError(e.message || "Search failed.");
      setView("search");
    } finally {
      setSearching(false);
    }
  }

  function listenSSE(paperId: string) {
    esRefs.current[paperId]?.close();
    const es = new EventSource(`${API}/api/papers/${paperId}/progress`);
    esRefs.current[paperId] = es;

    es.addEventListener("progress", (e: MessageEvent) => {
      const d = JSON.parse(e.data);
      setPaperStages(s => ({ ...s, [paperId]: `${d.stage} — ${d.status}` }));
    });

    es.addEventListener("done", (e: MessageEvent) => {
      const d = JSON.parse(e.data);
      es.close();
      delete esRefs.current[paperId];
      if (d.success) {
        setPaperStages(s => ({ ...s, [paperId]: "processed" }));
        setSearchResults(prev =>
          prev.map(p => p.paper_id === paperId ? { ...p, pipeline_stage: "processed" } : p)
        );
        if (d.session_id) {
          openSession(d.session_id);
          fetchSessions();
        }
      } else {
        setPaperStages(s => ({ ...s, [paperId]: `failed: ${d.error}` }));
      }
    });
    es.onerror = () => { es.close(); delete esRefs.current[paperId]; };
  }

  // ── Chat ──────────────────────────────────────────────────────────────────

  async function openSession(sessionId: string) {
    try {
      const d = await fetch(`${API}/api/chat/sessions/${sessionId}`).then(r => r.json());
      setActiveSession({
        session_id: d.session_id,
        paper_id: d.paper_id,
        topic: d.topic,
        title: d.paper_title || d.topic,
        level: d.level,
        last_active_at: new Date().toISOString(),
      });
      setMessages(d.messages || []);
      setLevel((d.level as Level) || "beginner");
      setView("chat");

      // Load paper details
      try {
        const pd = await fetch(`${API}/api/papers`).then(r => r.json());
        const found = pd.papers?.find((p: Paper) => p.paper_id === d.paper_id);
        if (found) setActivePaper(found);
      } catch {}

      setTimeout(() => inputRef.current?.focus(), 100);
    } catch {}
  }

  async function sendMessage() {
    if (!input.trim() || !activeSession || sending) return;
    const msg = input.trim();
    setInput("");
    setSending(true);
    setChatError("");

    setMessages(prev => [...prev, { role: "user", content: msg, level }]);

    try {
      const res = await fetch(
        `${API}/api/chat/sessions/${activeSession.session_id}/message`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: msg, level }),
        }
      );
      if (!res.ok) throw new Error((await res.json()).detail || "Message failed");
      const d = await res.json();
      setMessages(prev => [...prev, { role: "assistant", content: d.content, level: d.level }]);
    } catch (e: any) {
      setChatError(e.message);
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }

  const allReady = searchResults.length > 0 &&
    searchResults.every(p =>
      paperStages[p.paper_id] === "processed" ||
      (paperStages[p.paper_id] || "").startsWith("failed")
    );

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <>
      <div className="app">

        {/* Sidebar */}
        <aside className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
          <div className="sidebar-top">
            <div className="logo">
              <span className="logo-gem">⬡</span>
              {sidebarOpen && <span className="logo-name">ResearchRAG</span>}
            </div>
            <button className="icon-btn toggle-btn" onClick={() => setSidebarOpen(o => !o)}>
              {sidebarOpen ? "‹" : "›"}
            </button>
          </div>

          <button className="new-btn" onClick={() => { setView("search"); setActiveSession(null); }}>
            <span className="new-icon">+</span>
            {sidebarOpen && <span>New Topic</span>}
          </button>

          {sidebarOpen && (
            <div className="sessions">
              {sessions.length === 0
                ? <div className="sessions-empty">Search a topic to begin</div>
                : sessions.map(s => (
                  <div
                    key={s.session_id}
                    className={`session ${activeSession?.session_id === s.session_id ? "active" : ""}`}
                    onClick={() => openSession(s.session_id)}
                  >
                    <div className="session-title">{s.title || s.topic}</div>
                    <div className="session-meta">
                      <span className={`level-pip lv-${s.level}`}>
                        {LEVELS[s.level as Level]?.icon}
                      </span>
                      <span>{reltime(s.last_active_at)}</span>
                    </div>
                  </div>
                ))
              }
            </div>
          )}
        </aside>

        {/* Main */}
        <main className="main">

          {/* ═══ SEARCH ═══ */}
          {view === "search" && (
            <div className="search-page">
              <div className="search-hero">
                <p className="hero-eyebrow">Research · Learn · Create</p>
                <h1 className="hero-title">Research, Simplified.</h1>
                <p className="hero-sub">
                  Enter a topic. We'll find the papers, teach them to you conversationally,
                  and turn insights into ready-to-post content.
                </p>
              </div>

              <div className="search-box">
                <div className="search-input-row">
                  <span className="search-icon-sym">⌕</span>
                  <input
                    className="search-input"
                    placeholder="e.g. CRISPR gene editing, transformer attention, quantum error correction…"
                    value={filters.topic}
                    onChange={e => setFilters(f => ({ ...f, topic: e.target.value }))}
                    onKeyDown={e => e.key === "Enter" && handleSearch()}
                    autoFocus
                  />
                </div>

                <div className="filter-row">
                  <div className="filter-item">
                    <label className="filter-lbl">Source</label>
                    <select className="filter-sel" value={filters.source}
                      onChange={e => setFilters(f => ({ ...f, source: e.target.value as any }))}>
                      <option value="both">arXiv + PubMed</option>
                      <option value="arxiv">arXiv only</option>
                      <option value="pubmed">PubMed only</option>
                    </select>
                  </div>
                  <div className="filter-item">
                    <label className="filter-lbl">Results</label>
                    <select className="filter-sel" value={filters.limit}
                      onChange={e => setFilters(f => ({ ...f, limit: Number(e.target.value) }))}>
                      {[3, 5, 8, 10].map(n => <option key={n} value={n}>{n} per source</option>)}
                    </select>
                  </div>
                  <div className="filter-item">
                    <label className="filter-lbl">From date</label>
                    <input className="filter-sel" type="date" value={filters.date_from}
                      onChange={e => setFilters(f => ({ ...f, date_from: e.target.value }))} />
                  </div>
                </div>

                {searchError && <div className="search-err">{searchError}</div>}

                <button className="search-btn" onClick={handleSearch} disabled={searching}>
                  {searching
                    ? <><span className="spinner" /> Searching…</>
                    : "Search & Learn →"}
                </button>
              </div>
            </div>
          )}

          {/* ═══ LOADING ═══ */}
          {view === "loading" && (
            <div className="loading-page">
              <div className="loading-head">
                <h2 className="loading-title">
                  Processing <em>"{filters.topic}"</em>
                </h2>
                <p className="loading-sub">
                  Downloading papers · Extracting text · Building embeddings
                </p>
              </div>

              <div className="progress-list">
                {searchResults.map(paper => {
                  const stage = paperStages[paper.paper_id] || "pending";
                  const isDone   = stage === "processed";
                  const isFailed = stage.startsWith("failed");
                  return (
                    <div key={paper.paper_id}
                      className={`prog-row ${isDone ? "prog-done" : isFailed ? "prog-fail" : "prog-run"}`}>
                      <div className="prog-left">
                        <span className={`src-dot src-${paper.source}`} />
                        <div className="prog-info">
                          <div className="prog-title">{paper.title || paper.paper_id}</div>
                          <div className="prog-authors">
                            {(paper.authors || []).slice(0, 2).join(", ")}
                          </div>
                        </div>
                      </div>
                      <div className="prog-right">
                        {isDone
                          ? <span className="prog-badge done">✓ Ready</span>
                          : isFailed
                          ? <span className="prog-badge fail">✗ Failed</span>
                          : <span className="prog-badge running">
                              <span className="spinner-sm" /> {stage}
                            </span>
                        }
                      </div>
                    </div>
                  );
                })}
              </div>

              {allReady && (
                <div className="loading-done">
                  <div className="loading-done-msg">
                    ✓ All papers ready. Chat opened automatically.
                  </div>
                  <button className="back-btn" onClick={() => setView("search")}>
                    ← Search again
                  </button>
                </div>
              )}
            </div>
          )}

          {/* ═══ CHAT ═══ */}
          {view === "chat" && activeSession && (
            <div className="chat-page">
              {/* Chat header */}
              <div className="chat-hdr">
                <div className="chat-hdr-left">
                  <div className="chat-paper-title">
                    {activePaper?.title || activeSession.title || activeSession.topic}
                  </div>
                  <div className="chat-paper-meta">
                    {activePaper && (
                      <>
                        <span className={`src-badge src-${activePaper.source}`}>
                          {activePaper.source}
                        </span>
                        <span className="chat-meta-divider">·</span>
                        <span>{activePaper.chunk_count} chunks</span>
                      </>
                    )}
                  </div>
                </div>
                <div className="chat-hdr-right">
                  {/* Level selector */}
                  <div className="level-tabs">
                    {(Object.entries(LEVELS) as [Level, any][]).map(([lv, info]) => (
                      <button
                        key={lv}
                        className={`level-tab ${level === lv ? "active" : ""}`}
                        onClick={() => setLevel(lv)}
                        title={info.desc}
                      >
                        <span>{info.icon}</span>
                        <span>{info.label}</span>
                      </button>
                    ))}
                  </div>
                  {/* Generate button */}
                  {activePaper?.pipeline_stage === "processed" && (
                    <button className="gen-btn" onClick={() => setGenerateOpen(true)}>
                      ✦ Generate Content
                    </button>
                  )}
                </div>
              </div>

              {/* Messages */}
              <div className="messages">
                {messages.length === 0 && (
                  <div className="chat-welcome">
                    <div className="welcome-icon">◎</div>
                    <div className="welcome-title">
                      Ready to teach you about this paper
                    </div>
                    <div className="welcome-sub">
                      Ask anything — "What does this paper do?", "Explain the methodology",
                      "What are the key findings?", "Give me an analogy"
                    </div>
                    <div className="welcome-starters">
                      {[
                        "What is this paper about?",
                        "What problem does it solve?",
                        "Explain the key methodology",
                        "What are the main findings?",
                      ].map(q => (
                        <button key={q} className="starter-btn"
                          onClick={() => { setInput(q); inputRef.current?.focus(); }}>
                          {q}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {messages.map((msg, i) => (
                  <div key={i} className={`msg msg-${msg.role}`}>
                    {msg.role === "assistant" && (
                      <div className="msg-avatar">◎</div>
                    )}
                    <div className="msg-bubble">
                      <div className="msg-content">{msg.content}</div>
                      {msg.role === "assistant" && msg.level && (
                        <div className="msg-level">
                          {LEVELS[msg.level as Level]?.icon} {msg.level}
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {sending && (
                  <div className="msg msg-assistant">
                    <div className="msg-avatar">◎</div>
                    <div className="msg-bubble typing">
                      <span /><span /><span />
                    </div>
                  </div>
                )}

                {chatError && (
                  <div className="chat-error">{chatError}</div>
                )}

                <div ref={bottomRef} />
              </div>

              {/* Input */}
              <div className="chat-input-area">
                <div className="chat-input-box">
                  <textarea
                    ref={inputRef}
                    className="chat-input"
                    placeholder={`Ask about this paper at ${level} level…`}
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                    rows={1}
                    style={{ resize: "none" }}
                  />
                  <button
                    className="send-btn"
                    onClick={sendMessage}
                    disabled={sending || !input.trim()}
                  >
                    {sending ? <span className="spinner" /> : "↑"}
                  </button>
                </div>
                <div className="input-hint">Enter to send · Shift+Enter for new line</div>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Generate Modal */}
      {generateOpen && activePaper && (
        <GenerateModal
          paper={activePaper}
          onClose={() => setGenerateOpen(false)}
        />
      )}
    </>
  );
}