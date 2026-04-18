import { useEffect, useState } from 'react'

type ModelInfo = {
  name: string
  n_layers: number
  n_heads: number
  d_model: number
  d_vocab: number
  device: string
}

type Health = { loaded: boolean; info: ModelInfo | null; error: string | null }

async function fetchModelInfo(): Promise<Health> {
  try {
    const res = await fetch('/api/model/info')
    if (res.status === 404) return { loaded: false, info: null, error: null }
    if (!res.ok) return { loaded: false, info: null, error: `HTTP ${res.status}` }
    const info = (await res.json()) as ModelInfo
    return { loaded: true, info, error: null }
  } catch (e) {
    return { loaded: false, info: null, error: (e as Error).message }
  }
}

function App() {
  const [health, setHealth] = useState<Health>({ loaded: false, info: null, error: null })
  const [polling, setPolling] = useState(true)

  useEffect(() => {
    if (!polling) return
    let cancelled = false
    const tick = async () => {
      const h = await fetchModelInfo()
      if (!cancelled) setHealth(h)
    }
    tick()
    const id = setInterval(tick, 2000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [polling])

  const statusColor = health.error
    ? 'bg-red-500'
    : health.loaded
      ? 'bg-emerald-500'
      : 'bg-amber-500'

  const statusLabel = health.error
    ? 'backend unreachable'
    : health.loaded
      ? 'model loaded'
      : 'no model loaded'

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center gap-4">
        <h1 className="text-xl font-medium tracking-tight">llm-scry</h1>
        <span className="text-xs text-zinc-500">interpretability visualizer</span>
        <div className="ml-auto flex items-center gap-2 text-sm">
          <span className={`inline-block w-2 h-2 rounded-full ${statusColor}`} />
          <span className="text-zinc-400">{statusLabel}</span>
          <button
            onClick={() => setPolling((p) => !p)}
            className="ml-2 text-xs px-2 py-1 rounded border border-zinc-700 hover:bg-zinc-800"
          >
            {polling ? 'pause' : 'resume'} polling
          </button>
        </div>
      </header>

      <main className="flex-1 p-6">
        <section className="max-w-3xl">
          <h2 className="text-sm uppercase tracking-wider text-zinc-500 mb-2">
            Phase 0 — backend health
          </h2>
          {health.error && (
            <p className="text-red-400 text-sm mono">{health.error}</p>
          )}
          {!health.error && !health.loaded && (
            <p className="text-sm text-zinc-400">
              Backend reachable but no model is loaded. Load one with:
              <pre className="mt-2 p-3 bg-zinc-900 border border-zinc-800 rounded text-xs overflow-x-auto">
{`curl -X POST localhost:8000/model/load \\
  -H "content-type: application/json" \\
  -d '{"name":"gpt2"}'`}
              </pre>
            </p>
          )}
          {health.loaded && health.info && (
            <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm mono">
              <dt className="text-zinc-500">name</dt><dd>{health.info.name}</dd>
              <dt className="text-zinc-500">device</dt><dd>{health.info.device}</dd>
              <dt className="text-zinc-500">n_layers</dt><dd>{health.info.n_layers}</dd>
              <dt className="text-zinc-500">n_heads</dt><dd>{health.info.n_heads}</dd>
              <dt className="text-zinc-500">d_model</dt><dd>{health.info.d_model}</dd>
              <dt className="text-zinc-500">d_vocab</dt><dd>{health.info.d_vocab}</dd>
            </dl>
          )}
        </section>

        <section className="max-w-3xl mt-10">
          <h2 className="text-sm uppercase tracking-wider text-zinc-500 mb-2">
            Next
          </h2>
          <p className="text-sm text-zinc-400">
            Phase 1 will wire up a prompt panel, streaming token view, and top-k inspector.
            See <code className="mono text-xs">docs/design.md</code>.
          </p>
        </section>
      </main>
    </div>
  )
}

export default App
