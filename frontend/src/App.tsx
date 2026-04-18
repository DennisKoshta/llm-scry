import { ModelBar } from './components/ModelBar'
import { PromptPanel } from './components/PromptPanel'
import { SessionSidebar } from './components/SessionSidebar'
import { TokenDetail } from './components/TokenDetail'
import { TokenStream } from './components/TokenStream'

function App() {
  return (
    <div className="h-screen flex flex-col">
      <ModelBar />
      <div
        className="flex-1 grid overflow-hidden"
        style={{ gridTemplateColumns: '280px 1fr 380px' }}
      >
        <SessionSidebar />
        <main className="flex flex-col overflow-hidden border-r border-zinc-800">
          <PromptPanel />
          <TokenStream />
        </main>
        <aside className="overflow-hidden flex flex-col">
          <TokenDetail />
        </aside>
      </div>
    </div>
  )
}

export default App
