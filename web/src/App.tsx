import { Routes, Route } from 'react-router-dom'
import { Layout } from '@/components/layout'
import {
  Dashboard,
  Repositories,
  Explorer,
  Chat,
  Graph,
  Analytics,
} from '@/pages'
import { TooltipProvider } from '@/components/ui/tooltip'

function App() {
  return (
    <TooltipProvider>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="repositories" element={<Repositories />} />
          <Route path="explorer" element={<Explorer />} />
          <Route path="chat" element={<Chat />} />
          <Route path="graph" element={<Graph />} />
          <Route path="analytics" element={<Analytics />} />
        </Route>
      </Routes>
    </TooltipProvider>
  )
}

export default App
