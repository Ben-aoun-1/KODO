import { create } from 'zustand'
import type {
  Repository,
  ChatMessage,
  CodeEntity,
  CallGraph,
} from '@/types'
import api from '@/api/client'

interface AppState {
  // Repository state
  repositories: Repository[]
  currentRepo: Repository | null
  isLoadingRepos: boolean
  repoError: string | null

  // Chat state
  messages: ChatMessage[]
  isQuerying: boolean

  // Code explorer state
  selectedEntity: CodeEntity | null
  callGraph: CallGraph | null
  isLoadingGraph: boolean

  // Theme
  theme: 'light' | 'dark'

  // Actions
  fetchRepositories: () => Promise<void>
  selectRepository: (repo: Repository) => void
  indexRepository: (url: string) => Promise<void>
  sendMessage: (message: string) => Promise<void>
  selectEntity: (entity: CodeEntity | null) => void
  loadCallGraph: (entityId: string, depth?: number) => Promise<void>
  toggleTheme: () => void
  clearMessages: () => void
}

export const useAppStore = create<AppState>((set, get) => ({
  // Initial state
  repositories: [],
  currentRepo: null,
  isLoadingRepos: false,
  repoError: null,
  messages: [],
  isQuerying: false,
  selectedEntity: null,
  callGraph: null,
  isLoadingGraph: false,
  theme: (localStorage.getItem('theme') as 'light' | 'dark') || 'light',

  // Repository actions
  fetchRepositories: async () => {
    set({ isLoadingRepos: true, repoError: null })
    try {
      const repositories = await api.getRepositories()
      set({ repositories, isLoadingRepos: false })
    } catch (error) {
      set({
        repoError: error instanceof Error ? error.message : 'Failed to load repositories',
        isLoadingRepos: false,
      })
    }
  },

  selectRepository: (repo: Repository) => {
    set({ currentRepo: repo, messages: [], selectedEntity: null, callGraph: null })
  },

  indexRepository: async (url: string) => {
    set({ isLoadingRepos: true, repoError: null })
    try {
      const repo = await api.indexRepository(url)
      set((state) => ({
        repositories: [...state.repositories, repo],
        currentRepo: repo,
        isLoadingRepos: false,
      }))
    } catch (error) {
      set({
        repoError: error instanceof Error ? error.message : 'Failed to index repository',
        isLoadingRepos: false,
      })
    }
  },

  // Chat actions
  sendMessage: async (content: string) => {
    const { currentRepo, messages } = get()
    if (!currentRepo) return

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: new Date(),
    }

    set({ messages: [...messages, userMessage], isQuerying: true })

    try {
      const response = await api.ask(currentRepo.id, content)
      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
        sources: response.sources,
      }
      set((state) => ({
        messages: [...state.messages, assistantMessage],
        isQuerying: false,
      }))
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to get response'}`,
        timestamp: new Date(),
      }
      set((state) => ({
        messages: [...state.messages, errorMessage],
        isQuerying: false,
      }))
    }
  },

  clearMessages: () => {
    set({ messages: [] })
  },

  // Code explorer actions
  selectEntity: (entity: CodeEntity | null) => {
    set({ selectedEntity: entity })
  },

  loadCallGraph: async (entityId: string, depth = 2) => {
    const { currentRepo } = get()
    if (!currentRepo) return

    set({ isLoadingGraph: true })
    try {
      const callGraph = await api.getCallGraph(currentRepo.id, entityId, depth)
      set({ callGraph, isLoadingGraph: false })
    } catch (error) {
      console.error('Failed to load call graph:', error)
      set({ isLoadingGraph: false })
    }
  },

  // Theme actions
  toggleTheme: () => {
    set((state) => {
      const newTheme = state.theme === 'light' ? 'dark' : 'light'
      localStorage.setItem('theme', newTheme)
      document.documentElement.classList.toggle('dark', newTheme === 'dark')
      return { theme: newTheme }
    })
  },
}))

// Initialize theme on load
if (typeof window !== 'undefined') {
  const theme = localStorage.getItem('theme') || 'light'
  document.documentElement.classList.toggle('dark', theme === 'dark')
}
