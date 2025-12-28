import axios, { AxiosInstance, AxiosError } from 'axios'
import type {
  Repository,
  QueryRequest,
  QueryResponse,
  CodeEntity,
  FileNode,
  CallGraph,
  ImpactAnalysis,
  SearchResult,
} from '@/types'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

class ApiClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        const message = this.getErrorMessage(error)
        return Promise.reject(new Error(message))
      }
    )
  }

  private getErrorMessage(error: AxiosError): string {
    if (error.response?.data && typeof error.response.data === 'object') {
      const data = error.response.data as Record<string, unknown>
      if (typeof data.detail === 'string') {
        return data.detail
      }
      if (typeof data.message === 'string') {
        return data.message
      }
    }
    return error.message || 'An unexpected error occurred'
  }

  // Repository endpoints
  async getRepositories(): Promise<Repository[]> {
    const response = await this.client.get('/repositories')
    return response.data
  }

  async getRepository(id: string): Promise<Repository> {
    const response = await this.client.get(`/repositories/${id}`)
    return response.data
  }

  async indexRepository(url: string): Promise<Repository> {
    const response = await this.client.post('/repositories/index', { url })
    return response.data
  }

  async refreshRepository(id: string): Promise<Repository> {
    const response = await this.client.post(`/repositories/${id}/refresh`)
    return response.data
  }

  async deleteRepository(id: string): Promise<void> {
    await this.client.delete(`/repositories/${id}`)
  }

  // File tree
  async getFileTree(repoId: string): Promise<FileNode> {
    const response = await this.client.get(`/repositories/${repoId}/files`)
    return response.data
  }

  // Entity endpoints
  async getEntity(repoId: string, entityId: string): Promise<CodeEntity> {
    const response = await this.client.get(
      `/repositories/${repoId}/entities/${entityId}`
    )
    return response.data
  }

  async getEntitiesInFile(repoId: string, filePath: string): Promise<CodeEntity[]> {
    const response = await this.client.get(
      `/repositories/${repoId}/files/${encodeURIComponent(filePath)}/entities`
    )
    return response.data
  }

  // Query endpoints
  async query(request: QueryRequest): Promise<QueryResponse> {
    const response = await this.client.post('/query', request)
    return response.data
  }

  async ask(repoId: string, question: string): Promise<QueryResponse> {
    return this.query({ repoId, question })
  }

  // Search
  async search(
    repoId: string,
    query: string,
    options?: {
      type?: string
      language?: string
      limit?: number
    }
  ): Promise<SearchResult[]> {
    const params = new URLSearchParams({ query })
    if (options?.type) params.append('type', options.type)
    if (options?.language) params.append('language', options.language)
    if (options?.limit) params.append('limit', options.limit.toString())

    const response = await this.client.get(
      `/repositories/${repoId}/search?${params}`
    )
    return response.data
  }

  // Analysis endpoints
  async getCallGraph(
    repoId: string,
    entityId: string,
    depth?: number
  ): Promise<CallGraph> {
    const params = depth ? `?depth=${depth}` : ''
    const response = await this.client.get(
      `/repositories/${repoId}/entities/${entityId}/call-graph${params}`
    )
    return response.data
  }

  async analyzeImpact(repoId: string, entityId: string): Promise<ImpactAnalysis> {
    const response = await this.client.get(
      `/repositories/${repoId}/entities/${entityId}/impact`
    )
    return response.data
  }

  async getReferences(repoId: string, entityId: string): Promise<CodeEntity[]> {
    const response = await this.client.get(
      `/repositories/${repoId}/entities/${entityId}/references`
    )
    return response.data
  }
}

export const api = new ApiClient()
export default api
