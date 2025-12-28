// Repository types
export interface Repository {
  id: string
  name: string
  url: string
  defaultBranch: string
  lastIndexed: string | null
  status: RepositoryStatus
  stats: RepositoryStats
}

export type RepositoryStatus = 'pending' | 'indexing' | 'ready' | 'error'

export interface RepositoryStats {
  totalFiles: number
  totalLines: number
  totalFunctions: number
  totalClasses: number
  languages: LanguageStat[]
}

export interface LanguageStat {
  language: string
  files: number
  lines: number
  percentage: number
}

// Code entity types
export interface CodeEntity {
  id: string
  name: string
  type: EntityType
  filePath: string
  startLine: number
  endLine: number
  sourceCode: string
  docstring: string | null
  language: string
}

export type EntityType = 'function' | 'class' | 'method' | 'module' | 'variable'

export interface FileNode {
  name: string
  path: string
  type: 'file' | 'directory'
  language?: string
  children?: FileNode[]
  entityCount?: number
}

// Query types
export interface QueryRequest {
  question: string
  repoId: string
  context?: QueryContext
}

export interface QueryContext {
  filePath?: string
  startLine?: number
  endLine?: number
  selectedCode?: string
}

export interface QueryResponse {
  answer: string
  sources: CodeSource[]
  queryType: QueryType
  confidence: number
}

export type QueryType = 'explain' | 'find' | 'trace' | 'generate' | 'analyze'

export interface CodeSource {
  entityId: string
  entityName: string
  entityType: EntityType
  filePath: string
  startLine: number
  endLine: number
  relevance: number
  snippet: string
}

// Chat types
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  sources?: CodeSource[]
}

// Graph types
export interface GraphNode {
  id: string
  name: string
  type: EntityType
  filePath: string
  x?: number
  y?: number
}

export interface GraphEdge {
  source: string
  target: string
  type: 'calls' | 'uses' | 'imports' | 'inherits'
}

export interface CallGraph {
  nodes: GraphNode[]
  edges: GraphEdge[]
  rootId: string
}

// Analysis types
export interface ImpactAnalysis {
  entityId: string
  directImpact: ImpactedEntity[]
  transitiveImpact: ImpactedEntity[]
  riskLevel: 'low' | 'medium' | 'high' | 'critical'
  summary: string
}

export interface ImpactedEntity {
  id: string
  name: string
  type: EntityType
  filePath: string
  impactType: 'direct' | 'transitive'
  reason: string
}

// Search types
export interface SearchResult {
  entity: CodeEntity
  score: number
  highlights: SearchHighlight[]
}

export interface SearchHighlight {
  field: string
  snippet: string
}

// API response wrapper
export interface ApiResponse<T> {
  data: T
  status: 'success' | 'error'
  message?: string
}
