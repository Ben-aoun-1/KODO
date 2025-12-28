/**
 * Type definitions for Kodo API responses.
 */

export interface Repository {
  id: string;
  name: string;
  path: string;
  language: string;
  lastIndexed: string | null;
  fileCount: number;
  entityCount: number;
}

export interface CodeEntity {
  id: string;
  name: string;
  type: EntityType;
  filePath: string;
  startLine: number;
  endLine: number;
  docstring: string | null;
  language: string;
}

export type EntityType =
  | "function"
  | "method"
  | "class"
  | "module"
  | "variable"
  | "import";

export interface QueryRequest {
  question: string;
  repoId: string;
  filePath?: string;
  context?: string;
  maxResults?: number;
}

export interface QueryResponse {
  answer: string;
  sources: SourceReference[];
  queryType: QueryType;
  confidence: number;
  executionTime: number;
}

export type QueryType =
  | "explain"
  | "find"
  | "trace"
  | "generate"
  | "analyze"
  | "general";

export interface SourceReference {
  filePath: string;
  entityName: string;
  entityType: EntityType;
  startLine: number;
  endLine: number;
  relevance: number;
  snippet?: string;
}

export interface AskRequest {
  question: string;
  repoId: string;
  context?: CodeContext;
}

export interface CodeContext {
  filePath: string;
  startLine: number;
  endLine: number;
  code: string;
  language: string;
}

export interface AskResponse {
  answer: string;
  sources: SourceReference[];
  followUpQuestions?: string[];
}

export interface ImpactAnalysis {
  entityId: string;
  entityName: string;
  changeType: ChangeType;
  overallImpact: ImpactLevel;
  affectedEntities: AffectedEntity[];
  affectedFiles: string[];
  riskScore: number;
  recommendations: string[];
  requiresReview: boolean;
}

export type ChangeType = "add" | "modify" | "delete" | "rename" | "move";

export type ImpactLevel = "none" | "low" | "medium" | "high" | "critical";

export interface AffectedEntity {
  entityId: string;
  entityName: string;
  entityType: EntityType;
  filePath: string;
  startLine: number;
  endLine: number;
  impactLevel: ImpactLevel;
  impactReason: string;
  distance: number;
  isDirect: boolean;
}

export interface CodeMetrics {
  cyclomaticComplexity: number;
  cognitiveComplexity: number;
  linesOfCode: number;
  nestingDepth: number;
  parameterCount: number;
  maintainabilityIndex: number;
}

export interface CallGraphNode {
  id: string;
  name: string;
  type: EntityType;
  filePath: string;
}

export interface CallGraphEdge {
  source: string;
  target: string;
  callCount: number;
}

export interface CallGraph {
  nodes: CallGraphNode[];
  edges: CallGraphEdge[];
  rootNode: string;
}

export interface HealthStatus {
  status: "healthy" | "degraded" | "unhealthy";
  version: string;
  services: {
    neo4j: boolean;
    qdrant: boolean;
    llm: boolean;
  };
}

export interface IndexRequest {
  repoPath: string;
  repoName?: string;
  branch?: string;
  incremental?: boolean;
}

export interface IndexProgress {
  repoId: string;
  status: "pending" | "indexing" | "completed" | "failed";
  progress: number;
  currentFile?: string;
  totalFiles: number;
  processedFiles: number;
  errors: string[];
}

export interface HoverInfo {
  entityName: string;
  entityType: EntityType;
  signature?: string;
  docstring?: string;
  filePath: string;
  definitionLine: number;
  metrics?: CodeMetrics;
  callers?: string[];
  callees?: string[];
}

export interface ReferenceInfo {
  entityName: string;
  entityType: EntityType;
  definition: {
    filePath: string;
    line: number;
  };
  references: Reference[];
  totalCount: number;
}

export interface Reference {
  filePath: string;
  line: number;
  column: number;
  context: string;
  referenceType: "call" | "import" | "inheritance" | "assignment";
}

export interface SearchResult {
  entities: CodeEntity[];
  totalCount: number;
  query: string;
  searchType: "semantic" | "exact" | "fuzzy";
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}
