/**
 * API client for communicating with the Kodo backend.
 */

import axios, { AxiosInstance, AxiosError } from "axios";
import * as vscode from "vscode";
import {
  Repository,
  CodeEntity,
  QueryRequest,
  QueryResponse,
  AskRequest,
  AskResponse,
  ImpactAnalysis,
  CallGraph,
  HealthStatus,
  IndexRequest,
  IndexProgress,
  HoverInfo,
  ReferenceInfo,
  SearchResult,
  ApiError,
  ChangeType,
} from "./types";

export class KodoApiClient {
  private client: AxiosInstance;
  private baseUrl: string;
  private apiKey: string | undefined;

  constructor() {
    const config = vscode.workspace.getConfiguration("kodo");
    this.baseUrl = config.get<string>("serverUrl") || "http://localhost:8000";
    this.apiKey = config.get<string>("apiKey");

    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
        ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
      },
    });

    // Listen for configuration changes
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("kodo")) {
        this.updateConfiguration();
      }
    });
  }

  private updateConfiguration(): void {
    const config = vscode.workspace.getConfiguration("kodo");
    this.baseUrl = config.get<string>("serverUrl") || "http://localhost:8000";
    this.apiKey = config.get<string>("apiKey");

    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
        ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
      },
    });
  }

  private handleError(error: AxiosError): ApiError {
    if (error.response) {
      const data = error.response.data as Record<string, unknown>;
      return {
        code: String(error.response.status),
        message: (data.detail as string) || error.message,
        details: data,
      };
    } else if (error.request) {
      return {
        code: "NETWORK_ERROR",
        message: "Unable to connect to Kodo server",
      };
    } else {
      return {
        code: "REQUEST_ERROR",
        message: error.message,
      };
    }
  }

  // Health check
  async checkHealth(): Promise<HealthStatus> {
    try {
      const response = await this.client.get<HealthStatus>("/health");
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  // Repository operations
  async listRepositories(): Promise<Repository[]> {
    try {
      const response = await this.client.get<Repository[]>("/api/v1/repos");
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  async getRepository(repoId: string): Promise<Repository> {
    try {
      const response = await this.client.get<Repository>(
        `/api/v1/repos/${repoId}`
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  async indexRepository(request: IndexRequest): Promise<IndexProgress> {
    try {
      const response = await this.client.post<IndexProgress>(
        "/api/v1/repos/index",
        request
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  async getIndexProgress(repoId: string): Promise<IndexProgress> {
    try {
      const response = await this.client.get<IndexProgress>(
        `/api/v1/repos/${repoId}/progress`
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  // Query operations
  async query(request: QueryRequest): Promise<QueryResponse> {
    try {
      const response = await this.client.post<QueryResponse>(
        "/api/v1/query",
        request
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  async ask(request: AskRequest): Promise<AskResponse> {
    try {
      const response = await this.client.post<AskResponse>(
        "/api/v1/ask",
        request
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  // Entity operations
  async getEntity(repoId: string, entityId: string): Promise<CodeEntity> {
    try {
      const response = await this.client.get<CodeEntity>(
        `/api/v1/repos/${repoId}/entities/${entityId}`
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  async searchEntities(
    repoId: string,
    query: string,
    limit: number = 20
  ): Promise<SearchResult> {
    try {
      const response = await this.client.get<SearchResult>(
        `/api/v1/repos/${repoId}/search`,
        {
          params: { query, limit },
        }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  async getHoverInfo(
    repoId: string,
    filePath: string,
    line: number,
    column: number
  ): Promise<HoverInfo | null> {
    try {
      const response = await this.client.get<HoverInfo>(
        `/api/v1/repos/${repoId}/hover`,
        {
          params: { file_path: filePath, line, column },
        }
      );
      return response.data;
    } catch (error) {
      const apiError = this.handleError(error as AxiosError);
      if (apiError.code === "404") {
        return null;
      }
      throw apiError;
    }
  }

  async getReferences(
    repoId: string,
    entityName: string,
    entityType?: string
  ): Promise<ReferenceInfo> {
    try {
      const response = await this.client.get<ReferenceInfo>(
        `/api/v1/repos/${repoId}/references`,
        {
          params: { entity_name: entityName, entity_type: entityType },
        }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  // Analysis operations
  async analyzeImpact(
    repoId: string,
    entityId: string,
    changeType: ChangeType
  ): Promise<ImpactAnalysis> {
    try {
      const response = await this.client.post<ImpactAnalysis>(
        `/api/v1/repos/${repoId}/analyze/impact`,
        {
          entity_id: entityId,
          change_type: changeType,
        }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  async getCallGraph(
    repoId: string,
    entityId: string,
    depth: number = 3
  ): Promise<CallGraph> {
    try {
      const response = await this.client.get<CallGraph>(
        `/api/v1/repos/${repoId}/graph/calls`,
        {
          params: { entity_id: entityId, depth },
        }
      );
      return response.data;
    } catch (error) {
      throw this.handleError(error as AxiosError);
    }
  }

  // Utility methods
  async findRepositoryByPath(workspacePath: string): Promise<Repository | null> {
    try {
      const repos = await this.listRepositories();
      return repos.find((repo) => repo.path === workspacePath) || null;
    } catch {
      return null;
    }
  }
}

// Singleton instance
let clientInstance: KodoApiClient | null = null;

export function getApiClient(): KodoApiClient {
  if (!clientInstance) {
    clientInstance = new KodoApiClient();
  }
  return clientInstance;
}

export function resetApiClient(): void {
  clientInstance = null;
}
