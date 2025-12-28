/**
 * Repository status tree view provider.
 */

import * as vscode from "vscode";
import { getApiClient } from "../api/client";
import { Repository } from "../api/types";

export class RepositoryStatusProvider
  implements vscode.TreeDataProvider<StatusItem>
{
  private _onDidChangeTreeData = new vscode.EventEmitter<
    StatusItem | undefined
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private repository: Repository | null = null;
  private serverStatus: "connected" | "disconnected" | "checking" = "checking";

  constructor() {
    this.checkServerStatus();
  }

  setRepository(repo: Repository | null): void {
    this.repository = repo;
    this._onDidChangeTreeData.fire(undefined);
  }

  refresh(): void {
    this.checkServerStatus();
    this._onDidChangeTreeData.fire(undefined);
  }

  private async checkServerStatus(): Promise<void> {
    try {
      const client = getApiClient();
      await client.checkHealth();
      this.serverStatus = "connected";
    } catch {
      this.serverStatus = "disconnected";
    }
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(element: StatusItem): vscode.TreeItem {
    return element;
  }

  getChildren(element?: StatusItem): StatusItem[] {
    if (element) {
      return [];
    }

    const items: StatusItem[] = [];

    // Server status
    items.push(
      new StatusItem(
        this.serverStatus === "connected"
          ? "$(check) Server Connected"
          : this.serverStatus === "checking"
            ? "$(sync~spin) Checking..."
            : "$(error) Server Disconnected",
        this.serverStatus === "connected"
          ? "Connected to Kodo server"
          : "Unable to connect to Kodo server",
        vscode.TreeItemCollapsibleState.None
      )
    );

    // Repository status
    if (this.repository) {
      items.push(
        new StatusItem(
          `$(repo) ${this.repository.name}`,
          `Repository ID: ${this.repository.id}`,
          vscode.TreeItemCollapsibleState.None
        )
      );

      items.push(
        new StatusItem(
          `$(file-code) ${this.repository.fileCount} files`,
          "Total files indexed",
          vscode.TreeItemCollapsibleState.None
        )
      );

      items.push(
        new StatusItem(
          `$(symbol-class) ${this.repository.entityCount} entities`,
          "Functions, classes, and methods",
          vscode.TreeItemCollapsibleState.None
        )
      );

      if (this.repository.lastIndexed) {
        const lastIndexed = new Date(this.repository.lastIndexed);
        items.push(
          new StatusItem(
            `$(history) ${this.formatTimeAgo(lastIndexed)}`,
            `Last indexed: ${lastIndexed.toLocaleString()}`,
            vscode.TreeItemCollapsibleState.None
          )
        );
      }
    } else {
      items.push(
        new StatusItem(
          "$(warning) No repository indexed",
          "Run 'Kodo: Index Current Repository' to get started",
          vscode.TreeItemCollapsibleState.None,
          {
            command: "kodo.indexRepository",
            title: "Index Repository",
          }
        )
      );
    }

    return items;
  }

  private formatTimeAgo(date: Date): string {
    const seconds = Math.floor((Date.now() - date.getTime()) / 1000);

    if (seconds < 60) {
      return "just now";
    }

    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) {
      return `${minutes}m ago`;
    }

    const hours = Math.floor(minutes / 60);
    if (hours < 24) {
      return `${hours}h ago`;
    }

    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  }
}

class StatusItem extends vscode.TreeItem {
  constructor(
    label: string,
    tooltip: string,
    collapsibleState: vscode.TreeItemCollapsibleState,
    command?: vscode.Command
  ) {
    super(label, collapsibleState);
    this.tooltip = tooltip;
    this.command = command;
  }
}

export class RecentQueriesProvider
  implements vscode.TreeDataProvider<QueryItem>
{
  private _onDidChangeTreeData = new vscode.EventEmitter<QueryItem | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private queries: { question: string; timestamp: Date }[] = [];

  addQuery(question: string): void {
    this.queries.unshift({ question, timestamp: new Date() });
    // Keep only last 10
    this.queries = this.queries.slice(0, 10);
    this._onDidChangeTreeData.fire(undefined);
  }

  clear(): void {
    this.queries = [];
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(element: QueryItem): vscode.TreeItem {
    return element;
  }

  getChildren(): QueryItem[] {
    if (this.queries.length === 0) {
      return [
        new QueryItem(
          "No recent queries",
          "",
          vscode.TreeItemCollapsibleState.None
        ),
      ];
    }

    return this.queries.map(
      (q) =>
        new QueryItem(
          q.question.length > 50
            ? q.question.substring(0, 47) + "..."
            : q.question,
          q.question,
          vscode.TreeItemCollapsibleState.None,
          {
            command: "kodo.askQuestion",
            title: "Ask Question",
            arguments: [q.question],
          }
        )
    );
  }
}

class QueryItem extends vscode.TreeItem {
  constructor(
    label: string,
    tooltip: string,
    collapsibleState: vscode.TreeItemCollapsibleState,
    command?: vscode.Command
  ) {
    super(label, collapsibleState);
    this.tooltip = tooltip;
    this.command = command;
    this.iconPath = new vscode.ThemeIcon("comment-discussion");
  }
}
