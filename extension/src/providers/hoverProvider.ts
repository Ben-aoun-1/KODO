/**
 * Hover provider for displaying code insights on hover.
 */

import * as vscode from "vscode";
import { getApiClient } from "../api/client";
import { HoverInfo, CodeMetrics } from "../api/types";

export class KodoHoverProvider implements vscode.HoverProvider {
  private repoId: string | null = null;
  private cache: Map<string, { info: HoverInfo; timestamp: number }> =
    new Map();
  private cacheTtl = 60000; // 1 minute cache

  setRepoId(repoId: string | null): void {
    this.repoId = repoId;
    this.cache.clear();
  }

  async provideHover(
    document: vscode.TextDocument,
    position: vscode.Position,
    _token: vscode.CancellationToken
  ): Promise<vscode.Hover | null> {
    // Check if hover is enabled
    const config = vscode.workspace.getConfiguration("kodo");
    if (!config.get<boolean>("enableHover", true)) {
      return null;
    }

    if (!this.repoId) {
      return null;
    }

    // Get the word at position
    const wordRange = document.getWordRangeAtPosition(position);
    if (!wordRange) {
      return null;
    }

    const word = document.getText(wordRange);
    if (!word || word.length < 2) {
      return null;
    }

    // Check cache
    const cacheKey = `${document.fileName}:${position.line}:${word}`;
    const cached = this.cache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < this.cacheTtl) {
      return this.createHover(cached.info);
    }

    try {
      const client = getApiClient();
      const hoverInfo = await client.getHoverInfo(
        this.repoId,
        this.getRelativePath(document.fileName),
        position.line + 1,
        position.character + 1
      );

      if (!hoverInfo) {
        return null;
      }

      // Cache the result
      this.cache.set(cacheKey, { info: hoverInfo, timestamp: Date.now() });

      return this.createHover(hoverInfo);
    } catch (error) {
      console.error("Kodo hover error:", error);
      return null;
    }
  }

  private createHover(info: HoverInfo): vscode.Hover {
    const contents: vscode.MarkdownString[] = [];

    // Entity header
    const header = new vscode.MarkdownString();
    header.appendMarkdown(`### ${this.getEntityIcon(info.entityType)} ${info.entityName}\n\n`);

    if (info.signature) {
      header.appendCodeblock(info.signature, this.getLanguageId(info.filePath));
    }
    contents.push(header);

    // Docstring
    if (info.docstring) {
      const docstring = new vscode.MarkdownString();
      docstring.appendMarkdown(info.docstring);
      contents.push(docstring);
    }

    // Metrics
    if (info.metrics) {
      const metrics = new vscode.MarkdownString();
      metrics.appendMarkdown(this.formatMetrics(info.metrics));
      contents.push(metrics);
    }

    // Callers and callees
    if (info.callers && info.callers.length > 0) {
      const callers = new vscode.MarkdownString();
      callers.appendMarkdown(`**Called by:** ${info.callers.slice(0, 5).join(", ")}`);
      if (info.callers.length > 5) {
        callers.appendMarkdown(` _and ${info.callers.length - 5} more..._`);
      }
      contents.push(callers);
    }

    if (info.callees && info.callees.length > 0) {
      const callees = new vscode.MarkdownString();
      callees.appendMarkdown(`**Calls:** ${info.callees.slice(0, 5).join(", ")}`);
      if (info.callees.length > 5) {
        callees.appendMarkdown(` _and ${info.callees.length - 5} more..._`);
      }
      contents.push(callees);
    }

    // Location
    const location = new vscode.MarkdownString();
    location.appendMarkdown(
      `\n---\n_Defined in [${info.filePath}:${info.definitionLine}](${info.filePath}#L${info.definitionLine})_`
    );
    contents.push(location);

    return new vscode.Hover(contents);
  }

  private formatMetrics(metrics: CodeMetrics): string {
    const parts: string[] = [];

    // Complexity badge
    const complexityColor = this.getComplexityColor(metrics.cyclomaticComplexity);
    parts.push(`**Complexity:** ${complexityColor} ${metrics.cyclomaticComplexity}`);

    // Other metrics
    parts.push(`**Lines:** ${metrics.linesOfCode}`);

    if (metrics.parameterCount > 0) {
      parts.push(`**Params:** ${metrics.parameterCount}`);
    }

    if (metrics.nestingDepth > 2) {
      parts.push(`**Nesting:** ${metrics.nestingDepth}`);
    }

    // Maintainability index
    const maintainability = Math.round(metrics.maintainabilityIndex);
    const maintColor = this.getMaintainabilityColor(maintainability);
    parts.push(`**Maintainability:** ${maintColor} ${maintainability}`);

    return `\n---\n${parts.join(" | ")}`;
  }

  private getComplexityColor(complexity: number): string {
    if (complexity <= 5) return "🟢";
    if (complexity <= 10) return "🟡";
    if (complexity <= 20) return "🟠";
    return "🔴";
  }

  private getMaintainabilityColor(index: number): string {
    if (index >= 80) return "🟢";
    if (index >= 60) return "🟡";
    if (index >= 40) return "🟠";
    return "🔴";
  }

  private getEntityIcon(entityType: string): string {
    switch (entityType) {
      case "function":
        return "$(symbol-function)";
      case "method":
        return "$(symbol-method)";
      case "class":
        return "$(symbol-class)";
      case "module":
        return "$(symbol-namespace)";
      case "variable":
        return "$(symbol-variable)";
      default:
        return "$(symbol-misc)";
    }
  }

  private getLanguageId(filePath: string): string {
    const ext = filePath.split(".").pop()?.toLowerCase();
    switch (ext) {
      case "py":
        return "python";
      case "js":
        return "javascript";
      case "ts":
        return "typescript";
      case "go":
        return "go";
      case "rs":
        return "rust";
      default:
        return "plaintext";
    }
  }

  private getRelativePath(absolutePath: string): string {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
      return absolutePath;
    }

    for (const folder of workspaceFolders) {
      if (absolutePath.startsWith(folder.uri.fsPath)) {
        return absolutePath.substring(folder.uri.fsPath.length + 1);
      }
    }

    return absolutePath;
  }

  clearCache(): void {
    this.cache.clear();
  }
}
