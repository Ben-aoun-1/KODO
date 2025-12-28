/**
 * Code lens provider for showing function references and metrics.
 */

import * as vscode from "vscode";
import { getApiClient } from "../api/client";
import { ReferenceInfo } from "../api/types";

interface FunctionInfo {
  name: string;
  range: vscode.Range;
  type: "function" | "method" | "class";
}

export class KodoCodeLensProvider implements vscode.CodeLensProvider {
  private repoId: string | null = null;
  private _onDidChangeCodeLenses = new vscode.EventEmitter<void>();
  public readonly onDidChangeCodeLenses = this._onDidChangeCodeLenses.event;

  // Regex patterns for detecting functions and classes
  private patterns: { [key: string]: RegExp[] } = {
    python: [
      /^\s*(?:async\s+)?def\s+(\w+)\s*\(/gm,
      /^\s*class\s+(\w+)\s*[:(]/gm,
    ],
    javascript: [
      /^\s*(?:async\s+)?function\s+(\w+)\s*\(/gm,
      /^\s*(?:export\s+)?(?:async\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>/gm,
      /^\s*(?:export\s+)?class\s+(\w+)/gm,
      /^\s*(\w+)\s*\([^)]*\)\s*\{/gm,
    ],
    typescript: [
      /^\s*(?:async\s+)?function\s+(\w+)\s*[<(]/gm,
      /^\s*(?:export\s+)?(?:async\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>/gm,
      /^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)/gm,
      /^\s*(?:public|private|protected)?\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*[:{]/gm,
    ],
    go: [
      /^\s*func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(/gm,
      /^\s*type\s+(\w+)\s+struct\s*\{/gm,
    ],
    rust: [
      /^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)/gm,
      /^\s*(?:pub\s+)?struct\s+(\w+)/gm,
      /^\s*(?:pub\s+)?impl(?:<[^>]+>)?\s+(\w+)/gm,
    ],
  };

  setRepoId(repoId: string | null): void {
    this.repoId = repoId;
    this._onDidChangeCodeLenses.fire();
  }

  async provideCodeLenses(
    document: vscode.TextDocument,
    _token: vscode.CancellationToken
  ): Promise<vscode.CodeLens[]> {
    // Check if code lens is enabled
    const config = vscode.workspace.getConfiguration("kodo");
    if (!config.get<boolean>("enableCodeLens", true)) {
      return [];
    }

    const codeLenses: vscode.CodeLens[] = [];
    const functions = this.findFunctions(document);

    for (const func of functions) {
      // Reference count lens
      codeLenses.push(
        new vscode.CodeLens(func.range, {
          title: "$(references) loading...",
          command: "",
        })
      );

      // Analyze lens
      codeLenses.push(
        new vscode.CodeLens(func.range, {
          title: "$(graph) Analyze",
          command: "kodo.analyzeFunction",
          arguments: [document.uri, func.name, func.range.start.line],
        })
      );
    }

    return codeLenses;
  }

  async resolveCodeLens(
    codeLens: vscode.CodeLens,
    _token: vscode.CancellationToken
  ): Promise<vscode.CodeLens> {
    // Skip already resolved lenses
    if (codeLens.command?.command) {
      return codeLens;
    }

    if (!this.repoId) {
      codeLens.command = {
        title: "$(references) -",
        command: "",
      };
      return codeLens;
    }

    // Extract function name from the lens title or arguments
    const title = codeLens.command?.title || "";
    if (!title.includes("loading")) {
      return codeLens;
    }

    try {
      const client = getApiClient();
      const editor = vscode.window.activeTextEditor;

      if (!editor) {
        return codeLens;
      }

      // Get the function name from the line
      const line = editor.document.lineAt(codeLens.range.start.line);
      const funcName = this.extractFunctionName(line.text, editor.document.languageId);

      if (!funcName) {
        codeLens.command = {
          title: "$(references) -",
          command: "",
        };
        return codeLens;
      }

      const references = await client.getReferences(this.repoId, funcName);
      const count = references.totalCount;

      codeLens.command = {
        title: `$(references) ${count} reference${count !== 1 ? "s" : ""}`,
        command: "kodo.showReferences",
        arguments: [funcName, references],
      };
    } catch (error) {
      console.error("Kodo code lens error:", error);
      codeLens.command = {
        title: "$(references) -",
        command: "",
      };
    }

    return codeLens;
  }

  private findFunctions(document: vscode.TextDocument): FunctionInfo[] {
    const functions: FunctionInfo[] = [];
    const text = document.getText();
    const languageId = document.languageId;

    // Get patterns for this language
    let patterns = this.patterns[languageId];
    if (!patterns) {
      // Fallback for similar languages
      if (languageId === "javascriptreact" || languageId === "vue") {
        patterns = this.patterns.javascript;
      } else if (languageId === "typescriptreact") {
        patterns = this.patterns.typescript;
      } else {
        return functions;
      }
    }

    for (const pattern of patterns) {
      // Reset regex state
      pattern.lastIndex = 0;

      let match;
      while ((match = pattern.exec(text)) !== null) {
        const name = match[1];
        if (!name) continue;

        // Skip common false positives
        if (["if", "while", "for", "switch", "catch", "constructor"].includes(name)) {
          continue;
        }

        const startPos = document.positionAt(match.index);
        const line = document.lineAt(startPos.line);

        // Determine type
        let type: "function" | "method" | "class" = "function";
        if (match[0].includes("class")) {
          type = "class";
        } else if (startPos.character > 0) {
          type = "method";
        }

        functions.push({
          name,
          range: line.range,
          type,
        });
      }
    }

    return functions;
  }

  private extractFunctionName(lineText: string, languageId: string): string | null {
    const patterns = this.patterns[languageId] || this.patterns.javascript;

    for (const pattern of patterns) {
      pattern.lastIndex = 0;
      const match = pattern.exec(lineText);
      if (match && match[1]) {
        return match[1];
      }
    }

    return null;
  }

  refresh(): void {
    this._onDidChangeCodeLenses.fire();
  }
}

export class ReferenceTreeDataProvider
  implements vscode.TreeDataProvider<ReferenceTreeItem>
{
  private _onDidChangeTreeData = new vscode.EventEmitter<
    ReferenceTreeItem | undefined
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private currentReferences: ReferenceInfo | null = null;

  setReferences(references: ReferenceInfo): void {
    this.currentReferences = references;
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(element: ReferenceTreeItem): vscode.TreeItem {
    return element;
  }

  getChildren(element?: ReferenceTreeItem): ReferenceTreeItem[] {
    if (!this.currentReferences) {
      return [];
    }

    if (!element) {
      // Root level - show definition and references count
      const items: ReferenceTreeItem[] = [];

      // Definition
      items.push(
        new ReferenceTreeItem(
          `Definition: ${this.currentReferences.definition.filePath}:${this.currentReferences.definition.line}`,
          vscode.TreeItemCollapsibleState.None,
          {
            command: "vscode.open",
            title: "Go to Definition",
            arguments: [
              vscode.Uri.file(this.currentReferences.definition.filePath),
              {
                selection: new vscode.Range(
                  this.currentReferences.definition.line - 1,
                  0,
                  this.currentReferences.definition.line - 1,
                  0
                ),
              },
            ],
          }
        )
      );

      // Group references by file
      const byFile = new Map<string, typeof this.currentReferences.references>();
      for (const ref of this.currentReferences.references) {
        const existing = byFile.get(ref.filePath) || [];
        existing.push(ref);
        byFile.set(ref.filePath, existing);
      }

      for (const [filePath, refs] of byFile) {
        items.push(
          new ReferenceTreeItem(
            `${filePath} (${refs.length})`,
            vscode.TreeItemCollapsibleState.Collapsed,
            undefined,
            refs
          )
        );
      }

      return items;
    }

    // Child level - show individual references
    if (element.references) {
      return element.references.map(
        (ref) =>
          new ReferenceTreeItem(
            `Line ${ref.line}: ${ref.context.trim()}`,
            vscode.TreeItemCollapsibleState.None,
            {
              command: "vscode.open",
              title: "Go to Reference",
              arguments: [
                vscode.Uri.file(ref.filePath),
                {
                  selection: new vscode.Range(
                    ref.line - 1,
                    ref.column - 1,
                    ref.line - 1,
                    ref.column - 1
                  ),
                },
              ],
            }
          )
      );
    }

    return [];
  }
}

class ReferenceTreeItem extends vscode.TreeItem {
  constructor(
    label: string,
    collapsibleState: vscode.TreeItemCollapsibleState,
    command?: vscode.Command,
    public readonly references?: ReferenceInfo["references"]
  ) {
    super(label, collapsibleState);
    this.command = command;
  }
}
