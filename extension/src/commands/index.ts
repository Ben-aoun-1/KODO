/**
 * Command implementations for Kodo extension.
 */

import * as vscode from "vscode";
import { getApiClient } from "../api/client";
import { ReferenceInfo, ImpactAnalysis, ChangeType } from "../api/types";

export interface CommandContext {
  repoId: string | null;
  setRepoId: (id: string | null) => void;
}

export function registerCommands(
  context: vscode.ExtensionContext,
  cmdContext: CommandContext
): void {
  // Ask a question about code
  context.subscriptions.push(
    vscode.commands.registerCommand("kodo.askQuestion", async () => {
      await askQuestionCommand(cmdContext);
    })
  );

  // Explain selected code
  context.subscriptions.push(
    vscode.commands.registerCommand("kodo.explainSelection", async () => {
      await explainSelectionCommand(cmdContext);
    })
  );

  // Find all references
  context.subscriptions.push(
    vscode.commands.registerCommand("kodo.findReferences", async () => {
      await findReferencesCommand(cmdContext);
    })
  );

  // Analyze impact
  context.subscriptions.push(
    vscode.commands.registerCommand("kodo.analyzeImpact", async () => {
      await analyzeImpactCommand(cmdContext);
    })
  );

  // Show call graph
  context.subscriptions.push(
    vscode.commands.registerCommand("kodo.showCallGraph", async () => {
      await showCallGraphCommand(cmdContext);
    })
  );

  // Index repository
  context.subscriptions.push(
    vscode.commands.registerCommand("kodo.indexRepository", async () => {
      await indexRepositoryCommand(cmdContext);
    })
  );

  // Refresh index
  context.subscriptions.push(
    vscode.commands.registerCommand("kodo.refreshIndex", async () => {
      await refreshIndexCommand(cmdContext);
    })
  );

  // Show references (called from code lens)
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "kodo.showReferences",
      async (entityName: string, references: ReferenceInfo) => {
        await showReferencesCommand(entityName, references);
      }
    )
  );

  // Analyze function (called from code lens)
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "kodo.analyzeFunction",
      async (uri: vscode.Uri, functionName: string, line: number) => {
        await analyzeFunctionCommand(cmdContext, uri, functionName, line);
      }
    )
  );
}

async function askQuestionCommand(context: CommandContext): Promise<void> {
  if (!context.repoId) {
    vscode.window.showWarningMessage(
      "No repository indexed. Run 'Kodo: Index Current Repository' first."
    );
    return;
  }

  const question = await vscode.window.showInputBox({
    prompt: "Ask a question about your code",
    placeHolder: "e.g., How does the authentication flow work?",
  });

  if (!question) {
    return;
  }

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Kodo",
      cancellable: false,
    },
    async (progress) => {
      progress.report({ message: "Thinking..." });

      try {
        const client = getApiClient();
        const editor = vscode.window.activeTextEditor;

        // Build context from current file
        let fileContext = undefined;
        if (editor) {
          const document = editor.document;
          fileContext = {
            filePath: getRelativePath(document.fileName),
            startLine: 1,
            endLine: document.lineCount,
            code: document.getText(),
            language: document.languageId,
          };
        }

        const response = await client.ask({
          question,
          repoId: context.repoId!,
          context: fileContext,
        });

        // Show response in new document
        const doc = await vscode.workspace.openTextDocument({
          content: formatAskResponse(question, response),
          language: "markdown",
        });
        await vscode.window.showTextDocument(doc, { preview: true });
      } catch (error) {
        vscode.window.showErrorMessage(`Kodo: ${(error as Error).message}`);
      }
    }
  );
}

async function explainSelectionCommand(context: CommandContext): Promise<void> {
  if (!context.repoId) {
    vscode.window.showWarningMessage("No repository indexed.");
    return;
  }

  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) {
    vscode.window.showWarningMessage("Please select some code to explain.");
    return;
  }

  const selectedText = editor.document.getText(editor.selection);

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Kodo",
      cancellable: false,
    },
    async (progress) => {
      progress.report({ message: "Analyzing code..." });

      try {
        const client = getApiClient();
        const response = await client.ask({
          question: `Explain this code:\n\`\`\`${editor.document.languageId}\n${selectedText}\n\`\`\``,
          repoId: context.repoId!,
          context: {
            filePath: getRelativePath(editor.document.fileName),
            startLine: editor.selection.start.line + 1,
            endLine: editor.selection.end.line + 1,
            code: selectedText,
            language: editor.document.languageId,
          },
        });

        // Show response in new document
        const doc = await vscode.workspace.openTextDocument({
          content: `# Code Explanation\n\n${response.answer}`,
          language: "markdown",
        });
        await vscode.window.showTextDocument(doc, {
          preview: true,
          viewColumn: vscode.ViewColumn.Beside,
        });
      } catch (error) {
        vscode.window.showErrorMessage(`Kodo: ${(error as Error).message}`);
      }
    }
  );
}

async function findReferencesCommand(context: CommandContext): Promise<void> {
  if (!context.repoId) {
    vscode.window.showWarningMessage("No repository indexed.");
    return;
  }

  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }

  const wordRange = editor.document.getWordRangeAtPosition(editor.selection.active);
  if (!wordRange) {
    vscode.window.showWarningMessage("No symbol found at cursor position.");
    return;
  }

  const word = editor.document.getText(wordRange);

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Kodo",
      cancellable: false,
    },
    async (progress) => {
      progress.report({ message: "Finding references..." });

      try {
        const client = getApiClient();
        const references = await client.getReferences(context.repoId!, word);

        if (references.totalCount === 0) {
          vscode.window.showInformationMessage(`No references found for "${word}".`);
          return;
        }

        // Convert to VS Code locations and show in peek view
        const locations: vscode.Location[] = references.references.map(
          (ref) =>
            new vscode.Location(
              vscode.Uri.file(resolveAbsolutePath(ref.filePath)),
              new vscode.Position(ref.line - 1, ref.column - 1)
            )
        );

        await vscode.commands.executeCommand(
          "editor.action.showReferences",
          editor.document.uri,
          editor.selection.active,
          locations
        );
      } catch (error) {
        vscode.window.showErrorMessage(`Kodo: ${(error as Error).message}`);
      }
    }
  );
}

async function analyzeImpactCommand(context: CommandContext): Promise<void> {
  if (!context.repoId) {
    vscode.window.showWarningMessage("No repository indexed.");
    return;
  }

  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }

  const wordRange = editor.document.getWordRangeAtPosition(editor.selection.active);
  if (!wordRange) {
    vscode.window.showWarningMessage("No symbol found at cursor position.");
    return;
  }

  const entityName = editor.document.getText(wordRange);

  // Ask for change type
  const changeType = await vscode.window.showQuickPick(
    [
      { label: "Modify", value: "modify" as ChangeType },
      { label: "Delete", value: "delete" as ChangeType },
      { label: "Rename", value: "rename" as ChangeType },
    ],
    { placeHolder: "What type of change are you planning?" }
  );

  if (!changeType) {
    return;
  }

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Kodo",
      cancellable: false,
    },
    async (progress) => {
      progress.report({ message: "Analyzing impact..." });

      try {
        const client = getApiClient();
        const analysis = await client.analyzeImpact(
          context.repoId!,
          entityName,
          changeType.value
        );

        // Show analysis in new document
        const doc = await vscode.workspace.openTextDocument({
          content: formatImpactAnalysis(analysis),
          language: "markdown",
        });
        await vscode.window.showTextDocument(doc, { preview: true });
      } catch (error) {
        vscode.window.showErrorMessage(`Kodo: ${(error as Error).message}`);
      }
    }
  );
}

async function showCallGraphCommand(context: CommandContext): Promise<void> {
  if (!context.repoId) {
    vscode.window.showWarningMessage("No repository indexed.");
    return;
  }

  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }

  const wordRange = editor.document.getWordRangeAtPosition(editor.selection.active);
  if (!wordRange) {
    vscode.window.showWarningMessage("No symbol found at cursor position.");
    return;
  }

  const entityName = editor.document.getText(wordRange);

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Kodo",
      cancellable: false,
    },
    async (progress) => {
      progress.report({ message: "Building call graph..." });

      try {
        const client = getApiClient();
        const callGraph = await client.getCallGraph(context.repoId!, entityName);

        // Format as text for now (could be visualized in webview)
        const doc = await vscode.workspace.openTextDocument({
          content: formatCallGraph(entityName, callGraph),
          language: "markdown",
        });
        await vscode.window.showTextDocument(doc, { preview: true });
      } catch (error) {
        vscode.window.showErrorMessage(`Kodo: ${(error as Error).message}`);
      }
    }
  );
}

async function indexRepositoryCommand(context: CommandContext): Promise<void> {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders || workspaceFolders.length === 0) {
    vscode.window.showWarningMessage("No workspace folder open.");
    return;
  }

  const folder =
    workspaceFolders.length === 1
      ? workspaceFolders[0]
      : await vscode.window.showWorkspaceFolderPick({
          placeHolder: "Select workspace to index",
        });

  if (!folder) {
    return;
  }

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Kodo",
      cancellable: false,
    },
    async (progress) => {
      progress.report({ message: "Starting indexing..." });

      try {
        const client = getApiClient();
        const result = await client.indexRepository({
          repoPath: folder.uri.fsPath,
          repoName: folder.name,
        });

        context.setRepoId(result.repoId);

        // Poll for progress
        let lastProgress = 0;
        while (result.status === "indexing" || result.status === "pending") {
          await new Promise((resolve) => setTimeout(resolve, 1000));

          const currentProgress = await client.getIndexProgress(result.repoId);
          if (currentProgress.progress !== lastProgress) {
            progress.report({
              message: `Indexing... ${currentProgress.progress}% (${currentProgress.processedFiles}/${currentProgress.totalFiles} files)`,
              increment: currentProgress.progress - lastProgress,
            });
            lastProgress = currentProgress.progress;
          }

          if (
            currentProgress.status === "completed" ||
            currentProgress.status === "failed"
          ) {
            if (currentProgress.status === "completed") {
              vscode.window.showInformationMessage(
                `Kodo: Repository indexed successfully! (${currentProgress.totalFiles} files)`
              );
            } else {
              vscode.window.showErrorMessage(
                `Kodo: Indexing failed. ${currentProgress.errors.join(", ")}`
              );
            }
            break;
          }
        }
      } catch (error) {
        vscode.window.showErrorMessage(`Kodo: ${(error as Error).message}`);
      }
    }
  );
}

async function refreshIndexCommand(context: CommandContext): Promise<void> {
  if (!context.repoId) {
    vscode.window.showWarningMessage("No repository indexed.");
    return;
  }

  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders || workspaceFolders.length === 0) {
    return;
  }

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Kodo",
      cancellable: false,
    },
    async (progress) => {
      progress.report({ message: "Refreshing index..." });

      try {
        const client = getApiClient();
        await client.indexRepository({
          repoPath: workspaceFolders[0].uri.fsPath,
          incremental: true,
        });

        vscode.window.showInformationMessage("Kodo: Index refreshed.");
      } catch (error) {
        vscode.window.showErrorMessage(`Kodo: ${(error as Error).message}`);
      }
    }
  );
}

async function showReferencesCommand(
  entityName: string,
  references: ReferenceInfo
): Promise<void> {
  const locations: vscode.Location[] = references.references.map(
    (ref) =>
      new vscode.Location(
        vscode.Uri.file(resolveAbsolutePath(ref.filePath)),
        new vscode.Position(ref.line - 1, ref.column - 1)
      )
  );

  const editor = vscode.window.activeTextEditor;
  if (editor) {
    await vscode.commands.executeCommand(
      "editor.action.showReferences",
      editor.document.uri,
      editor.selection.active,
      locations
    );
  }
}

async function analyzeFunctionCommand(
  context: CommandContext,
  _uri: vscode.Uri,
  functionName: string,
  _line: number
): Promise<void> {
  if (!context.repoId) {
    return;
  }

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Kodo",
      cancellable: false,
    },
    async (progress) => {
      progress.report({ message: `Analyzing ${functionName}...` });

      try {
        const client = getApiClient();
        const callGraph = await client.getCallGraph(context.repoId!, functionName);

        const doc = await vscode.workspace.openTextDocument({
          content: formatCallGraph(functionName, callGraph),
          language: "markdown",
        });
        await vscode.window.showTextDocument(doc, {
          preview: true,
          viewColumn: vscode.ViewColumn.Beside,
        });
      } catch (error) {
        vscode.window.showErrorMessage(`Kodo: ${(error as Error).message}`);
      }
    }
  );
}

// Helper functions

function getRelativePath(absolutePath: string): string {
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

function resolveAbsolutePath(relativePath: string): string {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders) {
    return relativePath;
  }

  return vscode.Uri.joinPath(workspaceFolders[0].uri, relativePath).fsPath;
}

function formatAskResponse(
  question: string,
  response: { answer: string; sources?: { filePath: string; entityName: string; startLine: number }[] }
): string {
  let content = `# Question\n\n${question}\n\n# Answer\n\n${response.answer}`;

  if (response.sources && response.sources.length > 0) {
    content += "\n\n## Sources\n\n";
    for (const source of response.sources) {
      content += `- [${source.entityName}](${source.filePath}#L${source.startLine})\n`;
    }
  }

  return content;
}

function formatImpactAnalysis(analysis: ImpactAnalysis): string {
  let content = `# Impact Analysis: ${analysis.entityName}\n\n`;
  content += `**Change Type:** ${analysis.changeType}\n`;
  content += `**Overall Impact:** ${getImpactBadge(analysis.overallImpact)}\n`;
  content += `**Risk Score:** ${analysis.riskScore}/100\n`;
  content += `**Requires Review:** ${analysis.requiresReview ? "Yes" : "No"}\n\n`;

  if (analysis.affectedFiles.length > 0) {
    content += `## Affected Files (${analysis.affectedFiles.length})\n\n`;
    for (const file of analysis.affectedFiles) {
      content += `- ${file}\n`;
    }
    content += "\n";
  }

  if (analysis.affectedEntities.length > 0) {
    content += `## Affected Entities (${analysis.affectedEntities.length})\n\n`;
    content += "| Entity | Type | File | Impact | Reason |\n";
    content += "|--------|------|------|--------|--------|\n";
    for (const entity of analysis.affectedEntities.slice(0, 20)) {
      content += `| ${entity.entityName} | ${entity.entityType} | ${entity.filePath} | ${getImpactBadge(entity.impactLevel)} | ${entity.impactReason} |\n`;
    }
    if (analysis.affectedEntities.length > 20) {
      content += `\n_...and ${analysis.affectedEntities.length - 20} more_\n`;
    }
    content += "\n";
  }

  if (analysis.recommendations.length > 0) {
    content += "## Recommendations\n\n";
    for (const rec of analysis.recommendations) {
      content += `- ${rec}\n`;
    }
  }

  return content;
}

function formatCallGraph(
  entityName: string,
  callGraph: { nodes: { id: string; name: string; type: string }[]; edges: { source: string; target: string }[] }
): string {
  let content = `# Call Graph: ${entityName}\n\n`;

  // Find callers (edges pointing to this entity)
  const callers = callGraph.edges
    .filter((e) => e.target === entityName)
    .map((e) => e.source);

  // Find callees (edges from this entity)
  const callees = callGraph.edges
    .filter((e) => e.source === entityName)
    .map((e) => e.target);

  if (callers.length > 0) {
    content += "## Called By\n\n";
    for (const caller of callers) {
      const node = callGraph.nodes.find((n) => n.id === caller || n.name === caller);
      content += `- ${node?.name || caller} (${node?.type || "unknown"})\n`;
    }
    content += "\n";
  }

  if (callees.length > 0) {
    content += "## Calls\n\n";
    for (const callee of callees) {
      const node = callGraph.nodes.find((n) => n.id === callee || n.name === callee);
      content += `- ${node?.name || callee} (${node?.type || "unknown"})\n`;
    }
  }

  return content;
}

function getImpactBadge(level: string): string {
  switch (level) {
    case "critical":
      return "🔴 Critical";
    case "high":
      return "🟠 High";
    case "medium":
      return "🟡 Medium";
    case "low":
      return "🟢 Low";
    default:
      return "⚪ None";
  }
}
