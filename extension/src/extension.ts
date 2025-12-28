/**
 * Kodo VS Code Extension
 *
 * A code-aware AI assistant that understands your codebase.
 */

import * as vscode from "vscode";
import { getApiClient, resetApiClient } from "./api/client";
import { KodoHoverProvider } from "./providers/hoverProvider";
import { KodoCodeLensProvider } from "./providers/codeLensProvider";
import { ChatPanelProvider } from "./views/chatPanel";
import { registerCommands } from "./commands";

// Global state
let repoId: string | null = null;
let hoverProvider: KodoHoverProvider;
let codeLensProvider: KodoCodeLensProvider;
let chatPanelProvider: ChatPanelProvider;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  console.log("Kodo extension activating...");

  // Initialize providers
  hoverProvider = new KodoHoverProvider();
  codeLensProvider = new KodoCodeLensProvider();
  chatPanelProvider = new ChatPanelProvider(context.extensionUri);

  // Register hover provider for supported languages
  const languages = [
    "python",
    "javascript",
    "typescript",
    "javascriptreact",
    "typescriptreact",
    "go",
    "rust",
  ];

  for (const language of languages) {
    context.subscriptions.push(
      vscode.languages.registerHoverProvider(
        { scheme: "file", language },
        hoverProvider
      )
    );

    context.subscriptions.push(
      vscode.languages.registerCodeLensProvider(
        { scheme: "file", language },
        codeLensProvider
      )
    );
  }

  // Register webview provider
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      ChatPanelProvider.viewType,
      chatPanelProvider
    )
  );

  // Register commands
  registerCommands(context, {
    repoId,
    setRepoId: (id: string | null) => {
      repoId = id;
      hoverProvider.setRepoId(id);
      codeLensProvider.setRepoId(id);
      chatPanelProvider.setRepoId(id);
    },
  });

  // Check for existing repository on activation
  await checkForExistingRepository();

  // Listen for workspace changes
  context.subscriptions.push(
    vscode.workspace.onDidChangeWorkspaceFolders(async () => {
      await checkForExistingRepository();
    })
  );

  // Auto-index if configured
  const config = vscode.workspace.getConfiguration("kodo");
  if (config.get<boolean>("autoIndex", false)) {
    await vscode.commands.executeCommand("kodo.indexRepository");
  }

  // Status bar item
  const statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100
  );
  statusBarItem.command = "kodo.showPanel";
  statusBarItem.text = "$(book) Kodo";
  statusBarItem.tooltip = "Kodo - Code Assistant";
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  // Update status bar based on repository status
  updateStatusBar(statusBarItem);

  console.log("Kodo extension activated");
}

export function deactivate(): void {
  console.log("Kodo extension deactivating...");
  resetApiClient();
}

async function checkForExistingRepository(): Promise<void> {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders || workspaceFolders.length === 0) {
    return;
  }

  try {
    const client = getApiClient();

    // Check server health first
    try {
      await client.checkHealth();
    } catch {
      console.log("Kodo server not available");
      return;
    }

    // Check if workspace is already indexed
    const repo = await client.findRepositoryByPath(
      workspaceFolders[0].uri.fsPath
    );

    if (repo) {
      repoId = repo.id;
      hoverProvider.setRepoId(repo.id);
      codeLensProvider.setRepoId(repo.id);
      chatPanelProvider.setRepoId(repo.id);
      console.log(`Found indexed repository: ${repo.name}`);
    }
  } catch (error) {
    console.error("Error checking for repository:", error);
  }
}

function updateStatusBar(statusBarItem: vscode.StatusBarItem): void {
  if (repoId) {
    statusBarItem.text = "$(book) Kodo";
    statusBarItem.backgroundColor = undefined;
  } else {
    statusBarItem.text = "$(book) Kodo (not indexed)";
    statusBarItem.backgroundColor = new vscode.ThemeColor(
      "statusBarItem.warningBackground"
    );
  }
}
