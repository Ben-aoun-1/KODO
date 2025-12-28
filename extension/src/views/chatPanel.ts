/**
 * Chat panel webview for interacting with Kodo.
 */

import * as vscode from "vscode";
import { getApiClient } from "../api/client";

export class ChatPanelProvider implements vscode.WebviewViewProvider {
  public static readonly viewType = "kodo.chatPanel";

  private _view?: vscode.WebviewView;
  private repoId: string | null = null;
  private messageHistory: { role: "user" | "assistant"; content: string }[] = [];

  constructor(private readonly _extensionUri: vscode.Uri) {}

  setRepoId(repoId: string | null): void {
    this.repoId = repoId;
    if (this._view) {
      this._view.webview.postMessage({
        type: "repoStatus",
        indexed: !!repoId,
      });
    }
  }

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this._view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    webviewView.webview.html = this._getHtmlContent(webviewView.webview);

    // Handle messages from webview
    webviewView.webview.onDidReceiveMessage(async (message) => {
      switch (message.type) {
        case "ask":
          await this.handleAsk(message.question);
          break;
        case "clear":
          this.messageHistory = [];
          break;
      }
    });

    // Send initial status
    webviewView.webview.postMessage({
      type: "repoStatus",
      indexed: !!this.repoId,
    });
  }

  private async handleAsk(question: string): Promise<void> {
    if (!this.repoId) {
      this._view?.webview.postMessage({
        type: "error",
        message: "No repository indexed. Run 'Kodo: Index Current Repository' first.",
      });
      return;
    }

    // Add user message
    this.messageHistory.push({ role: "user", content: question });
    this._view?.webview.postMessage({
      type: "message",
      role: "user",
      content: question,
    });

    // Show loading
    this._view?.webview.postMessage({ type: "loading", show: true });

    try {
      const client = getApiClient();
      const editor = vscode.window.activeTextEditor;

      // Build context
      let context = undefined;
      if (editor) {
        const config = vscode.workspace.getConfiguration("kodo");
        const maxLines = config.get<number>("maxContextLines", 50);
        const selection = editor.selection;

        let startLine = 0;
        let endLine = editor.document.lineCount - 1;
        let code = "";

        if (!selection.isEmpty) {
          // Use selection
          startLine = selection.start.line;
          endLine = selection.end.line;
          code = editor.document.getText(selection);
        } else {
          // Use surrounding context
          const currentLine = editor.selection.active.line;
          startLine = Math.max(0, currentLine - Math.floor(maxLines / 2));
          endLine = Math.min(
            editor.document.lineCount - 1,
            currentLine + Math.floor(maxLines / 2)
          );
          code = editor.document.getText(
            new vscode.Range(startLine, 0, endLine, editor.document.lineAt(endLine).text.length)
          );
        }

        context = {
          filePath: this.getRelativePath(editor.document.fileName),
          startLine: startLine + 1,
          endLine: endLine + 1,
          code,
          language: editor.document.languageId,
        };
      }

      const response = await client.ask({
        question,
        repoId: this.repoId,
        context,
      });

      // Add assistant message
      this.messageHistory.push({ role: "assistant", content: response.answer });
      this._view?.webview.postMessage({
        type: "message",
        role: "assistant",
        content: response.answer,
        sources: response.sources,
        followUp: response.followUpQuestions,
      });
    } catch (error) {
      this._view?.webview.postMessage({
        type: "error",
        message: (error as Error).message,
      });
    } finally {
      this._view?.webview.postMessage({ type: "loading", show: false });
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

  private _getHtmlContent(_webview: vscode.Webview): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kodo Chat</title>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: var(--vscode-font-family);
      font-size: var(--vscode-font-size);
      color: var(--vscode-foreground);
      background: var(--vscode-sideBar-background);
      height: 100vh;
      display: flex;
      flex-direction: column;
    }

    .status-bar {
      padding: 8px 12px;
      background: var(--vscode-statusBar-background);
      font-size: 11px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }

    .status-dot.indexed {
      background: var(--vscode-testing-iconPassed);
    }

    .status-dot.not-indexed {
      background: var(--vscode-testing-iconFailed);
    }

    .chat-container {
      flex: 1;
      overflow-y: auto;
      padding: 12px;
    }

    .message {
      margin-bottom: 16px;
      animation: fadeIn 0.2s ease-in;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .message-header {
      font-size: 11px;
      font-weight: 600;
      margin-bottom: 4px;
      color: var(--vscode-descriptionForeground);
    }

    .message-content {
      padding: 8px 12px;
      border-radius: 6px;
      line-height: 1.5;
    }

    .user .message-content {
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
    }

    .assistant .message-content {
      background: var(--vscode-editor-background);
      border: 1px solid var(--vscode-widget-border);
    }

    .message-content pre {
      background: var(--vscode-textCodeBlock-background);
      padding: 8px;
      border-radius: 4px;
      overflow-x: auto;
      margin: 8px 0;
    }

    .message-content code {
      font-family: var(--vscode-editor-font-family);
      font-size: 12px;
    }

    .sources {
      margin-top: 8px;
      font-size: 11px;
    }

    .sources a {
      color: var(--vscode-textLink-foreground);
      text-decoration: none;
    }

    .sources a:hover {
      text-decoration: underline;
    }

    .follow-up {
      margin-top: 8px;
    }

    .follow-up button {
      background: var(--vscode-button-secondaryBackground);
      color: var(--vscode-button-secondaryForeground);
      border: none;
      padding: 4px 8px;
      border-radius: 4px;
      cursor: pointer;
      margin-right: 4px;
      margin-bottom: 4px;
      font-size: 11px;
    }

    .follow-up button:hover {
      background: var(--vscode-button-secondaryHoverBackground);
    }

    .loading {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px;
      color: var(--vscode-descriptionForeground);
    }

    .spinner {
      width: 16px;
      height: 16px;
      border: 2px solid var(--vscode-progressBar-background);
      border-top-color: var(--vscode-button-background);
      border-radius: 50%;
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    .error {
      background: var(--vscode-inputValidation-errorBackground);
      border: 1px solid var(--vscode-inputValidation-errorBorder);
      padding: 8px 12px;
      border-radius: 6px;
      margin-bottom: 16px;
    }

    .input-container {
      padding: 12px;
      border-top: 1px solid var(--vscode-widget-border);
      background: var(--vscode-sideBar-background);
    }

    .input-wrapper {
      display: flex;
      gap: 8px;
    }

    textarea {
      flex: 1;
      background: var(--vscode-input-background);
      color: var(--vscode-input-foreground);
      border: 1px solid var(--vscode-input-border);
      border-radius: 4px;
      padding: 8px;
      font-family: var(--vscode-font-family);
      font-size: var(--vscode-font-size);
      resize: none;
      min-height: 36px;
      max-height: 120px;
    }

    textarea:focus {
      outline: 1px solid var(--vscode-focusBorder);
    }

    button.send {
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
      border: none;
      border-radius: 4px;
      padding: 8px 16px;
      cursor: pointer;
      font-weight: 500;
    }

    button.send:hover {
      background: var(--vscode-button-hoverBackground);
    }

    button.send:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .welcome {
      text-align: center;
      padding: 24px;
      color: var(--vscode-descriptionForeground);
    }

    .welcome h3 {
      margin-bottom: 8px;
      color: var(--vscode-foreground);
    }
  </style>
</head>
<body>
  <div class="status-bar">
    <span class="status-dot" id="statusDot"></span>
    <span id="statusText">Checking status...</span>
  </div>

  <div class="chat-container" id="chatContainer">
    <div class="welcome">
      <h3>Welcome to Kodo</h3>
      <p>Ask questions about your codebase.</p>
    </div>
  </div>

  <div class="input-container">
    <div class="input-wrapper">
      <textarea
        id="questionInput"
        placeholder="Ask a question..."
        rows="1"
      ></textarea>
      <button class="send" id="sendButton">Send</button>
    </div>
  </div>

  <script>
    const vscode = acquireVsCodeApi();
    const chatContainer = document.getElementById('chatContainer');
    const questionInput = document.getElementById('questionInput');
    const sendButton = document.getElementById('sendButton');
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');

    let isIndexed = false;
    let isLoading = false;

    // Handle messages from extension
    window.addEventListener('message', (event) => {
      const message = event.data;

      switch (message.type) {
        case 'repoStatus':
          isIndexed = message.indexed;
          updateStatus();
          break;

        case 'message':
          addMessage(message.role, message.content, message.sources, message.followUp);
          break;

        case 'error':
          addError(message.message);
          break;

        case 'loading':
          isLoading = message.show;
          updateLoading();
          break;
      }
    });

    function updateStatus() {
      if (isIndexed) {
        statusDot.className = 'status-dot indexed';
        statusText.textContent = 'Repository indexed';
        sendButton.disabled = false;
      } else {
        statusDot.className = 'status-dot not-indexed';
        statusText.textContent = 'Repository not indexed';
        sendButton.disabled = true;
      }
    }

    function updateLoading() {
      const existing = document.querySelector('.loading');
      if (isLoading && !existing) {
        const loading = document.createElement('div');
        loading.className = 'loading';
        loading.innerHTML = '<div class="spinner"></div><span>Thinking...</span>';
        chatContainer.appendChild(loading);
        chatContainer.scrollTop = chatContainer.scrollHeight;
      } else if (!isLoading && existing) {
        existing.remove();
      }
      sendButton.disabled = isLoading || !isIndexed;
    }

    function addMessage(role, content, sources, followUp) {
      // Remove welcome message
      const welcome = chatContainer.querySelector('.welcome');
      if (welcome) welcome.remove();

      const messageDiv = document.createElement('div');
      messageDiv.className = 'message ' + role;

      const header = document.createElement('div');
      header.className = 'message-header';
      header.textContent = role === 'user' ? 'You' : 'Kodo';
      messageDiv.appendChild(header);

      const contentDiv = document.createElement('div');
      contentDiv.className = 'message-content';
      contentDiv.innerHTML = formatContent(content);
      messageDiv.appendChild(contentDiv);

      if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources';
        sourcesDiv.innerHTML = '<strong>Sources:</strong> ' +
          sources.map(s => '<a href="#">' + s.entityName + '</a>').join(', ');
        messageDiv.appendChild(sourcesDiv);
      }

      if (followUp && followUp.length > 0) {
        const followUpDiv = document.createElement('div');
        followUpDiv.className = 'follow-up';
        for (const question of followUp) {
          const btn = document.createElement('button');
          btn.textContent = question;
          btn.onclick = () => {
            questionInput.value = question;
            sendQuestion();
          };
          followUpDiv.appendChild(btn);
        }
        messageDiv.appendChild(followUpDiv);
      }

      chatContainer.appendChild(messageDiv);
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function addError(message) {
      const errorDiv = document.createElement('div');
      errorDiv.className = 'error';
      errorDiv.textContent = message;
      chatContainer.appendChild(errorDiv);
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function formatContent(content) {
      // Simple markdown-like formatting
      return content
        .replace(/\`\`\`(\w+)?\n([\s\S]*?)\`\`\`/g, '<pre><code>$2</code></pre>')
        .replace(/\`([^\`]+)\`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
    }

    function sendQuestion() {
      const question = questionInput.value.trim();
      if (!question || isLoading || !isIndexed) return;

      vscode.postMessage({ type: 'ask', question });
      questionInput.value = '';
      questionInput.style.height = 'auto';
    }

    // Event listeners
    sendButton.onclick = sendQuestion;

    questionInput.onkeydown = (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
      }
    };

    questionInput.oninput = () => {
      questionInput.style.height = 'auto';
      questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + 'px';
    };

    // Initial status
    updateStatus();
  </script>
</body>
</html>`;
  }
}
