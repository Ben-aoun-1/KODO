# Kodo VS Code Extension

A code-aware AI assistant that understands your codebase.

## Features

- **Ask Questions**: Ask natural language questions about your code
- **Code Explanation**: Select code and get AI-powered explanations
- **Smart Hover**: View function signatures, docstrings, and metrics on hover
- **Code Lens**: See reference counts and analyze functions inline
- **Impact Analysis**: Understand how changes will affect your codebase
- **Call Graph**: Visualize function call relationships

## Requirements

- VS Code 1.85.0 or higher
- Kodo server running (default: http://localhost:8000)

## Getting Started

1. Install the extension
2. Open a workspace folder
3. Run "Kodo: Index Current Repository" from the command palette
4. Start asking questions!

## Commands

| Command | Description | Shortcut |
|---------|-------------|----------|
| `Kodo: Ask a Question` | Ask about your code | `Ctrl+Shift+K` |
| `Kodo: Explain Selected Code` | Explain selection | `Ctrl+Shift+E` |
| `Kodo: Find All References` | Find entity references | - |
| `Kodo: Analyze Impact` | Analyze change impact | - |
| `Kodo: Show Call Graph` | View call relationships | - |
| `Kodo: Index Repository` | Index current workspace | - |

## Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `kodo.serverUrl` | Kodo API server URL | `http://localhost:8000` |
| `kodo.apiKey` | API key for authentication | - |
| `kodo.enableHover` | Enable hover information | `true` |
| `kodo.enableCodeLens` | Enable code lens | `true` |
| `kodo.autoIndex` | Auto-index on open | `false` |
| `kodo.maxContextLines` | Max context lines | `50` |

## Development

```bash
# Install dependencies
npm install

# Compile
npm run compile

# Watch mode
npm run watch

# Run tests
npm test

# Package extension
npm run package
```

## License

MIT
