# Kodo Web Dashboard

A modern React dashboard for the Kodo code-aware AI assistant.

## Features

- **Dashboard**: Overview of repository stats, language distribution, and quick actions
- **Repositories**: Manage indexed repositories - add, refresh, and delete
- **Code Explorer**: Browse file tree, view entities, and search code semantically
- **Chat**: Natural language interface to ask questions about your codebase
- **Call Graph**: Interactive D3.js visualization of function call relationships
- **Analytics**: Code metrics, complexity analysis, coupling, and security insights

## Tech Stack

- React 18 with TypeScript
- Vite for build tooling
- Tailwind CSS for styling
- shadcn/ui components (Radix UI primitives)
- D3.js for graph visualization
- Zustand for state management
- Axios for API requests
- React Router for navigation

## Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
web/
├── src/
│   ├── api/           # API client
│   ├── components/
│   │   ├── layout/    # Header, Sidebar, Layout
│   │   └── ui/        # shadcn/ui components
│   ├── lib/           # Utilities
│   ├── pages/         # Route pages
│   ├── store/         # Zustand store
│   ├── types/         # TypeScript types
│   ├── App.tsx        # Main app with routes
│   ├── main.tsx       # Entry point
│   └── index.css      # Global styles + Tailwind
├── index.html
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
└── package.json
```

## Configuration

The dashboard expects the Kodo API server to be running. By default, it proxies `/api` requests to `http://localhost:8000`.

Set `VITE_API_URL` environment variable to configure a different API endpoint.

## Pages

### Dashboard (`/`)
- Repository stats (files, lines, functions, classes)
- Language distribution chart
- Quick action buttons
- Recent queries list

### Repositories (`/repositories`)
- List of indexed repositories
- Add new repository via GitHub URL
- Refresh or delete repositories
- Repository status indicators

### Explorer (`/explorer`)
- File tree navigation
- Entity list per file (functions, classes)
- Code viewer with syntax highlighting
- Semantic search across codebase

### Chat (`/chat`)
- Natural language Q&A interface
- Markdown rendering for responses
- Source code references
- Suggested questions

### Call Graph (`/graph`)
- D3.js force-directed graph
- Zoom/pan controls
- Node type legend
- Edge type indicators
- Click to explore relationships

### Analytics (`/analytics`)
- Complexity distribution
- Module coupling analysis
- Language breakdown
- Security vulnerability scanning

## License

MIT
