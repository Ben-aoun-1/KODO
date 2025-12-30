# Kodo (コード)

A code-aware AI assistant that understands codebases as interconnected systems, not just text files.

## Features

- **Graph-Based Code Understanding**: Uses Neo4j to model code relationships (calls, imports, inheritance)
- **Semantic Search**: Vector embeddings for natural language code search via Qdrant
- **AI-Powered Queries**: Ask questions about your codebase in plain English using Claude
- **Multi-Language Parsing**: Tree-sitter based parsing for Python, JavaScript, TypeScript
- **Incremental Indexing**: Only re-parse changed files using git diff

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| API | FastAPI |
| Graph DB | Neo4j 5.x |
| Vector DB | Qdrant |
| Parsing | Tree-sitter |
| LLM | Claude API |
| Embeddings | Voyage Code 2 / OpenAI |

## Current Status

### Completed

- [x] Project setup with Poetry/uv, Docker Compose
- [x] Tree-sitter parsing for Python, JavaScript, TypeScript
- [x] Neo4j schema and graph operations (nodes, relationships)
- [x] Repository ingestion pipeline (discovery, parsing, storage)
- [x] FastAPI endpoints for repository management
- [x] Query engine with routing and handlers
- [x] LLM integration (Claude) for AI-powered responses
- [x] Embedding generation and Qdrant storage
- [x] Semantic search endpoint

### In Progress

- [ ] Impact analysis for code changes
- [ ] Data flow tracking
- [ ] VS Code extension

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Poetry or uv

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kodo.git
cd kodo

# Create virtual environment and install dependencies
python -m venv .venv
.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -e .

# Start infrastructure (Neo4j, Qdrant)
docker compose up -d

# Run the API
python -m uvicorn api.main:app --reload --port 8000
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Optional - for AI features
ANTHROPIC_API_KEY=sk-ant-...      # For Claude LLM
VOYAGE_API_KEY=vo-...              # For code embeddings (preferred)
OPENAI_API_KEY=sk-...              # Alternative embeddings
```

## API Endpoints

### Health
- `GET /health` - Health check

### Repositories
- `GET /api/v1/repos` - List all repositories
- `POST /api/v1/repos` - Add a repository
- `GET /api/v1/repos/{repo_id}` - Get repository details
- `DELETE /api/v1/repos/{repo_id}` - Delete a repository
- `POST /api/v1/repos/{repo_id}/sync` - Trigger indexing

### Code Entities
- `GET /api/v1/repos/{repo_id}/files` - List files
- `GET /api/v1/repos/{repo_id}/functions` - List functions
- `GET /api/v1/repos/{repo_id}/classes` - List classes

### Graph Queries
- `GET /api/v1/repos/{repo_id}/graph/callers/{function_name}` - Find callers
- `GET /api/v1/repos/{repo_id}/graph/callees/{function_name}` - Find callees
- `GET /api/v1/repos/{repo_id}/graph/impact/{entity_name}` - Impact analysis

### Natural Language
- `POST /api/v1/repos/{repo_id}/ask` - Ask a question about the code
- `POST /api/v1/repos/{repo_id}/ask/search` - Semantic code search
- `GET /api/v1/repos/{repo_id}/ask/classify` - Classify query type

## Project Structure

```
kodo/
├── core/                 # Core business logic
│   ├── parser/          # Tree-sitter code parsing
│   ├── graph/           # Neo4j graph operations
│   ├── embeddings/      # Vector embeddings (Voyage/OpenAI)
│   ├── ingestion/       # Repository ingestion pipeline
│   ├── query/           # Query engine with handlers
│   └── llm/             # Claude LLM integration
├── api/                  # FastAPI application
│   ├── routers/         # API endpoints
│   ├── services/        # Business logic (indexing)
│   └── dependencies.py  # Dependency injection
├── tests/               # Test suite
└── docker-compose.yml   # Neo4j, Qdrant services
```

## Development

```bash
# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy .

# Format code
ruff format .
```

## Example Usage

```bash
# Add a repository
curl -X POST http://localhost:8000/api/v1/repos \
  -H "Content-Type: application/json" \
  -d '{"name": "my-project", "url": "https://github.com/user/repo", "local_path": "/path/to/repo"}'

# Trigger indexing
curl -X POST http://localhost:8000/api/v1/repos/repo:my-project/sync

# Ask a question (requires ANTHROPIC_API_KEY)
curl -X POST http://localhost:8000/api/v1/repos/repo:my-project/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the main function do?"}'

# Semantic search (requires VOYAGE_API_KEY or OPENAI_API_KEY)
curl -X POST http://localhost:8000/api/v1/repos/repo:my-project/ask/search \
  -H "Content-Type: application/json" \
  -d '{"query": "error handling"}'
```

## License

MIT
