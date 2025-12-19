# Kōdo (コード)

A code-aware AI assistant that understands codebases as interconnected systems, not just text files.

## Features

- **Graph-Based Code Understanding**: Uses Neo4j to model code relationships (calls, imports, inheritance)
- **Semantic Search**: Vector embeddings for natural language code search
- **AI-Powered Queries**: Ask questions about your codebase in plain English
- **Impact Analysis**: Understand what breaks when you change code
- **Data Flow Tracking**: Trace how data moves through your application

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| API | FastAPI |
| Graph DB | Neo4j |
| Vector DB | Qdrant |
| Parsing | Tree-sitter |
| LLM | Claude API |
| Embeddings | Voyage Code 2 |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- uv (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kodo.git
cd kodo

# Install dependencies
uv sync --all-extras

# Start infrastructure
docker compose up -d

# Run the API
uv run uvicorn api.main:app --reload
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

## Development

```bash
# Run tests
uv run pytest

# Run linting
uv run ruff check .

# Run type checking
uv run mypy .

# Format code
uv run ruff format .
```

## Project Structure

```
kodo/
├── core/                 # Core business logic
│   ├── parser/          # Tree-sitter code parsing
│   ├── graph/           # Neo4j graph operations
│   ├── embeddings/      # Vector embeddings
│   ├── ingestion/       # Repository ingestion
│   ├── query/           # Query engine
│   └── llm/             # LLM integration
├── api/                  # FastAPI application
│   ├── routers/         # API endpoints
│   └── middleware/      # Request handling
├── tests/               # Test suite
└── docker-compose.yml   # Infrastructure services
```

## Roadmap

- [x] Phase 1: Core Infrastructure & Parsing
- [ ] Phase 2: Embeddings & Natural Language
- [ ] Phase 3: Deep Code Analysis
- [ ] Phase 4: Code Generation & Integrations

## License

MIT
