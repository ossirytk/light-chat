# Light Chat - Character AI Chatbot

A character AI chatbot system using ChromaDB for RAG (Retrieval Augmented Generation) with comprehensive text analysis and collection management tools.

## Features

- **RAG-powered conversations**: Uses ChromaDB vector database for context retrieval
- **Character-specific knowledge**: Manages character data and message examples
- **Comprehensive analysis tools**: Extract metadata and topics from text
- **Collection management**: Full CRUD operations for ChromaDB collections
- **Metadata enrichment**: Automatic metadata tagging for improved retrieval
- **Flexible configuration**: Configurable chunking, embedding, and processing

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Basic Usage

1. **Prepare RAG data** (batch processing):
```bash
python prepare_rag.py
```

2. **Analyze text and extract metadata**:
```bash
python analyze_rag_text.py analyze rag_data/shodan.txt -v
```

3. **Push single file to ChromaDB**:
```bash
python push_rag_data.py rag_data/shodan.txt -c shodan
```

4. **Manage collections**:
```bash
# List all collections
python manage_collections.py list-collections -v

# Test a collection
python manage_collections.py test shodan -q "SHODAN artificial intelligence"

# Delete a collection
python manage_collections.py delete old_collection
```

## RAG Scripts

### 1. analyze_rag_text.py

Analyze text files to extract metadata and topics:

- Extract named entities (capitalized phrases, dates, numbers)
- Identify key phrases (frequently occurring terms)
- Generate metadata JSON files
- Validate existing metadata files
- Scan directories for missing metadata

**Commands**: `analyze`, `validate`, `scan`

### 2. push_rag_data.py

Push individual text files to ChromaDB with enhanced features:

- Single file processing with explicit collection naming
- Dry-run mode for testing
- Overwrite protection
- Custom metadata file selection
- Detailed progress tracking

### 3. manage_collections.py

Comprehensive collection management:

- List collections with statistics
- Delete single or multiple collections
- Test collections with similarity search
- Export collection data to JSON
- Show detailed collection information

### 4. prepare_rag.py

Original batch processing script:

- Process all text files in a directory
- Create collections for base files and message examples
- Parallel metadata enrichment
- Automatic collection naming

### 5. collection_helper.py

Original collection helper:

- List, delete, and test collections
- Metadata-based filtering
- Simple command-line interface

## Documentation

See [docs/RAG_SCRIPTS_GUIDE.md](docs/RAG_SCRIPTS_GUIDE.md) for comprehensive documentation including:

- Detailed command reference
- Configuration options
- Common workflows
- Troubleshooting guide
- Best practices

## Project Structure

```
light-chat/
├── rag_data/              # RAG text files and metadata
│   ├── shodan.txt         # Character context data
│   ├── shodan_message_examples.txt
│   └── shodan.json        # Metadata keys
├── configs/               # Configuration files
│   └── appconf.json       # Application configuration
├── character_storage/     # ChromaDB persistent storage
├── docs/                  # Documentation
│   └── RAG_SCRIPTS_GUIDE.md
├── analyze_rag_text.py    # Text analysis and metadata extraction
├── push_rag_data.py       # Single file upload to ChromaDB
├── manage_collections.py  # Collection management
├── prepare_rag.py         # Batch processing script
├── collection_helper.py   # Original collection helper
├── context_manager.py     # Runtime RAG retrieval
├── conversation_manager.py # Conversation handling
└── main.py                # Main application
```

## Configuration

Edit `configs/appconf.json` to customize:

```json
{
  "PERSIST_DIRECTORY": "./character_storage/",
  "KEY_STORAGE": "./rag_data/",
  "DOCUMENTS_DIRECTORY": "./rag_data/",
  "CHUNK_SIZE": 2048,
  "CHUNK_OVERLAP": 1024,
  "THREADS": 6,
  "EMBEDDING_DEVICE": "cpu",
  "RAG_COLLECTION": "shodan",
  "RAG_K": 2
}
```

## Metadata Format

Metadata files should follow this structure:

```json
[
  {
    "uuid": "unique-identifier",
    "text": "Searchable content"
  }
]
```

Supported text field names: `text`, `content`, `value`, `text_field`, `text_fields`

## Common Workflows

### Adding New Character Data

```bash
# 1. Analyze and extract metadata
python analyze_rag_text.py analyze new_character.txt -o new_character.json

# 2. Validate metadata
python analyze_rag_text.py validate new_character.json

# 3. Push to ChromaDB
python push_rag_data.py new_character.txt -c new_character

# 4. Test the collection
python manage_collections.py test new_character -q "test query"
```

### Updating Existing Collection

```bash
# 1. Backup
python manage_collections.py export shodan -o shodan_backup.json

# 2. Update with overwrite
python push_rag_data.py rag_data/shodan.txt -c shodan -w

# 3. Test
python manage_collections.py test shodan -q "verification query"
```

### Batch Processing

```bash
# Scan and auto-generate missing metadata
python analyze_rag_text.py scan rag_data/ --auto-generate

# Process all files
python prepare_rag.py
```

## Advanced Features

### Dry-Run Mode

Test configuration without making changes:

```bash
python push_rag_data.py file.txt -c collection -d
```

### Custom Chunk Sizes

Optimize for your use case:

```bash
python push_rag_data.py file.txt -c collection -cs 1024 -co 512
```

### Metadata Filtering

Search with metadata filters:

```bash
python manage_collections.py test collection -q "query with metadata"
```

### Bulk Operations

Delete multiple collections:

```bash
python manage_collections.py delete-multiple --pattern "test_*" -y
```

## Dependencies

- **chromadb**: Vector database for embeddings
- **langchain**: RAG orchestration and document processing
- **langchain-chroma**: ChromaDB integration
- **langchain-huggingface**: Embedding models
- **sentence-transformers**: Embedding backend
- **click**: CLI framework
- **loguru**: Logging

## Development

### Running Tests

```bash
# Test scripts with sample data
python analyze_rag_text.py analyze rag_data/shodan.txt -v
python manage_collections.py list-collections -v
```

### Linting

```bash
# Using ruff
ruff check .
ruff format .
```

## Troubleshooting

**No output from scripts?**
- Set `SHOW_LOGS: true` in `configs/appconf.json`

**Collection already exists?**
- Use `--overwrite` flag or delete first: `python manage_collections.py delete <name> -y`

**Out of memory?**
- Reduce chunk size: `--chunk-size 1024`
- Reduce threads: `--threads 2`

**Metadata not applied?**
- Validate: `python analyze_rag_text.py validate <file.json>`
- Check filename matches (e.g., `shodan.txt` → `shodan.json`)

## Contributing

Contributions are welcome! Please:

1. Follow the existing code style (ruff configuration)
2. Add tests for new features
3. Update documentation
4. Submit pull requests

## License

See [LICENSE](LICENSE) for details.

## Support

For issues or questions:
- Check [docs/RAG_SCRIPTS_GUIDE.md](docs/RAG_SCRIPTS_GUIDE.md)
- Open an issue on GitHub
