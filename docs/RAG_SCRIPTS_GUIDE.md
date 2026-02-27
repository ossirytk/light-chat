# RAG Scripts Guide

This guide describes the comprehensive RAG (Retrieval Augmented Generation) scripts for managing text data, metadata, and ChromaDB collections in the light-chat character AI chatbot.

## Overview

The RAG system consists of three main scripts with enhanced capabilities:

1. **scripts/rag/analyze_rag_text.py** - Analyze text files, extract metadata and topics
2. **scripts/rag/push_rag_data.py** - Push RAG data to ChromaDB with enrichment
3. **scripts/rag/manage_collections.py** - Manage ChromaDB collections (CRUD operations)

## Prerequisites

Install required dependencies:

```bash
pip install click loguru chromadb langchain langchain-chroma langchain-community langchain-huggingface sentence-transformers
```

Or using uv (recommended):

```bash
uv sync
```

## Script 1: scripts/rag/analyze_rag_text.py

Analyze text files to extract metadata, topics, and validate existing metadata files.

### Commands

#### analyze

Analyze a text file and extract metadata and topics.

```bash
uv run python scripts/rag/analyze_rag_text.py analyze <FILE_PATH> [OPTIONS]
```

**Options:**
- `--output`, `-o`: Output JSON file for extracted metadata
- `--min-freq`, `-f`: Minimum frequency for key phrase extraction (default: 3)
- `--auto-categories/--no-auto-categories`: Enable/disable category generation (default: enabled)
- `--auto-aliases/--no-auto-aliases`: Enable/disable alias generation (default: enabled)
- `--max-aliases`: Maximum aliases per entry (default: 5)
- `--strict`: Keep only high-confidence category/alias enrichments
- `--review-report`: Output JSON report with kept/dropped enrichment decisions
- `--verbose`, `-v`: Show detailed analysis results

**Examples:**

```bash
# Basic analysis
uv run python scripts/rag/analyze_rag_text.py analyze rag_data/shodan.txt

# Detailed analysis with output
uv run python scripts/rag/analyze_rag_text.py analyze rag_data/shodan.txt -v -o rag_data/shodan_extracted.json

# Extract with custom frequency threshold
uv run python scripts/rag/analyze_rag_text.py analyze rag_data/shodan.txt -f 5 -o output.json

# Strict enrichment with review report
uv run python scripts/rag/analyze_rag_text.py analyze rag_data/shodan.txt \
  --strict \
  --review-report rag_data/shodan_enrichment_review.json \
  -o rag_data/shodan_generated.json
```

**Output:**
- Total character, line, word counts
- Named entities (capitalized phrases, dates, numbers)
- Key phrases (frequently occurring bigrams and trigrams)
- Potential metadata entries with UUIDs
- Auto-enriched metadata includes `category` and optional `aliases` by default
- Optional enrichment review report (`--review-report`) includes confidence and keep/drop decisions

#### validate

Validate a metadata JSON file structure.

```bash
uv run python scripts/rag/analyze_rag_text.py validate <METADATA_PATH>
```

**Examples:**

```bash
# Validate metadata file
uv run python scripts/rag/analyze_rag_text.py validate rag_data/shodan.json
```

**Validation checks:**
- JSON syntax validity
- Required `uuid` field presence
- Text field presence (text, content, value, etc.)
- Duplicate UUID detection
- List structure validation

#### scan

Scan a directory for text files and their associated metadata.

```bash
uv run python scripts/rag/analyze_rag_text.py scan <DIRECTORY> [OPTIONS]
```

**Options:**
- `--auto-generate`, `-g`: Auto-generate missing metadata files
- `--auto-categories/--no-auto-categories`: Enable/disable category generation (default: enabled)
- `--auto-aliases/--no-auto-aliases`: Enable/disable alias generation (default: enabled)
- `--max-aliases`: Maximum aliases per entry (default: 5)
- `--strict`: Keep only high-confidence category/alias enrichments

**Examples:**

```bash
# Scan directory
uv run python scripts/rag/analyze_rag_text.py scan rag_data/

# Scan and auto-generate missing metadata
uv run python scripts/rag/analyze_rag_text.py scan rag_data/ --auto-generate
```

**Output:**
- Lists all .txt files
- Checks for corresponding .json metadata files
- Validates existing metadata
- Optionally generates missing metadata

---

## Script 2: scripts/rag/push_rag_data.py

Enhanced script for pushing individual text files to ChromaDB collections with metadata enrichment.

### Usage

```bash
uv run python scripts/rag/push_rag_data.py <FILE_PATH> -c <COLLECTION_NAME> [OPTIONS]
```

### Options

- `-c, --collection-name`: Name of the ChromaDB collection to create **(required)**
- `-p, --persist-directory`: Directory where ChromaDB stores collections
- `-k, --key-storage`: Directory containing metadata JSON files
- `-m, --metadata-file`: Specific metadata JSON file to use (overrides auto-detection)
- `-cs, --chunk-size`: Chunk size for text splitting (default: 2048)
- `-co, --chunk-overlap`: Overlap size for chunks (default: 1024)
- `-t, --threads`: Number of threads for processing (default: 6)
- `-d, --dry-run`: Show what would be done without making changes
- `-w, --overwrite`: Overwrite existing collection if it exists

### Examples

```bash
# Basic upload
uv run python scripts/rag/push_rag_data.py rag_data/shodan.txt -c shodan

# Upload with custom settings
uv run python scripts/rag/push_rag_data.py rag_data/shodan.txt -c shodan_v2 -cs 1024 -co 512 -t 8

# Dry run to preview
uv run python scripts/rag/push_rag_data.py rag_data/shodan.txt -c shodan -d

# Upload with specific metadata file
uv run python scripts/rag/push_rag_data.py rag_data/shodan.txt -c shodan -m rag_data/custom_metadata.json

# Overwrite existing collection
uv run python scripts/rag/push_rag_data.py rag_data/shodan.txt -c shodan -w
```

### Features

- **Single file processing**: Focused on one file at a time with explicit collection naming
- **Dry-run mode**: Test configuration without making changes
- **Overwrite protection**: Prevents accidental collection deletion
- **Custom metadata**: Specify exact metadata file to use
- **Progress tracking**: Detailed logging of each step
- **Auto-detection**: Automatically finds matching metadata files

### Metadata Auto-Detection

The script automatically looks for metadata files based on the input filename:

- Input: `shodan.txt` → Metadata: `shodan.json`
- Input: `shodan_message_examples.txt` → Metadata: `shodan.json`

---

## Script 3: scripts/rag/manage_collections.py

Enhanced collection management with comprehensive CRUD operations.

### Commands

#### list-collections

List all ChromaDB collections.

```bash
uv run python scripts/rag/manage_collections.py list-collections [OPTIONS]
```

**Options:**
- `-p, --persist-directory`: Directory where ChromaDB stores collections
- `-v, --verbose`: Show detailed collection information

**Examples:**

```bash
# Basic listing
uv run python scripts/rag/manage_collections.py list-collections

# Detailed information
uv run python scripts/rag/manage_collections.py list-collections -v
```

#### delete

Delete a single ChromaDB collection.

```bash
uv run python scripts/rag/manage_collections.py delete <COLLECTION_NAME> [OPTIONS]
```

**Options:**
- `-p, --persist-directory`: Directory where ChromaDB stores collections
- `-y, --yes`: Skip confirmation prompt

**Examples:**

```bash
# Delete with confirmation
uv run python scripts/rag/manage_collections.py delete shodan_old

# Delete without confirmation
uv run python scripts/rag/manage_collections.py delete shodan_old -y
```

#### delete-multiple

Delete multiple collections matching a pattern.

```bash
uv run python scripts/rag/manage_collections.py delete-multiple [OPTIONS]
```

**Options:**
- `-p, --persist-directory`: Directory where ChromaDB stores collections
- `--pattern`: Delete collections matching pattern (use * for wildcard) **(required)**
- `-y, --yes`: Skip confirmation prompt

**Examples:**

```bash
# Delete all test collections
uv run python scripts/rag/manage_collections.py delete-multiple --pattern "test_*"

# Delete all version 1 collections
uv run python scripts/rag/manage_collections.py delete-multiple --pattern "*_v1" -y
```

#### test

Test a collection with a similarity search query.

```bash
uv run python scripts/rag/manage_collections.py test <COLLECTION_NAME> -q <QUERY> [OPTIONS]
```

**Options:**
- `-q, --query`: Query text to search **(required)**
- `-k`: Number of results to return (default: 5)
- `-p, --persist-directory`: Directory where ChromaDB stores collections
- `-k, --key-storage`: Directory containing metadata JSON files

**Examples:**

```bash
# Basic search
uv run python scripts/rag/manage_collections.py test shodan -q "SHODAN artificial intelligence"

# Search with more results
uv run python scripts/rag/manage_collections.py test shodan -q "Von Braun starship" -k 10

# Search with metadata filtering
uv run python scripts/rag/manage_collections.py test shodan -q "XERXES system 2114" -k 5
```

#### export

Export collection data to JSON.

```bash
uv run python scripts/rag/manage_collections.py export <COLLECTION_NAME> -o <OUTPUT_FILE> [OPTIONS]
```

**Options:**
- `-o, --output`: Output JSON file **(required)**
- `-p, --persist-directory`: Directory where ChromaDB stores collections

**Examples:**

```bash
# Export collection
uv run python scripts/rag/manage_collections.py export shodan -o shodan_backup.json

# Export to custom directory
uv run python scripts/rag/manage_collections.py export shodan_mes -o backups/shodan_mes_$(date +%Y%m%d).json
```

#### info

Show detailed information about a collection.

```bash
uv run python scripts/rag/manage_collections.py info <COLLECTION_NAME> [OPTIONS]
```

**Options:**
- `-p, --persist-directory`: Directory where ChromaDB stores collections

**Examples:**

```bash
# Show collection info
uv run python scripts/rag/manage_collections.py info shodan

# Show info for message examples collection
uv run python scripts/rag/manage_collections.py info shodan_mes
```

---

## Configuration

All scripts use the application configuration from `configs/appconf.json`. Default values:

```json
{
  "PERSIST_DIRECTORY": "./character_storage/",
  "KEY_STORAGE": "./rag_data/",
  "EMBEDDING_CACHE": "./embedding_models/",
  "EMBEDDING_DEVICE": "cpu",
  "DOCUMENTS_DIRECTORY": "./rag_data/",
  "CHUNK_SIZE": 2048,
  "CHUNK_OVERLAP": 1024,
  "THREADS": 6,
  "LOG_LEVEL": "DEBUG",
  "SHOW_LOGS": true
}
```

## Metadata File Format

Metadata files should follow this structure:

```json
[
  {
    "uuid": "unique-identifier-1",
    "text": "Searchable text content"
  },
  {
    "uuid": "unique-identifier-2",
    "text": "Another searchable term"
  }
]
```

**Alternative format with Content wrapper:**

```json
{
  "Content": [
    {"uuid": "id-1", "text": "Content 1"},
    {"uuid": "id-2", "text": "Content 2"}
  ]
}
```

**Supported text field names:**
- `text`
- `text_fields`
- `text_field`
- `content`
- `value`

## Common Workflows

### Workflow 1: Adding New Character Data

```bash
# 1. Analyze the text file
uv run python scripts/rag/analyze_rag_text.py analyze new_character.txt -v -o new_character_metadata.json

# 2. Review and edit the generated metadata file as needed

# 3. Validate the metadata
uv run python scripts/rag/analyze_rag_text.py validate new_character_metadata.json

# 4. Test upload with dry-run
uv run python scripts/rag/push_rag_data.py new_character.txt -c new_character -d

# 5. Perform actual upload
uv run python scripts/rag/push_rag_data.py new_character.txt -c new_character

# 6. Test the collection
uv run python scripts/rag/manage_collections.py test new_character -q "test query" -k 5

# 7. View collection info
uv run python scripts/rag/manage_collections.py info new_character
```

### Workflow 2: Updating Existing Collection

```bash
# 1. Export current collection as backup
uv run python scripts/rag/manage_collections.py export shodan -o backups/shodan_backup_$(date +%Y%m%d).json

# 2. Update the source text file as needed

# 3. Push with overwrite
uv run python scripts/rag/push_rag_data.py rag_data/shodan.txt -c shodan -w

# 4. Test the updated collection
uv run python scripts/rag/manage_collections.py test shodan -q "test query"
```

### Workflow 3: Batch Analysis

```bash
# Scan directory and generate missing metadata
uv run python scripts/rag/analyze_rag_text.py scan rag_data/ --auto-generate

# Validate all metadata files
for json_file in rag_data/*.json; do
  uv run python scripts/rag/analyze_rag_text.py validate "$json_file"
done
```

### Workflow 4: Collection Maintenance

```bash
# List all collections
uv run python scripts/rag/manage_collections.py list-collections -v

# Delete old test collections
uv run python scripts/rag/manage_collections.py delete-multiple --pattern "test_*" -y

# Export all collections
for collection in shodan shodan_mes; do
  uv run python scripts/rag/manage_collections.py export "$collection" -o "backups/${collection}.json"
done
```

## Troubleshooting

### Issue: No output from scripts

**Solution:** Check `SHOW_LOGS` in `configs/appconf.json`. Set to `true` to see output.

### Issue: Collection already exists error

**Solution:** Use the `--overwrite` flag with `scripts/rag/push_rag_data.py` or delete the collection first:

```bash
uv run python scripts/rag/manage_collections.py delete <collection_name> -y
```

### Issue: Metadata not being applied

**Solution:** 
1. Validate metadata file: `uv run python scripts/rag/analyze_rag_text.py validate <metadata_file>`
2. Ensure filename matches (e.g., `shodan.txt` uses `shodan.json`)
3. Use `--metadata-file` to explicitly specify metadata

### Issue: Out of memory during processing

**Solution:** Reduce chunk size and thread count:

```bash
uv run python scripts/rag/push_rag_data.py file.txt -c collection -cs 1024 -t 2
```

## Advanced Usage

### Custom Embedding Models

Modify `configs/appconf.json` to change the embedding model:

```json
{
  "EMBEDDING_DEVICE": "cuda",  // Use GPU
  "EMBEDDING_CACHE": "./custom_models/"
}
```

### Parallel Processing

Process multiple files in parallel using shell scripting:

```bash
# Process all txt files in parallel
for txt_file in rag_data/*.txt; do
  base_name=$(basename "$txt_file" .txt)
  uv run python scripts/rag/push_rag_data.py "$txt_file" -c "$base_name" &
done
wait
```

### Integration with scripts/rag/old_prepare_rag.py

The new scripts complement the existing `scripts/rag/old_prepare_rag.py`:

- **scripts/rag/old_prepare_rag.py**: Batch processing of all files in a directory
- **scripts/rag/push_rag_data.py**: Single file processing with more control

Use `scripts/rag/old_prepare_rag.py` for initial bulk setup, then use `scripts/rag/push_rag_data.py` for updates.

## Best Practices

1. **Always validate metadata** before pushing to ChromaDB
2. **Use dry-run mode** to test configurations
3. **Export collections** before overwriting
4. **Use descriptive collection names** (e.g., `character_v2`, `character_messages`)
5. **Keep metadata files** in sync with text files
6. **Monitor collection sizes** with `list-collections -v`
7. **Test collections** after updates with sample queries

## See Also

- [scripts/rag/old_prepare_rag.py](../scripts/rag/old_prepare_rag.py) - Original batch processing script
- [core/collection_helper.py](../core/collection_helper.py) - Original collection helper
- [core/context_manager.py](../core/context_manager.py) - Runtime RAG retrieval

## Support

For issues or questions, refer to the project README or open an issue on GitHub.
