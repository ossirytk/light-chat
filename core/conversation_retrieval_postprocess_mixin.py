import re


class ConversationRetrievalPostprocessMixin:
    @staticmethod
    def _describe_where_filter(where: dict[str, object] | None) -> str:
        if where is None:
            return "unfiltered"
        if "$and" in where:
            return "$and"
        if "$or" in where:
            return "$or"
        return "metadata"

    def _is_low_quality_context_chunk(self, chunk: str) -> bool:
        text = chunk.strip()
        if not text:
            return True
        return any(pattern.search(text) for pattern in self._LOW_QUALITY_CONTEXT_PATTERNS)

    def _filter_context_chunks(self, chunks: list[str]) -> list[str]:
        """Remove low-quality boilerplate chunks and exact duplicates while preserving order."""
        filtered: list[str] = []
        seen: set[str] = set()
        seen_signatures: set[str] = set()
        for chunk in chunks:
            normalized = chunk.strip()
            if not normalized:
                continue
            normalized = self._dedupe_chunk_sections(normalized)
            if not normalized:
                continue
            if normalized in seen:
                continue
            if self._is_low_quality_context_chunk(normalized):
                continue
            signature = self._chunk_signature(normalized)
            if signature in seen_signatures:
                continue
            seen.add(normalized)
            seen_signatures.add(signature)
            filtered.append(normalized)
        return filtered

    def _dedupe_chunk_sections(self, chunk: str) -> str:
        """Remove repeated markdown sections inside a single chunk."""
        lines = chunk.splitlines()
        if not lines:
            return chunk

        output_lines: list[str] = []
        current_section: list[str] = []
        seen_section_titles: set[str] = set()
        current_title: str | None = None

        def flush_section() -> None:
            nonlocal current_section, current_title
            if not current_section:
                return
            if current_title is None:
                output_lines.extend(current_section)
            else:
                title_key = current_title.lower().strip()
                if title_key not in seen_section_titles:
                    seen_section_titles.add(title_key)
                    output_lines.extend(current_section)
            current_section = []
            current_title = None

        for line in lines:
            heading_match = self._MARKDOWN_HEADING_RE.match(line.strip())
            if heading_match:
                flush_section()
                current_title = heading_match.group(1)
                current_section = [line]
            elif current_section:
                current_section.append(line)
            else:
                output_lines.append(line)

        flush_section()

        return "\n".join(output_lines).strip()

    def _chunk_signature(self, chunk: str) -> str:
        """Build a lightweight signature for near-duplicate chunk elimination."""
        signature_lines: list[str] = []
        for line in chunk.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Ignore heading markers in signature so equivalent sections with
            # minor heading level differences collapse together.
            stripped = self._MARKDOWN_HEADING_RE.sub(r"\\1", stripped)
            stripped = re.sub(r"\s+", " ", stripped).lower()
            signature_lines.append(stripped)
            if len(signature_lines) >= self._CHUNK_SIGNATURE_LINES:
                break
        return "|".join(signature_lines)

    def _cap_context_text(self, text: str) -> str:
        """Cap vector context length to avoid prompt bloat while preserving boundaries."""
        max_chars = self.runtime_config.max_vector_context_chars
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        clipped = text[:max_chars]
        if "\n\n" in clipped:
            clipped = clipped.rsplit("\n\n", 1)[0]
        elif "\n" in clipped:
            clipped = clipped.rsplit("\n", 1)[0]
        return clipped.strip()

    def _score_context_sentences(self, query_terms: set[str], context: str) -> list[tuple[int, int, str]]:
        blocks = [block.strip() for block in context.split("\n\n") if block.strip()]
        scored_sentences: list[tuple[int, int, str]] = []
        sentence_position = 0

        for block in blocks:
            normalized_block = re.sub(r"\s+", " ", block).strip()
            if not normalized_block:
                continue

            sentences = [s.strip() for s in self._SENTENCE_SPLIT_RE.split(normalized_block) if s.strip()]
            for sentence in sentences:
                sentence_terms = self._query_terms(sentence)
                overlap = len(query_terms.intersection(sentence_terms))
                if overlap > 0:
                    scored_sentences.append((overlap, -sentence_position, sentence))
                sentence_position += 1

        scored_sentences.sort(reverse=True)
        return scored_sentences

    def _compress_context_sentences(self, query: str, context: str) -> str:
        if not context or not self.runtime_config.rag_sentence_compression_enabled:
            return context

        max_sentences = self.runtime_config.rag_sentence_compression_max_sentences
        if max_sentences <= 0:
            return context

        query_terms = self._query_terms(query)
        if not query_terms:
            return context

        scored_sentences = self._score_context_sentences(query_terms, context)
        if not scored_sentences:
            return context

        selected: list[str] = []
        seen_sentences: set[str] = set()
        for _score, _position, sentence in scored_sentences:
            sentence_key = sentence.lower()
            if sentence_key in seen_sentences:
                continue
            seen_sentences.add(sentence_key)
            selected.append(sentence)
            if len(selected) >= max_sentences:
                break

        return "\n\n".join(selected) if selected else context

    def _dedupe_cross_collection_chunks(
        self, context_chunks: list[str], mes_chunks: list[str]
    ) -> tuple[list[str], list[str], int]:
        if not context_chunks or not mes_chunks:
            return context_chunks, mes_chunks, 0

        seen_signatures = {self._chunk_signature(chunk) for chunk in context_chunks}
        deduped_mes: list[str] = []
        removed = 0

        for chunk in mes_chunks:
            signature = self._chunk_signature(chunk)
            if signature in seen_signatures:
                removed += 1
                continue
            seen_signatures.add(signature)
            deduped_mes.append(chunk)

        return context_chunks, deduped_mes, removed
