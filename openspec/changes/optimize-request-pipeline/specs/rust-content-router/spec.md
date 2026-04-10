## ADDED Requirements

### Requirement: Content routing executes in Rust with parallel AST parsing

The system SHALL perform content routing (language detection, code identification, structure parsing) in a Rust worker that uses tree-sitter for parallel AST parsing. The Rust implementation SHALL process multiple content blocks concurrently using rayon iterators and SHALL release the GIL during computation.

#### Scenario: Parallel language detection

- **WHEN** a request contains 10 content blocks of different types (Python, JavaScript, Markdown, etc.)
- **THEN** the Rust content router SHALL detect each language in parallel using rayon
- **AND** all detections SHALL complete before returning results

#### Scenario: Code block extraction

- **WHEN** content contains multiple code blocks in different languages
- **THEN** the Rust router SHALL extract each code block with its language label
- **AND** the extraction SHALL use tree-sitter syntax highlighting

### Requirement: Content type classification uses ML-based detection

The system SHALL use a lightweight ML model (Magika or equivalent) to classify content types. The model SHALL run in Rust to avoid GIL contention and SHALL be pre-loaded at startup to avoid cold-start latency.

#### Scenario: ML-based content classification

- **WHEN** content arrives that is not clearly code or markdown
- **THEN** the Rust router SHALL run ML classification to determine if it's prose, config, data, etc.
- **AND** classification SHALL complete within 10ms per block

### Requirement: Routing decisions are cached for repeated content

The system SHALL cache routing decisions (language, content type, structure) keyed by content hash. Repeated content SHALL retrieve the cached decision without re-parsing.

#### Scenario: Cache hit on repeated content

- **WHEN** content "print('hello')" was previously routed as Python with result {language: "python", type: "code"}
- **AND** same content appears in a new request
- **THEN** the cached routing decision SHALL be returned
- **AND** no tree-sitter parsing SHALL occur

### Requirement: Content router integrates with async pipeline

The system SHALL expose an async interface to the content router that returns futures. The Python async event loop SHALL be able to await routing results without blocking. The interface SHALL be:

```
async def route_content(content: bytes) -> RoutingResult
```

#### Scenario: Async routing result

- **WHEN** Python code calls route_content(content) on a 100KB file
- **THEN** the function SHALL return a Future that resolves with RoutingResult
- **AND** the Python event loop SHALL NOT block while waiting