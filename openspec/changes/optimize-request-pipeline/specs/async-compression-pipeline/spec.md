## ADDED Requirements

### Requirement: Pipeline executes transforms concurrently via Rust workers

The system SHALL execute independent transforms concurrently using a pool of async Rust workers. Each worker SHALL release the Python GIL during computation to allow other tasks to proceed. Workers SHALL communicate with the Python async event loop via channels that send serialized messages.

#### Scenario: Concurrent transform execution

- **WHEN** a request enters the pipeline with transforms A, B, and C where B and C have no dependency on A
- **THEN** transforms B and C SHALL be scheduled to execute in parallel while A executes
- **AND** all transforms SHALL complete before the response is sent

#### Scenario: GIL release during Rust computation

- **WHEN** a Rust worker is computing (e.g., simhash, content routing)
- **THEN** the Python GIL SHALL be released so the async event loop can process other requests
- **AND** no blocking Python code SHALL run during GIL release

### Requirement: Worker pool manages transform lifecycle

The system SHALL maintain a pool of Rust workers that are pre-initialized and reused across requests. Workers SHALL be created at startup with a configurable pool size based on CPU cores. Failed workers SHALL be automatically restarted without disrupting the proxy.

#### Scenario: Worker reuse

- **WHEN** a request completes and releases a worker back to the pool
- **THEN** that worker SHALL be immediately available for the next request
- **AND** no worker re-initialization SHALL occur

#### Scenario: Worker failure recovery

- **WHEN** a worker panics or crashes during computation
- **THEN** the proxy SHALL log the error and restart a replacement worker
- **AND** the failed request SHALL return an error to the client
- **AND** other requests SHALL continue processing

### Requirement: Transform results are serialized back to Python

The system SHALL serialize Rust computation results (fingerprints, token counts, routing decisions) back to Python memory using efficient binary formats. Serialization SHALL complete before the async task resumes in Python.

#### Scenario: Result serialization

- **WHEN** a Rust worker completes a transform computation
- **THEN** results SHALL be serialized to a buffer accessible by Python
- **AND** the Python async task SHALL receive results within 1ms of computation completion

### Requirement: Pipeline gracefully handles worker overload

The system SHALL queue transform requests when all workers are busy. Queue size SHALL be configurable and requests SHALL fail fast if the queue exceeds limits. Backpressure SHALL be communicated to upstream callers.

#### Scenario: Queue overflow handling

- **WHEN** all workers are busy and the queue exceeds max size
- **THEN** new transform requests SHALL be rejected immediately
- **AND** the proxy SHALL return a 503 Service Unavailable response