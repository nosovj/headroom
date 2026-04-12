#!/usr/bin/env python3
"""
Claude Code Compression Model Benchmark

Tests multiple compression models on realistic Claude Code tool outputs:
1. Kompress (ModernBERT-base) - current headroom default
2. LLMLingua-2 (XLM-RoBERTa-large) - Microsoft 2024, explicit rate control
3. SmartCrusher - JSON array compression
4. Truncation - baseline (keep first N items)

Metrics:
- Compression ratio (lower = more compression)
- Token savings percentage
- Latency (ms)
- Quality preservation (semantic similarity if possible)

Usage:
    python benchmark_compression_models.py
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add headroom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import statistics


@dataclass
class CompressionResult:
    """Result of compressing a single sample."""
    model_name: str
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float  # compressed / original (lower = better)
    token_savings_pct: float  # 100 * (original - compressed) / original
    latency_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results for a model."""
    model_name: str
    avg_compression_ratio: float
    avg_token_savings_pct: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    success_rate: float
    min_ratio: float
    max_ratio: float


# =============================================================================
# REALISTIC CLAUDE CODE TRAFFIC SAMPLES
# =============================================================================

def get_claude_code_samples() -> list[dict]:
    """Get 20+ realistic Claude Code tool output samples."""

    samples = []

    # 1. Grep search results - many matches
    grep_results = "\n".join([
        "src/components/Button.tsx:45:export const Button = ({children, onClick, disabled}) => (",
        "src/components/Button.tsx:46:  <button className='btn' onClick={onClick} disabled={disabled}>",
        "src/components/Button.tsx:47:    {children}",
        "src/components/Button.tsx:48:  </button>",
        "src/components/Button.tsx:49:);",
        "src/components/Button.tsx:52:export const PrimaryButton = ({children, ...props}) => (",
        "src/components/Button.tsx:53:  <Button className='btn-primary' {...props}>",
        "src/components/Button.tsx:54:    {children}",
        "src/components/Button.tsx:55:  </Button>",
        "src/components/Button.tsx:56:);",
        "src/components/IconButton.tsx:12:import { Button } from './Button';",
        "src/components/IconButton.tsx:13:export const IconButton = ({icon, ...props}) => (",
        "src/components/IconButton.tsx:14:  <Button className='btn-icon' {...props}>",
        "src/components/IconButton.tsx:15:    <span className='icon'>{icon}</span>",
        "src/components/IconButton.tsx:16:  </Button>",
        "src/components/IconButton.tsx:17:);",
        "src/components/Button.test.tsx:8:describe('Button', () => {",
        "src/components/Button.test.tsx:9:  it('renders children', () => {",
        "src/components/Button.test.tsx:10:    render(<Button>Click me</Button>);",
        "src/components/Button.test.tsx:11:    expect(screen.getByText('Click me')).toBeInTheDocument();",
        "src/components/Button.test.tsx:12:  });",
        "src/components/Button.test.tsx:13:  it('handles click events', async () => {",
        "src/components/Button.test.tsx:14:    const handleClick = jest.fn();",
        "src/components/Button.test.tsx:15:    render(<Button onClick={handleClick}>Click</Button>);",
        "src/components/Button.test.tsx:16:    await userEvent.click(screen.getByRole('button'));",
        "src/components/Button.test.tsx:17:    expect(handleClick).toHaveBeenCalledTimes(1);",
        "src/components/Button.test.tsx:18:  });",
        "src/components/Button.test.tsx:19:});",
        "src/hooks/useButton.ts:5:export function useButton() {",
        "src/hooks/useButton.ts:6:  const [isPressed, setIsPressed] = useState(false);",
        "src/hooks/useButton.ts:7:  const handlePress = () => setIsPressed(true);",
        "src/hooks/useButton.ts:8:  const handleRelease = () => setIsPressed(false);",
        "src/hooks/useButton.ts:9:  return { isPressed, handlePress, handleRelease };",
        "src/hooks/useButton.ts:10:}",
    ])
    samples.append({
        "name": "grep_search_button_component",
        "content": grep_results,
        "type": "search",
        "expected_winner": "search_compressor"
    })

    # 2. JSON array - database query results
    db_results = json.dumps([
        {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "created_at": "2024-01-15T10:30:00Z", "status": "active", "login_count": 142},
        {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "created_at": "2024-01-16T14:22:00Z", "status": "active", "login_count": 89},
        {"id": 3, "name": "Carol Davis", "email": "carol@example.com", "created_at": "2024-01-17T09:15:00Z", "status": "inactive", "login_count": 12},
        {"id": 4, "name": "David Wilson", "email": "david@example.com", "created_at": "2024-01-18T16:45:00Z", "status": "active", "login_count": 234},
        {"id": 5, "name": "Eve Martinez", "email": "eve@example.com", "created_at": "2024-01-19T11:00:00Z", "status": "pending", "login_count": 3},
        {"id": 6, "name": "Frank Brown", "email": "frank@example.com", "created_at": "2024-01-20T08:30:00Z", "status": "active", "login_count": 567},
        {"id": 7, "name": "Grace Lee", "email": "grace@example.com", "created_at": "2024-01-21T13:20:00Z", "status": "active", "login_count": 98},
        {"id": 8, "name": "Henry Taylor", "email": "henry@example.com", "created_at": "2024-01-22T10:00:00Z", "status": "inactive", "login_count": 0},
        {"id": 9, "name": "Ivy Chen", "email": "ivy@example.com", "created_at": "2024-01-23T15:30:00Z", "status": "active", "login_count": 201},
        {"id": 10, "name": "Jack Anderson", "email": "jack@example.com", "created_at": "2024-01-24T09:45:00Z", "status": "active", "login_count": 156},
        {"id": 11, "name": "Karen Thomas", "email": "karen@example.com", "created_at": "2024-01-25T12:15:00Z", "status": "pending", "login_count": 1},
        {"id": 12, "name": "Leo Garcia", "email": "leo@example.com", "created_at": "2024-01-26T14:00:00Z", "status": "active", "login_count": 445},
        {"id": 13, "name": "Mia Rodriguez", "email": "mia@example.com", "created_at": "2024-01-27T11:30:00Z", "status": "active", "login_count": 78},
        {"id": 14, "name": "Noah Martinez", "email": "noah@example.com", "created_at": "2024-01-28T16:20:00Z", "status": "inactive", "login_count": 5},
        {"id": 15, "name": "Olivia Hernandez", "email": "olivia@example.com", "created_at": "2024-01-29T08:00:00Z", "status": "active", "login_count": 312},
    ])
    samples.append({
        "name": "database_query_users",
        "content": db_results,
        "type": "json",
        "expected_winner": "smart_crusher"
    })

    # 3. Log file output
    log_output = """
[2024-01-15 10:30:00] INFO: Application started successfully
[2024-01-15 10:30:01] INFO: Database connection pool initialized (min=5, max=20)
[2024-01-15 10:30:02] INFO: Cache server connected at redis://localhost:6379
[2024-01-15 10:30:03] INFO: Starting HTTP server on port 8080
[2024-01-15 10:30:04] DEBUG: Registered 15 route handlers
[2024-01-15 10:30:05] INFO: Health check endpoint available at /health
[2024-01-15 10:31:00] INFO: New client connection from 192.168.1.100:54321
[2024-01-15 10:31:01] DEBUG: Authenticated user: alice@example.com (session_id: sess_abc123)
[2024-01-15 10:31:02] INFO: GET /api/users - 200 OK (45ms)
[2024-01-15 10:31:15] DEBUG: Cache hit for key: user_profile_1
[2024-01-15 10:31:30] INFO: POST /api/orders - 201 Created (120ms)
[2024-01-15 10:31:45] WARN: Rate limit approaching for IP 192.168.1.101 (950/1000 requests)
[2024-01-15 10:32:00] INFO: GET /api/products - 200 OK (32ms)
[2024-01-15 10:32:15] DEBUG: Database query executed in 15ms: SELECT * FROM products WHERE active=true
[2024-01-15 10:32:30] INFO: WebSocket connection established (client_id: ws_def456)
[2024-01-15 10:32:45] DEBUG: Cache miss for key: product_catalog_v2
[2024-01-15 10:33:00] INFO: PUT /api/users/5 - 200 OK (78ms)
[2024-01-15 10:33:15] WARN: Deprecated endpoint /api/v1/users called by client_version=1.0.3
[2024-01-15 10:33:30] ERROR: Failed to process payment for order #12345: Stripe API timeout
[2024-01-15 10:33:31] ERROR: Retrying payment (attempt 2/3) for order #12345
[2024-01-15 10:33:45] ERROR: Payment retry failed: Card declined
[2024-01-15 10:33:46] ERROR: Order #12345 marked as PAYMENT_FAILED
[2024-01-15 10:34:00] INFO: GET /api/orders/12345/status - 200 OK (25ms)
[2024-01-15 10:34:15] DEBUG: Session cleanup: removed 3 expired sessions
[2024-01-15 10:34:30] INFO: DELETE /api/cart/7 - 204 No Content (18ms)
[2024-01-15 10:35:00] INFO: Scheduled job 'cleanup_expired_tokens' completed (removed 47 tokens)
[2024-01-15 10:35:15] DEBUG: Metrics exported: cpu=23.5%, memory=512MB, active_connections=42
[2024-01-15 10:35:30] INFO: Client disconnected: 192.168.1.100 (session_duration: 4m30s)
[2024-01-15 10:36:00] INFO: New client connection from 192.168.1.102:54322
[2024-01-15 10:36:01] DEBUG: Authenticated user: bob@example.com (session_id: sess_ghi789)
[2024-01-15 10:36:15] INFO: GET /api/dashboard - 200 OK (156ms)
[2024-01-15 10:36:30] DEBUG: Rendered dashboard with 12 widgets in 145ms
[2024-01-15 10:37:00] INFO: POST /api/reports - 202 Accepted (queued)
"""
    samples.append({
        "name": "application_server_logs",
        "content": log_output,
        "type": "log",
        "expected_winner": "log_compressor"
    })

    # 4. Python code file
    python_code = '''"""User authentication and authorization module."""

from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
import hashlib
import secrets

from .database import get_user_by_id, create_user, update_user
from .tokens import create_access_token, verify_token
from .exceptions import AuthenticationError, AuthorizationError


@dataclass
class User:
    """User domain model."""
    id: int
    email: str
    hashed_password: str
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None


class AuthService:
    """Handles user authentication and authorization."""

    def __init__(self, secret_key: str, token_expiry_hours: int = 24):
        self.secret_key = secret_key
        self.token_expiry = timedelta(hours=token_expiry_hours)
        self._password_iterations = 100000  # PBKDF2 iterations

    def register_user(self, email: str, password: str, role: str = "user") -> User:
        """Register a new user account."""
        if not email or "@" not in email:
            raise ValueError("Invalid email address")

        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        hashed = self._hash_password(password)
        user_data = {
            "email": email,
            "hashed_password": hashed,
            "role": role,
            "is_active": True,
            "created_at": datetime.utcnow(),
        }
        user_id = create_user(user_data)
        return User(id=user_id, **user_data)

    def authenticate(self, email: str, password: str) -> tuple[User, str]:
        """Authenticate user and return user with access token."""
        user_data = get_user_by_id(email=email)
        if not user_data:
            raise AuthenticationError("Invalid credentials")

        user = User(**user_data)

        if not user.is_active:
            raise AuthenticationError("Account is disabled")

        if not self._verify_password(password, user.hashed_password):
            raise AuthenticationError("Invalid credentials")

        # Update last login
        update_user(user.id, {"last_login": datetime.utcnow()})

        # Generate token
        token = create_access_token(
            data={"sub": str(user.id), "role": user.role},
            secret_key=self.secret_key,
            expires_delta=self.token_expiry,
        )

        return user, token

    def verify_access(self, token: str, required_role: Optional[str] = None) -> User:
        """Verify access token and optionally check role."""
        payload = verify_token(token, self.secret_key)
        if not payload:
            raise AuthenticationError("Invalid or expired token")

        user_id = int(payload.get("sub"))
        user_data = get_user_by_id(id=user_id)
        if not user_data:
            raise AuthenticationError("User not found")

        user = User(**user_data)

        if required_role and user.role != required_role:
            if user.role != "admin":  # Admins can access everything
                raise AuthorizationError(f"Role '{required_role}' required")

        return user

    def _hash_password(self, password: str) -> str:
        """Hash password using PBKDF2."""
        salt = secrets.token_hex(32)
        hash_obj = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt.encode(),
            self._password_iterations,
        )
        return f"{salt}${hash_obj.hex()}"

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash."""
        salt, hash_hex = stored_hash.split("$")
        hash_obj = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            salt.encode(),
            self._password_iterations,
        )
        return secrets.compare_digest(hash_obj.hex(), hash_hex)
'''
    samples.append({
        "name": "python_auth_service",
        "content": python_code,
        "type": "code",
        "expected_winner": "code_aware"
    })

    # 5. API response - GitHub issues
    github_issues = json.dumps([
        {"id": 1, "number": 142, "title": "Add dark mode support", "state": "open", "labels": ["enhancement", "ui"], "created_at": "2024-01-10T09:00:00Z", "author": "alice", "comments": 5},
        {"id": 2, "number": 143, "title": "Fix login redirect loop on Safari", "state": "open", "labels": ["bug", "browser"], "created_at": "2024-01-11T14:30:00Z", "author": "bob", "comments": 12},
        {"id": 3, "number": 144, "title": "Performance: slow startup in large workspaces", "state": "open", "labels": ["performance", "bug"], "created_at": "2024-01-12T10:15:00Z", "author": "carol", "comments": 8},
        {"id": 4, "number": 145, "title": "Add keyboard shortcuts documentation", "state": "closed", "labels": ["documentation"], "created_at": "2024-01-05T11:00:00Z", "author": "david", "comments": 3},
        {"id": 5, "number": 146, "title": "Export to CSV not working with unicode", "state": "open", "labels": ["bug", "export"], "created_at": "2024-01-13T16:45:00Z", "author": "eve", "comments": 6},
        {"id": 6, "number": 147, "title": "Improve error messages for API failures", "state": "open", "labels": ["enhancement", "api"], "created_at": "2024-01-14T08:20:00Z", "author": "frank", "comments": 2},
        {"id": 7, "number": 148, "title": "Memory leak in WebSocket connection handler", "state": "open", "labels": ["bug", "memory"], "created_at": "2024-01-14T15:00:00Z", "author": "grace", "comments": 15},
        {"id": 8, "number": 149, "title": "Add support for OAuth2 refresh tokens", "state": "open", "labels": ["enhancement", "security"], "created_at": "2024-01-15T09:30:00Z", "author": "henry", "comments": 4},
        {"id": 9, "number": 150, "title": "Fix race condition in cache invalidation", "state": "closed", "labels": ["bug", "cache"], "created_at": "2024-01-08T13:15:00Z", "author": "ivy", "comments": 7},
        {"id": 10, "number": 151, "title": "Add custom themes support", "state": "open", "labels": ["enhancement", "ui"], "created_at": "2024-01-15T11:45:00Z", "author": "jack", "comments": 9},
    ])
    samples.append({
        "name": "github_issues_list",
        "content": github_issues,
        "type": "json",
        "expected_winner": "smart_crusher"
    })

    # 6. File tree output
    file_tree = """/project
├── src/
│   ├── main.py (3.2KB)
│   ├── config.py (1.5KB)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py (2.1KB)
│   │   ├── order.py (1.8KB)
│   │   └── product.py (2.3KB)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth_service.py (4.5KB)
│   │   ├── payment_service.py (3.8KB)
│   │   └── notification_service.py (2.9KB)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── users.py (2.2KB)
│   │   ├── orders.py (2.0KB)
│   │   └── products.py (1.9KB)
│   └── utils/
│       ├── __init__.py
│       ├── validators.py (1.4KB)
│       └── helpers.py (1.6KB)
├── tests/
│   ├── __init__.py
│   ├── test_auth.py (3.2KB)
│   ├── test_payment.py (2.8KB)
│   └── test_orders.py (2.5KB)
├── node_modules/ (45 packages)
├── package.json (2.1KB)
├── requirements.txt (0.8KB)
└── README.md (4.5KB)"""
    samples.append({
        "name": "filesystem_tree",
        "content": file_tree,
        "type": "text",
        "expected_winner": "kompress"
    })

    # 7. Test output - pytest
    pytest_output = """
============================= test session starts ==============================
platform darwin -- Python 3.11.0
collected 45 items

tests/test_auth.py::test_login_success PASSED                             [  2%]
tests/test_auth.py::test_login_invalid_password PASSED                    [  4%]
tests/test_auth.py::test_login_nonexistent_user PASSED                    [  6%]
tests/test_auth.py::test_register_new_user PASSED                         [  8%]
tests/test_auth.py::test_register_duplicate_email PASSED                   [ 9%]
tests/test_auth.py::test_password_hashing PASSED                           [ 11%]
tests/test_auth.py::test_token_expiration PASSED                           [ 13%]
tests/test_auth.py::test_refresh_token PASSED                              [ 15%]
tests/test_auth.py::test_logout PASSED                                     [ 17%]
tests/test_auth.py::test_session_management PASSED                        [ 19%]
tests/test_payment.py::test_process_payment_success PASSED                  [ 22%]
tests/test_payment.py::test_process_payment_card_declined PASSED            [ 24%]
tests/test_payment.py::test_process_payment_insufficient_funds PASSED       [ 26%]
tests/test_payment.py::test_process_payment_timeout PASSED                  [ 28%]
tests/test_payment.py::test_refund_full PASSED                              [ 31%]
tests/test_payment.py::test_refund_partial PASSED                          [ 33%]
tests/test_payment.py::test_refund_already_refunded PASSED                 [ 35%]
tests/test_orders.py::test_create_order PASSED                              [ 37%]
tests/test_orders.py::test_create_order_empty_cart PASSED                   [ 39%]
tests/test_orders.py::test_update_order_status PASSED                       [ 42%]
tests/test_orders.py::test_cancel_order PASSED                              [ 44%]
tests/test_orders.py::test_order_not_found PASSED                           [ 46%]
tests/test_orders.py::test_order_pagination PASSED                         [ 48%]
tests/test_api.py::test_get_user PASSED                                    [ 50%]
tests/test_api.py::test_get_user_not_found PASSED                          [ 52%]
tests/test_api.py::test_update_user PASSED                                 [ 54%]
tests/test_api.py::test_delete_user PASSED                                 [ 56%]
tests/test_api.py::test_list_users_with_pagination PASSED                  [ 58%]
tests/test_api.py::test_search_users PASSED                                [ 60%]

=========================== 42 passed in 12.45s ============================"""
    samples.append({
        "name": "pytest_results_all_passed",
        "content": pytest_output,
        "type": "log",
        "expected_winner": "log_compressor"
    })

    # 8. Mixed content - error in logs
    error_logs = """
[2024-01-15 10:00:00] INFO: Server started on port 8080
[2024-01-15 10:00:01] INFO: Database connected successfully
[2024-01-15 10:00:02] DEBUG: Loaded configuration from config.yaml
[2024-01-15 10:01:00] INFO: New request: GET /api/health
[2024-01-15 10:01:01] INFO: Response: 200 OK (5ms)
[2024-01-15 10:02:00] INFO: New request: POST /api/users
[2024-01-15 10:02:01] DEBUG: Validating user data
[2024-01-15 10:02:02] DEBUG: Checking email uniqueness
[2024-01-15 10:02:03] DEBUG: Hashing password
[2024-01-15 10:02:04] DEBUG: Inserting into database
[2024-01-15 10:02:05] INFO: Response: 201 Created (45ms)
[2024-01-15 10:03:00] ERROR: Connection refused to payment-gateway:8080
[2024-01-15 10:03:01] ERROR: Payment processing failed for order #1234
[2024-01-15 10:03:02] DEBUG: Retrying in 5 seconds...
[2024-01-15 10:03:07] ERROR: Retry failed: timeout after 30000ms
[2024-01-15 10:03:08] ERROR: Order #1234 marked as PAYMENT_FAILED
[2024-01-15 10:04:00] INFO: New request: GET /api/products
[2024-01-15 10:04:01] DEBUG: Query: SELECT * FROM products WHERE active = true
[2024-01-15 10:04:02] DEBUG: Found 156 products
[2024-01-15 10:04:03] DEBUG: Applying pagination (limit=50, offset=0)
[2024-01-15 10:04:04] INFO: Response: 200 OK (85ms)
[2024-01-15 10:05:00] WARN: High memory usage detected: 1.8GB / 2GB
[2024-01-15 10:05:01] WARN: Triggering garbage collection
[2024-01-15 10:05:02] DEBUG: GC freed 256MB
[2024-01-15 10:06:00] INFO: Scheduled task 'cleanup_sessions' started
[2024-01-15 10:06:01] DEBUG: Removed 45 expired sessions
[2024-01-15 10:06:02] INFO: Scheduled task completed in 1.2s
"""
    samples.append({
        "name": "mixed_logs_with_errors",
        "content": error_logs,
        "type": "log",
        "expected_winner": "log_compressor"
    })

    # 9. Large JSON array - search results with scores
    search_results = json.dumps([
        {"id": "doc_1", "score": 0.95, "title": "Getting Started with React", "snippet": "Learn how to build modern web applications with React...", "url": "/docs/react-getting-started"},
        {"id": "doc_2", "score": 0.92, "title": "React Hooks Guide", "snippet": "A comprehensive guide to React hooks including useState, useEffect...", "url": "/docs/react-hooks"},
        {"id": "doc_3", "score": 0.89, "title": "State Management in React", "snippet": "Compare Redux, Context API, and other state management solutions...", "url": "/docs/react-state-management"},
        {"id": "doc_4", "score": 0.87, "title": "React Performance Optimization", "snippet": "Tips and tricks to optimize your React application's performance...", "url": "/docs/react-performance"},
        {"id": "doc_5", "score": 0.85, "title": "Testing React Components", "snippet": "Learn how to write effective tests for your React components...", "url": "/docs/react-testing"},
        {"id": "doc_6", "score": 0.82, "title": "React Server Components", "snippet": "Understanding the new React Server Components architecture...", "url": "/docs/react-server-components"},
        {"id": "doc_7", "score": 0.78, "title": "Next.js with React", "snippet": "Build full-stack applications with Next.js and React...", "url": "/docs/nextjs-react"},
        {"id": "doc_8", "score": 0.75, "title": "React Native Fundamentals", "snippet": "Build mobile applications using React Native framework...", "url": "/docs/react-native"},
        {"id": "doc_9", "score": 0.72, "title": "TypeScript with React", "snippet": "Add type safety to your React applications with TypeScript...", "url": "/docs/typescript-react"},
        {"id": "doc_10", "score": 0.68, "title": "CSS-in-JS in React", "snippet": "Compare styled-components, emotion, and other CSS-in-JS solutions...", "url": "/docs/css-in-js"},
    ])
    samples.append({
        "name": "search_results_with_scores",
        "content": search_results,
        "type": "json",
        "expected_winner": "smart_crusher"
    })

    # 10. Build output - webpack
    build_output = """
> project@1.0.0 build /project
> webpack --mode production

Hash: a1b2c3d4e5f6
Version: webpack 5.89.0
Time: 12345ms
Built at: 2024-01-15 10:30:00

                           Asset       Size  Chunks           Chunk Names
       main.abcd1234.js    1.25 MB    0  [emitted]    main
      vendor.efgh5678.js    890 KB    1  [emitted]    vendor
     runtime.ijkl9012.js    5.2 KB    2  [emitted]    runtime
       main.abcd1234.js.map   2.1 MB    0  [emitted]    main
      vendor.efgh5678.js.map  1.8 MB    1  [emitted]    vendor

WARNING in ./src/components/Button.tsx 12:5
Emitted value was already compressed (modification of prop)
 @ ./src/pages/Checkout.tsx 45:12
 @ ./src/App.tsx 23:1

WARNING in ./src/utils/format.ts 5:12
Unused export function 'formatCurrency'
 @ ./src/App.tsx 28:1

WARNING in ./src/hooks/useAuth.ts
'user' is assigned a value but never used

Built successfully!

Warnings: 3
Errors: 0
"""
    samples.append({
        "name": "webpack_build_output",
        "content": build_output,
        "type": "log",
        "expected_winner": "log_compressor"
    })

    # 11. Read multiple files output
    file_contents = """
=== FILE: package.json ===
{
  "name": "my-project",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  }
}

=== FILE: src/index.tsx ===
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

=== FILE: src/App.tsx ===
import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import MainRoutes from './routes/MainRoutes';

function App() {
  return (
    <BrowserRouter>
      <MainRoutes />
    </BrowserRouter>
  );
}

export default App;
"""
    samples.append({
        "name": "multiple_file_reads",
        "content": file_contents,
        "type": "text",
        "expected_winner": "kompress"
    })

    # 12. Elasticsearch query results
    es_results = {
        "took": 45,
        "timed_out": False,
        "hits": {
            "total": {"value": 1523, "relation": "eq"},
            "max_score": 0.95,
            "hits": [
                {"_index": "products", "_id": "1", "_score": 0.95, "_source": {"id": 1, "name": "Laptop Pro 15", "price": 1299.99, "category": "electronics", "in_stock": True}},
                {"_index": "products", "_id": "2", "_score": 0.92, "_source": {"id": 2, "name": "Wireless Mouse", "price": 29.99, "category": "electronics", "in_stock": True}},
                {"_index": "products", "_id": "3", "_score": 0.88, "_source": {"id": 3, "name": "USB-C Cable", "price": 15.99, "category": "electronics", "in_stock": False}},
            ]
        }
    }
    samples.append({
        "name": "elasticsearch_results",
        "content": json.dumps(es_results),
        "type": "json",
        "expected_winner": "smart_crusher"
    })

    # 13. cURL output
    curl_output = """
HTTP/1.1 200 OK
Server: nginx/1.24.0
Date: Mon, 15 Jan 2024 10:30:00 GMT
Content-Type: application/json
Content-Length: 1234
Connection: keep-alive
Cache-Control: no-cache
X-Request-ID: req_abc123

{
  "id": 12345,
  "status": "success",
  "data": {
    "user_id": 67890,
    "name": "John Doe",
    "email": "john@example.com",
    "created_at": "2024-01-01T00:00:00Z"
  },
  "meta": {
    "request_id": "req_abc123",
    "processing_time_ms": 45
  }
}"""
    samples.append({
        "name": "http_response_headers",
        "content": curl_output,
        "type": "text",
        "expected_winner": "kompress"
    })

    # 14. Kubernetes pod logs
    k8s_logs = """
[pod/nginx-deployment-7fb96c846b-xk9p2] 2024-01-15T10:30:00.000Z INFO Starting nginx...
[pod/nginx-deployment-7fb96c846b-xk9p2] 2024-01-15T10:30:00.123Z INFO Listening on port 80
[pod/nginx-deployment-7fb96c846b-xk9p2] 2024-01-15T10:31:00.456Z INFO 10.244.0.5 - - [15/Jan/2024:10:31:00 +0000] "GET / HTTP/1.1" 200 612 "-" "Mozilla/5.0"
[pod/nginx-deployment-7fb96c846b-xk9p2] 2024-01-15T10:31:01.789Z INFO 10.244.0.5 - - [15/Jan/2024:10:31:01 +0000] "GET /static/css/main.css HTTP/1.1" 200 15420 "http://example.com/" "Mozilla/5.0"
[pod/nginx-deployment-7fb96c846b-xk9p2] 2024-01-15T10:31:02.012Z INFO 10.244.0.5 - - [15/Jan/2024:10:31:02 +0000] "GET /static/js/main.js HTTP/1.1" 200 89234 "http://example.com/" "Mozilla/5.0"
[pod/api-deployment-5f9d8c7b9c-2j8k4] 2024-01-15T10:30:00.000Z INFO API server starting on port 8080
[pod/api-deployment-5f9d8c7b9c-2j8k4] 2024-01-15T10:30:01.234Z INFO Database connection pool: min=5, max=20
[pod/api-deployment-5f9d8c7b9c-2j8k4] 2024-01-15T10:30:02.567Z INFO Cache connected: redis://redis:6379
[pod/api-deployment-5f9d8c7b9c-2j8k4] 2024-01-15T10:31:15.890Z ERROR Failed to process request: connection timeout
[pod/api-deployment-5f9d8c7b9c-2j8k4] 2024-01-15T10:31:16.123Z WARN Retrying request (attempt 2/3)
[pod/api-deployment-5f9d8c7b9c-2j8k4] 2024-01-15T10:31:20.456Z INFO Request succeeded after retry"""
    samples.append({
        "name": "kubernetes_pod_logs",
        "content": k8s_logs,
        "type": "log",
        "expected_winner": "log_compressor"
    })

    # 15. Environment/config file
    env_content = """
# Application Configuration
APP_NAME=MyApplication
APP_ENV=production
APP_DEBUG=false
APP_URL=https://api.example.com

# Database Configuration
DB_HOST=db.example.com
DB_PORT=5432
DB_NAME=myapp_production
DB_USER=app_user
DB_PASSWORD=super_secret_password_123
DB_POOL_MIN=5
DB_POOL_MAX=20

# Redis Configuration
REDIS_HOST=cache.example.com
REDIS_PORT=6379
REDIS_PASSWORD=another_secret_456
REDIS_DB=0

# API Keys
STRIPE_API_KEY=sk_live_abc123xyz
SENDGRID_API_KEY=SG.abcdefghijklmnopqrstuvwxyz
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

# Feature Flags
FEATURE_DARK_MODE=true
FEATURE_NEW_CHECKOUT=false
FEATURE_BETA_API=true
"""
    samples.append({
        "name": "environment_config",
        "content": env_content,
        "type": "text",
        "expected_winner": "kompress"
    })

    # 16. SQL query results
    sql_results = [
        {"id": 1, "department": "Engineering", "employee": "Alice", "salary": 95000, "hire_date": "2020-03-15"},
        {"id": 2, "department": "Engineering", "employee": "Bob", "salary": 87000, "hire_date": "2021-06-01"},
        {"id": 3, "department": "Engineering", "employee": "Carol", "salary": 110000, "hire_date": "2019-01-10"},
        {"id": 4, "department": "Marketing", "employee": "Dave", "salary": 75000, "hire_date": "2022-02-20"},
        {"id": 5, "department": "Marketing", "employee": "Eve", "salary": 82000, "hire_date": "2021-09-05"},
        {"id": 6, "department": "Sales", "employee": "Frank", "salary": 70000, "hire_date": "2023-01-15"},
        {"id": 7, "department": "Sales", "employee": "Grace", "salary": 78000, "hire_date": "2022-08-10"},
        {"id": 8, "department": "HR", "employee": "Henry", "salary": 65000, "hire_date": "2020-11-30"},
        {"id": 9, "department": "Engineering", "employee": "Ivy", "salary": 105000, "hire_date": "2018-07-20"},
        {"id": 10, "department": "Finance", "employee": "Jack", "salary": 90000, "hire_date": "2021-04-25"},
    ]
    samples.append({
        "name": "sql_query_results",
        "content": json.dumps(sql_results),
        "type": "json",
        "expected_winner": "smart_crusher"
    })

    # 17. Docker inspect output
    docker_inspect = json.dumps({
        "Id": "abc123def456",
        "Name": "/nginx-webserver",
        "Created": "2024-01-10T15:30:00.000000000Z",
        "State": {"Status": "running", "Running": True, "Pid": 12345},
        "Config": {
            "Image": "nginx:latest",
            "ExposedPorts": {"80/tcp": {}, "443/tcp": {}},
            "Env": ["NODE_ENV=production", "PORT=80"],
            "Cmd": ["nginx", "-g", "daemon off;"],
        },
        "NetworkSettings": {
            "IPAddress": "172.17.0.10",
            "Ports": {"80/tcp": [{"HostPort": "8080"}]},
        },
        "Mounts": [
            {"Type": "bind", "Source": "/data/nginx/html", "Destination": "/usr/share/nginx/html"}
        ],
    })
    samples.append({
        "name": "docker_inspect",
        "content": docker_inspect,
        "type": "json",
        "expected_winner": "smart_crusher"
    })

    # 18. Git log output
    git_log = """
commit abc123def456789012345678901234567890abcd
Author: Alice Johnson <alice@example.com>
Date:   Mon Jan 15 10:30:00 2024 +0000

    Add user authentication module

    - Implement login/logout functionality
    - Add password hashing with PBKDF2
    - Create session management
    - Add OAuth2 support (Google, GitHub)

commit 987fedcba0987654321fedcba0987654321fedcba
Author: Bob Smith <bob@example.com>
Date:   Sun Jan 14 14:20:00 2024 +0000

    Fix pagination in API endpoints

    - Correct offset calculation
    - Add total_count to response
    - Update unit tests

commit 456abc789def0123456789abcdef0123456789ab
Author: Carol Davis <carol@example.com>
Date:   Sat Jan 13 09:15:00 2024 +0000

    Refactor database connection pooling

    - Add connection health checks
    - Implement automatic reconnection
    - Add metrics for pool utilization
"""
    samples.append({
        "name": "git_log_output",
        "content": git_log,
        "type": "text",
        "expected_winner": "kompress"
    })

    # 19. XML configuration
    xml_config = """<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <appSettings>
        <add key="Environment" value="Production" />
        <add key="DebugMode" value="false" />
        <add key="MaxConnections" value="100" />
        <add key="RequestTimeout" value="30000" />
        <add key="EnableCaching" value="true" />
        <add key="CacheTTL" value="3600" />
    </appSettings>
    <connectionStrings>
        <add name="DefaultConnection" connectionString="Server=db.example.com;Database=MyApp;User=app_user;Password=secret;" />
        <add name="CacheConnection" connectionString="redis://cache.example.com:6379" />
    </connectionStrings>
    <logging>
        <add key="LogLevel" value="Information" />
        <add key="LogToFile" value="true" />
        <add key="LogFilePath" value="/var/log/myapp/app.log" />
        <add key="LogToConsole" value="false" />
    </logging>
</configuration>"""
    samples.append({
        "name": "xml_configuration",
        "content": xml_config,
        "type": "text",
        "expected_winner": "kompress"
    })

    # 20. Network trace / tcpdump output
    network_trace = """
10:30:00.000000 IP 192.168.1.100.54321 > 10.0.0.1.80: Flags [S], seq 1234567890, win 65535
10:30:00.001234 IP 10.0.0.1.80 > 192.168.1.100.54321: Flags [S.], seq 987654321, ack 1234567891, win 65535
10:30:00.002345 IP 192.168.1.100.54321 > 10.0.0.1.80: Flags [P.], ack 987654322, win 65535
10:30:00.003456 IP 10.0.0.1.80 > 192.168.1.100.54321: Flags [P.], ack 1234567891, win 65535
10:30:00.100000 IP 192.168.1.100.54321 > 10.0.0.1.80: Flags [P.], seq 1234567891:1234568900, win 65535
10:30:00.101234 IP 10.0.0.1.80 > 192.168.1.100.54321: Flags [P.], ack 1234568900, win 65535
10:30:00.200000 IP 192.168.1.100.54322 > 10.0.0.2.443: Flags [P.], seq 1:100, win 65535
10:30:00.201234 IP 10.0.0.2.443 > 192.168.1.100.54322: Flags [P.], ack 100, win 65535
10:30:00.300000 IP 192.168.1.100.54321 > 10.0.0.1.80: Flags [F.], seq 1234568900, win 65535
10:30:00.301234 IP 10.0.0.1.80 > 192.168.1.100.54321: Flags [F.], ack 1234568901, win 65535"""
    samples.append({
        "name": "network_trace",
        "content": network_trace,
        "type": "text",
        "expected_winner": "kompress"
    })

    return samples


# =============================================================================
# COMPRESSION MODEL IMPLEMENTATIONS
# =============================================================================

def count_tokens(text: str) -> int:
    """Estimate token count (simple word-based approximation)."""
    return len(text.split())


def compress_truncation(content: str, target_ratio: float = 0.1) -> tuple[str, int, float]:
    """Simple truncation baseline - keep first N words."""
    start = time.perf_counter()
    words = content.split()
    target_words = max(1, int(len(words) * target_ratio))
    compressed = " ".join(words[:target_words])
    latency = (time.perf_counter() - start) * 1000
    return compressed, len(compressed.split()), latency


def compress_kompress(content: str, target_ratio: float = 0.1) -> tuple[str, int, float]:
    """Kompress (ModernBERT-base) compression."""
    start = time.perf_counter()
    try:
        from headroom.transforms.kompress_compressor import KompressCompressor

        compressor = KompressCompressor()
        result = compressor.compress(content, target_ratio=target_ratio)
        latency = (time.perf_counter() - start) * 1000
        return result.compressed, result.compressed_tokens, latency
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        raise RuntimeError(f"Kompress failed: {e}")


def compress_llmlingua2(content: str, target_ratio: float = 0.1) -> tuple[str, int, float]:
    """LLMLingua-2 compression with explicit rate control."""
    start = time.perf_counter()
    try:
        from llmlingua import PromptCompressor

        compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True
        )

        # LLMLingua-2 uses 'rate' parameter directly
        result = compressor.compress_prompt_llmlingua2(
            content,
            rate=target_ratio,
            force_tokens=['\n', '.', '!', '?', ','],
            return_word_label=True
        )

        compressed = result['compressed_prompt']
        compressed_tokens = result['compressed_tokens']
        latency = (time.perf_counter() - start) * 1000

        return compressed, compressed_tokens, latency
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        raise RuntimeError(f"LLMLingua-2 failed: {e}")


def compress_smart_crusher(content: str, target_ratio: float = 0.1) -> tuple[str, int, float]:
    """SmartCrusher for JSON arrays."""
    start = time.perf_counter()
    try:
        from headroom.transforms.smart_crusher import SmartCrusher

        crusher = SmartCrusher()
        result = crusher.crush(content, query="", bias=1.0)
        latency = (time.perf_counter() - start) * 1000
        return result.compressed, len(result.compressed.split()), latency
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        raise RuntimeError(f"SmartCrusher failed: {e}")


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_benchmark(
    samples: list[dict],
    target_ratio: float = 0.1
) -> dict[str, list[CompressionResult]]:
    """Run compression benchmark on all samples."""

    results = {
        "kompress": [],
        "llmlingua2": [],
        "smart_crusher": [],
        "truncation": [],
    }

    print(f"\nRunning benchmark on {len(samples)} samples (target_ratio={target_ratio})...")
    print("=" * 80)

    for i, sample in enumerate(samples):
        name = sample["name"]
        content = sample["content"]
        original_tokens = count_tokens(content)

        print(f"\n[{i+1}/{len(samples)}] {name}")
        print(f"    Original: {original_tokens} tokens, {len(content)} chars")

        # Test each compressor
        for model_name, compress_func in [
            ("kompress", compress_kompress),
            ("llmlingua2", compress_llmlingua2),
            ("smart_crusher", compress_smart_crusher),
            ("truncation", compress_truncation),
        ]:
            try:
                compressed, compressed_tokens, latency = compress_func(content, target_ratio)

                # Handle case where compression returns same content
                if compressed == content:
                    ratio = 1.0
                    savings = 0.0
                else:
                    ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
                    savings = 100 * (original_tokens - compressed_tokens) / original_tokens

                result = CompressionResult(
                    model_name=model_name,
                    original_text=content[:200] + "..." if len(content) > 200 else content,
                    compressed_text=compressed[:200] + "..." if len(compressed) > 200 else compressed,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    compression_ratio=ratio,
                    token_savings_pct=savings,
                    latency_ms=latency,
                    success=True,
                )

                results[model_name].append(result)
                print(f"    {model_name}: {ratio:.2%} ratio, {savings:.1f}% savings, {latency:.1f}ms")

            except Exception as e:
                result = CompressionResult(
                    model_name=model_name,
                    original_text=content,
                    compressed_text="",
                    original_tokens=original_tokens,
                    compressed_tokens=0,
                    compression_ratio=1.0,
                    token_savings_pct=0.0,
                    latency_ms=0.0,
                    success=False,
                    error=str(e),
                )
                results[model_name].append(result)
                print(f"    {model_name}: FAILED - {e}")

    return results


def compute_summary(results: list[CompressionResult]) -> BenchmarkSummary:
    """Compute summary statistics for a model's results."""
    successful = [r for r in results if r.success]
    if not successful:
        return BenchmarkSummary(
            model_name=results[0].model_name if results else "unknown",
            avg_compression_ratio=1.0,
            avg_token_savings_pct=0.0,
            avg_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            success_rate=0.0,
            min_ratio=1.0,
            max_ratio=1.0,
        )

    ratios = [r.compression_ratio for r in successful]
    savings = [r.token_savings_pct for r in successful]
    latencies = [r.latency_ms for r in successful]

    latencies.sort()

    return BenchmarkSummary(
        model_name=successful[0].model_name,
        avg_compression_ratio=statistics.mean(ratios),
        avg_token_savings_pct=statistics.mean(savings),
        avg_latency_ms=statistics.mean(latencies),
        p50_latency_ms=latencies[len(latencies) // 2],
        p95_latency_ms=latencies[int(len(latencies) * 0.95)] if latencies else 0,
        success_rate=100 * len(successful) / len(results),
        min_ratio=min(ratios),
        max_ratio=max(ratios),
    )


def print_results(results: dict[str, list[CompressionResult]]):
    """Print benchmark results summary."""

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    summaries = []
    for model_name, model_results in results.items():
        summary = compute_summary(model_results)
        summaries.append(summary)

        print(f"\n{model_name.upper()}")
        print("-" * 40)
        print(f"  Avg Compression Ratio: {summary.avg_compression_ratio:.2%}")
        print(f"  Avg Token Savings:     {summary.avg_token_savings_pct:.1f}%")
        print(f"  Avg Latency:           {summary.avg_latency_ms:.1f}ms")
        print(f"  P50 Latency:           {summary.p50_latency_ms:.1f}ms")
        print(f"  P95 Latency:           {summary.p95_latency_ms:.1f}ms")
        print(f"  Success Rate:          {summary.success_rate:.0f}%")
        print(f"  Min Ratio:             {summary.min_ratio:.2%}")
        print(f"  Max Ratio:             {summary.max_ratio:.2%}")

    # Determine winner
    print("\n" + "=" * 80)
    print("WINNER ANALYSIS")
    print("=" * 80)

    # Best compression ratio (lower is better)
    best_ratio = min(summaries, key=lambda s: s.avg_compression_ratio)
    print(f"\nBest Compression Ratio: {best_ratio.model_name} ({best_ratio.avg_compression_ratio:.2%})")

    # Best token savings (higher is better)
    best_savings = max(summaries, key=lambda s: s.avg_token_savings_pct)
    print(f"Best Token Savings: {best_savings.model_name} ({best_savings.avg_token_savings_pct:.1f}%)")

    # Fastest (lower is better)
    fastest = min(summaries, key=lambda s: s.avg_latency_ms)
    print(f"Fastest Latency: {fastest.model_name} ({fastest.avg_latency_ms:.1f}ms)")

    # Best success rate
    best_success = max(summaries, key=lambda s: s.success_rate)
    print(f"Best Success Rate: {best_success.model_name} ({best_success.success_rate:.0f}%)")

    # Overall winner (weighted score)
    print("\n" + "-" * 40)
    print("OVERALL RANKING (weighted: 50% savings, 30% speed, 20% reliability)")

    scored = []
    max_savings = max(s.avg_token_savings_pct for s in summaries) or 1.0
    max_speed = max(s.avg_latency_ms for s in summaries) or 1.0

    for s in summaries:
        savings_score = s.avg_token_savings_pct / max_savings
        speed_score = 1.0 - (s.avg_latency_ms / max_speed)
        reliability_score = s.success_rate / 100.0
        overall = 0.5 * savings_score + 0.3 * speed_score + 0.2 * reliability_score
        scored.append((s.model_name, overall, s.avg_token_savings_pct, s.avg_latency_ms, s.success_rate))

    scored.sort(key=lambda x: x[1], reverse=True)

    for i, (name, score, savings, latency, reliability) in enumerate(scored):
        print(f"  {i+1}. {name}: score={score:.3f} (savings={savings:.1f}%, latency={latency:.1f}ms, reliability={reliability:.0f}%)")

    print("\n" + "=" * 80)
    print(f"CLEAR WINNER: {scored[0][0].upper()}")
    print("=" * 80)


def main():
    print("Claude Code Compression Model Benchmark")
    print("=" * 80)

    # Get test samples
    samples = get_claude_code_samples()
    print(f"\nLoaded {len(samples)} realistic Claude Code traffic samples")

    # Run benchmark at different target ratios
    for target_ratio in [0.05, 0.10, 0.20]:
        print(f"\n\n{'#' * 80}")
        print(f"# BENCHMARK AT TARGET RATIO: {target_ratio:.0%}")
        print(f"{'#' * 80}")

        results = run_benchmark(samples, target_ratio)
        print_results(results)

    print("\n\nBenchmark complete!")


if __name__ == "__main__":
    main()