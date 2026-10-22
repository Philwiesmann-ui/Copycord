# =============================================================================
#  Copycord
#  Copyright (C) 2026 github.com/Copycord
#
#  This source code is released under the GNU Affero General Public License
#  version 3.0. A copy of the license is available at:
#  https://www.gnu.org/licenses/agpl-3.0.en.html
# =============================================================================

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Optional, Dict
from dotenv import load_dotenv

from fastapi import FastAPI, Request, status
from fastapi.responses import PlainTextResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from admin.logging_setup import LOGGER

BASE_DIR = Path(__file__).resolve().parent        
PROJECT_ROOT = BASE_DIR.parent  
load_dotenv(PROJECT_ROOT / ".env")

ADMIN_COOKIE_NAME = "cc_admin"
SESSION_MAX_AGE = 60 * 60 * 12

# Paths that stay reachable without a session: the login/registration pages
# and the OAuth round-trip, static assets and the health probe.
PUBLIC_PATHS = (
    "/login",
    "/register",
    "/logout",
    "/auth/",
    "/health",
    "/static",
)


ADMIN_PASSWORD: str = ""
_session_signer: URLSafeTimedSerializer | None = None
_account_checker = None


def client_ip(request: Request) -> str:
    """
    Best-effort client IP for logging, aware of reverse proxies / Cloudflare.
    """
    try:
        hdrs = request.headers

        cf_ip = hdrs.get("cf-connecting-ip")
        if cf_ip:
            return cf_ip.strip()

        real_ip = hdrs.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        xff = hdrs.get("x-forwarded-for")
        if xff:
            first = xff.split(",")[0].strip()
            if first:
                return first

        if request.client:
            return request.client.host

    except Exception:
        pass

    return "unknown"


def _load_or_create_secret_key(
    data_dir: Path,
    secret_env: str,
    filename: str = "secret.key",
) -> str:
    """
    Load a stable secret key for signing admin session cookies.

    Priority:
      1) SECRET_KEY env (advanced users)
      2) /data/admin_secret.key (auto-persisted)
      3) freshly generated (non-persisted if write fails)
    """
    if secret_env:
        return secret_env

    secret_file = data_dir / filename

    if secret_file.exists():
        try:
            key = secret_file.read_text(encoding="utf-8").strip()
            if key:
                return key
        except Exception:
            LOGGER.warning(
                "Failed reading %s; regenerating", secret_file, exc_info=True
            )

    key = secrets.token_urlsafe(32)
    try:
        secret_file.write_text(key, encoding="utf-8")
    except Exception:

        LOGGER.warning(
            "Failed writing %s; key will rotate on restart", secret_file, exc_info=True
        )
    return key


def _get_signer() -> URLSafeTimedSerializer:
    if _session_signer is None:
        raise RuntimeError("admin auth not initialized: call init_admin_auth() first")
    return _session_signer


def create_session_token(payload: Dict) -> str:
    """Sign an arbitrary session payload for the admin cookie."""
    return _get_signer().dumps(payload)


def set_session_cookie(response, request: Request, token: str) -> None:
    """Attach a signed session token to a response."""
    response.set_cookie(
        ADMIN_COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_MAX_AGE,
        secure=request.url.scheme == "https",
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie(ADMIN_COOKIE_NAME)


def set_account_checker(checker) -> None:
    """
    Registered by admin.web_auth: reports whether any dashboard account
    exists, so accounts alone can require a login (without PASSWORD set).
    """
    global _account_checker
    _account_checker = checker


def login_gate_active() -> bool:
    """
    Is a session required at all?

    Active when a shared admin password is configured (the classic setup) or
    when at least one dashboard account exists.
    """
    if ADMIN_PASSWORD:
        return True

    checker = _account_checker
    if checker is None:
        return False

    try:
        return bool(checker())
    except Exception:
        LOGGER.warning("Account check failed", exc_info=True)
        return False


def decode_admin_session(token: str | None) -> Optional[Dict]:
    """
    Return session payload if token is valid & not expired, else None.
    """
    if not token:
        return None
    try:
        return _get_signer().loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


class PasswordGuardMiddleware(BaseHTTPMiddleware):
    """
    Session gate for the admin dashboard.

    - When neither a shared PASSWORD nor any dashboard account exists, it
      does nothing (login disabled, as documented in the README).
    - Otherwise:
        * the login/registration/oauth pages, /health and static assets are
          always allowed
        * every other request must have a valid signed session cookie,
          otherwise you get redirected to /login (for GET/HEAD)
          or a 401 for other methods.
    """

    async def dispatch(self, request: Request, call_next):

        if not login_gate_active():
            return await call_next(request)

        path = request.url.path or "/"

        if path.startswith(PUBLIC_PATHS):
            return await call_next(request)

        token = request.cookies.get(ADMIN_COOKIE_NAME)
        sess = decode_admin_session(token)
        if sess and sess.get("ok") is True:
            return await call_next(request)

        if request.method in ("GET", "HEAD"):
            return RedirectResponse(url="/login")
        return PlainTextResponse(
            "Unauthorized",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


def init_admin_auth(app: FastAPI, data_dir: Path) -> None:
    """
    Initialize admin sessions:

    - Reads PASSWORD & SECRET_KEY from env
    - Sets up the session signer
    - Adds PasswordGuardMiddleware

    The login, registration and OAuth routes are registered separately by
    admin.web_auth.init_web_auth().
    """
    global ADMIN_PASSWORD, _session_signer

    ADMIN_PASSWORD = os.getenv("PASSWORD", "").strip()
    secret_env = os.getenv("SECRET_KEY", "").strip()

    secret_key = _load_or_create_secret_key(data_dir, secret_env)
    _session_signer = URLSafeTimedSerializer(secret_key, salt="copycord-admin")

    app.add_middleware(PasswordGuardMiddleware)
