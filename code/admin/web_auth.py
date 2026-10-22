# =============================================================================
#  Copycord
#  Copyright (C) 2026 github.com/Copycord
#
#  This source code is released under the GNU Affero General Public License
#  version 3.0. A copy of the license is available at:
#  https://www.gnu.org/licenses/agpl-3.0.en.html
# =============================================================================

"""
Dashboard login / registration.

Besides the classic single shared password (PASSWORD env, see admin.auth),
the dashboard can hold real user accounts:

  * e-mail + password registration and login
  * "Continue with Discord" / "Continue with Google" (OAuth 2.0 code flow)

OAuth app credentials are stored in the app_config table and are edited in
the "Login & Registration" card on the dashboard. Every key can also be
provided through the matching environment variable.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
from typing import Any, Dict, Optional
from urllib.parse import quote, urlencode

import aiohttp
from fastapi import FastAPI, Form, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from admin import auth as admin_auth
from admin.logging_setup import LOGGER
from common.config import CURRENT_VERSION

# ── config keys (app_config table, same names as the env fallbacks) ─────────
CFG_ALLOW_REGISTRATION = "WEB_AUTH_ALLOW_REGISTRATION"
CFG_PUBLIC_BASE_URL = "WEB_AUTH_PUBLIC_BASE_URL"
CFG_DISCORD_CLIENT_ID = "DISCORD_CLIENT_ID"
CFG_DISCORD_CLIENT_SECRET = "DISCORD_CLIENT_SECRET"
CFG_GOOGLE_CLIENT_ID = "GOOGLE_CLIENT_ID"
CFG_GOOGLE_CLIENT_SECRET = "GOOGLE_CLIENT_SECRET"

CONFIG_KEYS = [
    CFG_ALLOW_REGISTRATION,
    CFG_PUBLIC_BASE_URL,
    CFG_DISCORD_CLIENT_ID,
    CFG_DISCORD_CLIENT_SECRET,
    CFG_GOOGLE_CLIENT_ID,
    CFG_GOOGLE_CLIENT_SECRET,
]

OAUTH_STATE_COOKIE = "cc_oauth_state"
OAUTH_STATE_MAX_AGE = 600

PBKDF2_ITERATIONS = 200_000
MIN_PASSWORD_LENGTH = 8

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

PROVIDERS: Dict[str, Dict[str, str]] = {
    "discord": {
        "label": "Discord",
        "authorize_url": "https://discord.com/oauth2/authorize",
        "token_url": "https://discord.com/api/oauth2/token",
        "profile_url": "https://discord.com/api/users/@me",
        "scope": "identify email",
        "client_id_key": CFG_DISCORD_CLIENT_ID,
        "client_secret_key": CFG_DISCORD_CLIENT_SECRET,
        "setup_hint": "https://discord.com/developers/applications",
    },
    "google": {
        "label": "Google",
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "profile_url": "https://openidconnect.googleapis.com/v1/userinfo",
        "scope": "openid email profile",
        "client_id_key": CFG_GOOGLE_CLIENT_ID,
        "client_secret_key": CFG_GOOGLE_CLIENT_SECRET,
        "setup_hint": "https://console.cloud.google.com/apis/credentials",
    },
}

_db = None
_templates: Jinja2Templates | None = None


class OAuthError(Exception):
    """Raised when a provider rejects the sign-in."""


# =============================================================================
#  Password hashing
# =============================================================================
def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt, digest = (stored or "").split("$")
        if algo != "pbkdf2_sha256":
            return False
        check = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), int(iterations)
        )
        return hmac.compare_digest(check.hex(), digest)
    except Exception:
        return False


# =============================================================================
#  Settings
# =============================================================================
def _get(key: str, default: str = "") -> str:
    """Config value from the database, falling back to the environment."""
    value = ""
    try:
        value = (_db.get_config(key, "") or "").strip()
    except Exception:
        value = ""
    if value:
        return value
    return (os.getenv(key, "") or "").strip() or default


def _as_bool(value: str, default: bool = False) -> bool:
    value = (value or "").strip().lower()
    if not value:
        return default
    return value not in ("0", "false", "no", "off")


def public_settings() -> Dict[str, Any]:
    """Everything the login/register pages and the settings card need."""
    settings = {
        "allow_registration": _as_bool(_get(CFG_ALLOW_REGISTRATION), default=True),
        "public_base_url": _get(CFG_PUBLIC_BASE_URL).rstrip("/"),
        "has_users": bool(_db.count_web_users()) if _db else False,
    }
    for key, spec in PROVIDERS.items():
        client_id = _get(spec["client_id_key"])
        client_secret = _get(spec["client_secret_key"])
        settings[f"{key}_client_id"] = client_id
        settings[f"{key}_client_secret"] = client_secret
        settings[f"{key}_configured"] = bool(client_id and client_secret)
    return settings


def _base_url(request: Request, settings: Dict[str, Any]) -> str:
    """Public base URL — configured value first, request host as fallback."""
    base = (settings.get("public_base_url") or "").rstrip("/")
    if base:
        return base
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if host:
        return f"{proto}://{host}"
    return str(request.base_url).rstrip("/")


def _redirect_uri(request: Request, settings: Dict[str, Any], provider: str) -> str:
    return f"{_base_url(request, settings)}/auth/{provider}/callback"


def _provider_list(request: Request, settings: Dict[str, Any]) -> list[dict]:
    return [
        {
            "key": key,
            "label": spec["label"],
            "configured": bool(settings.get(f"{key}_configured")),
            "setup_hint": spec["setup_hint"],
            "redirect_uri": _redirect_uri(request, settings, key),
        }
        for key, spec in PROVIDERS.items()
    ]


# =============================================================================
#  Accounts
# =============================================================================
def _user_payload(user) -> Dict[str, Any]:
    return {
        "ok": True,
        "uid": int(user["user_id"]),
        "name": user["username"] or user["email"] or "user",
        "email": user["email"] or "",
        "provider": user["provider"] or "local",
    }


def login_with_password(email: str, password: str) -> Dict[str, Any]:
    """
    Check e-mail + password against a stored account.

    Returns {"user": <payload>} on success and {"error": <message>} otherwise.
    """
    email = (email or "").strip()
    if not email:
        return {"error": ""}
    if not EMAIL_RE.match(email):
        return {"error": "Please enter a valid e-mail address."}

    user = _db.get_web_user_by_email(email)
    provider_label = None
    if user and user["provider"] in PROVIDERS:
        provider_label = PROVIDERS[user["provider"]]["label"]

    if not user or not user["password_hash"]:
        if provider_label:
            return {
                "error": f"This account signs in with {provider_label}. Use the "
                f"{provider_label} button below."
            }
        return {"error": "Incorrect e-mail or password. Please try again."}

    if not verify_password(password, user["password_hash"]):
        return {"error": "Incorrect e-mail or password. Please try again."}

    _db.touch_web_user_login(user["user_id"])
    return {"user": _user_payload(user)}


def _create_account(
    email: str,
    username: str,
    password_hash: str = "",
    provider: str = "local",
    provider_id: str = "",
    avatar_url: str = "",
) -> Dict[str, Any]:
    user_id = _db.create_web_user(
        email=email,
        username=username,
        password_hash=password_hash,
        provider=provider,
        provider_id=provider_id,
        avatar_url=avatar_url,
    )
    if user_id is None:
        return {"error": "An account with this e-mail address already exists."}
    return {"user": _user_payload(_db.get_web_user(user_id))}


def _username_from_email(email: str) -> str:
    return (email.split("@")[0] or "user").strip()[:32] or "user"


# =============================================================================
#  OAuth
# =============================================================================
def _authorize_url(
    provider: str, settings: Dict[str, Any], redirect_uri: str, state: str
) -> str:
    spec = PROVIDERS[provider]
    query = urlencode(
        {
            "client_id": settings[f"{provider}_client_id"],
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": spec["scope"],
            "state": state,
            **({"prompt": "select_account"} if provider == "google" else {}),
        }
    )
    return f"{spec['authorize_url']}?{query}"


async def _exchange_code(
    provider: str, settings: Dict[str, Any], code: str, redirect_uri: str
) -> str:
    spec = PROVIDERS[provider]
    data = {
        "client_id": settings[f"{provider}_client_id"],
        "client_secret": settings[f"{provider}_client_secret"],
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            spec["token_url"], data=data, headers={"Accept": "application/json"}
        ) as resp:
            payload = await resp.json(content_type=None)
            if resp.status != 200 or not isinstance(payload, dict):
                detail = ""
                if isinstance(payload, dict):
                    detail = payload.get("error_description") or payload.get("error") or ""
                raise OAuthError(detail or f"token request failed ({resp.status})")
            token = payload.get("access_token")
            if not token:
                raise OAuthError("no access token in the provider response")
            return str(token)


async def _fetch_identity(provider: str, access_token: str) -> Dict[str, Any]:
    spec = PROVIDERS[provider]
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(
            spec["profile_url"],
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        ) as resp:
            profile = await resp.json(content_type=None)
            if resp.status != 200 or not isinstance(profile, dict):
                raise OAuthError(f"profile request failed ({resp.status})")

    email = (profile.get("email") or "").strip().lower()
    verified = profile.get("verified")
    if verified is None:
        verified = profile.get("email_verified")
    if email and verified is False:
        raise OAuthError("your provider account has an unverified e-mail address")

    if provider == "discord":
        provider_id = str(profile.get("id") or "")
        username = (
            profile.get("global_name") or profile.get("username") or ""
        ).strip()
        avatar = ""
        if profile.get("avatar"):
            avatar = (
                f"https://cdn.discordapp.com/avatars/{provider_id}/{profile['avatar']}.png?size=64"
                if provider_id
                else ""
            )
    else:
        provider_id = str(profile.get("sub") or "")
        username = (profile.get("name") or "").strip()
        avatar = (profile.get("picture") or "").strip()

    if not provider_id:
        raise OAuthError("the provider did not return an account id")

    return {
        "provider_id": provider_id,
        "email": email,
        "username": username[:32] or _username_from_email(email),
        "avatar_url": avatar,
    }


def _upsert_oauth_user(provider: str, identity: Dict[str, Any]) -> Dict[str, Any]:
    user = _db.get_web_user_by_provider(provider, identity["provider_id"])
    if user is None and identity["email"]:
        # Same e-mail already registered with a password? Link the accounts.
        existing = _db.get_web_user_by_email(identity["email"])
        if existing is not None:
            _db.link_web_user_provider(
                existing["user_id"],
                provider,
                identity["provider_id"],
                identity["avatar_url"],
            )
            user = _db.get_web_user(existing["user_id"])

    if user is None:
        created = _create_account(
            email=identity["email"],
            username=identity["username"],
            provider=provider,
            provider_id=identity["provider_id"],
            avatar_url=identity["avatar_url"],
        )
        if created.get("error"):
            raise OAuthError(created["error"])
        user = _db.get_web_user_by_provider(provider, identity["provider_id"])

    _db.touch_web_user_login(user["user_id"])
    return _user_payload(user)


# =============================================================================
#  Routes
# =============================================================================
def init_web_auth(app: FastAPI, templates: Jinja2Templates, db) -> None:
    global _db, _templates
    _db = db
    _templates = templates

    admin_auth.set_account_checker(lambda: db.count_web_users() > 0)

    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):

        if not admin_auth.login_gate_active():
            return RedirectResponse("/", status_code=status.HTTP_302_FOUND)

        session = admin_auth.decode_admin_session(
            request.cookies.get(admin_auth.ADMIN_COOKIE_NAME)
        )
        if session and session.get("ok") is True:
            return RedirectResponse("/", status_code=status.HTTP_302_FOUND)

        settings = public_settings()
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "version": CURRENT_VERSION,
                "error": request.query_params.get("error") or None,
                "providers": _provider_list(request, settings),
                "allow_registration": settings["allow_registration"],
                "has_users": settings["has_users"],
            },
        )

    @app.post("/login", response_class=HTMLResponse)
    async def login_submit(
        request: Request,
        email: str = Form(""),
        password: str = Form(...),
    ):

        if not admin_auth.login_gate_active():
            return RedirectResponse("/", status_code=status.HTTP_302_FOUND)

        result = login_with_password(email, password)
        if result.get("user"):
            LOGGER.info(
                "Dashboard login SUCCESS for %s from %s",
                result["user"].get("email") or "(account)",
                admin_auth.client_ip(request),
            )
            response = RedirectResponse("/", status_code=status.HTTP_302_FOUND)
            admin_auth.set_session_cookie(
                response, request, admin_auth.create_session_token(result["user"])
            )
            return response

        if admin_auth.ADMIN_PASSWORD and password == admin_auth.ADMIN_PASSWORD:
            LOGGER.info(
                "Dashboard login SUCCESS (admin password) from %s",
                admin_auth.client_ip(request),
            )
            response = RedirectResponse("/", status_code=status.HTTP_302_FOUND)
            admin_auth.set_session_cookie(
                response,
                request,
                admin_auth.create_session_token({"ok": True, "name": "admin"}),
            )
            return response

        if not email.strip() and not admin_auth.ADMIN_PASSWORD:
            error = "Please enter your e-mail address and password."
        else:
            error = result.get("error") or "Incorrect e-mail or password. Please try again."

        LOGGER.warning("Dashboard login FAILED from %s", admin_auth.client_ip(request))

        settings = public_settings()
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "version": CURRENT_VERSION,
                "error": error,
                "providers": _provider_list(request, settings),
                "allow_registration": settings["allow_registration"],
                "has_users": settings["has_users"],
                "email": email,
            },
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    def _render_register(request: Request, error: Optional[str] = None, status_code: int = 200, **values):
        settings = public_settings()
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "version": CURRENT_VERSION,
                "error": error,
                "providers": _provider_list(request, settings),
                "allow_registration": settings["allow_registration"],
                **values,
            },
            status_code=status_code,
        )

    @app.get("/register", response_class=HTMLResponse)
    async def register_page(request: Request):

        settings = public_settings()
        if not settings["allow_registration"] and not admin_auth.login_gate_active():
            return RedirectResponse("/", status_code=status.HTTP_302_FOUND)

        if not admin_auth.login_gate_active():
            return RedirectResponse("/", status_code=status.HTTP_302_FOUND)

        session = admin_auth.decode_admin_session(
            request.cookies.get(admin_auth.ADMIN_COOKIE_NAME)
        )
        if session and session.get("ok") is True:
            return RedirectResponse("/", status_code=status.HTTP_302_FOUND)

        if not settings["allow_registration"]:
            return RedirectResponse(
                "/login?error=" + quote("Registration is disabled on this dashboard."),
                status_code=status.HTTP_302_FOUND,
            )

        return _render_register(request)

    @app.post("/register", response_class=HTMLResponse)
    async def register_submit(
        request: Request,
        email: str = Form(""),
        username: str = Form(""),
        password: str = Form(""),
        password_confirm: str = Form(""),
    ):

        if not admin_auth.login_gate_active():
            return RedirectResponse("/", status_code=status.HTTP_302_FOUND)

        settings = public_settings()
        if not settings["allow_registration"]:
            return RedirectResponse(
                "/login?error=" + quote("Registration is disabled on this dashboard."),
                status_code=status.HTTP_302_FOUND,
            )

        email = (email or "").strip()
        username = (username or "").strip()
        values = {"email": email, "username": username}

        error = None
        if not EMAIL_RE.match(email):
            error = "Please enter a valid e-mail address."
        elif len(password) < MIN_PASSWORD_LENGTH:
            error = f"Your password needs at least {MIN_PASSWORD_LENGTH} characters."
        elif password != password_confirm:
            error = "The two passwords do not match."
        elif _db.get_web_user_by_email(email) is not None:
            error = "An account with this e-mail address already exists."

        if error:
            LOGGER.warning(
                "Dashboard registration rejected from %s: %s",
                admin_auth.client_ip(request),
                error,
            )
            return _render_register(
                request, error=error, status_code=status.HTTP_400_BAD_REQUEST, **values
            )

        created = _create_account(
            email=email,
            username=username or _username_from_email(email),
            password_hash=hash_password(password),
        )
        if created.get("error"):
            return _render_register(
                request, error=created["error"], status_code=status.HTTP_400_BAD_REQUEST, **values
            )

        LOGGER.info(
            "Dashboard account created for %s from %s", email, admin_auth.client_ip(request)
        )
        response = RedirectResponse("/", status_code=status.HTTP_302_FOUND)
        admin_auth.set_session_cookie(
            response, request, admin_auth.create_session_token(created["user"])
        )
        return response

    @app.get("/logout")
    async def logout(request: Request):
        response = RedirectResponse(
            "/login" if admin_auth.login_gate_active() else "/",
            status_code=status.HTTP_302_FOUND,
        )
        session = admin_auth.decode_admin_session(
            request.cookies.get(admin_auth.ADMIN_COOKIE_NAME)
        )
        if session and session.get("email"):
            LOGGER.info("Dashboard logout for %s", session.get("email"))
        admin_auth.clear_session_cookie(response)
        return response

    @app.get("/auth/{provider}")
    async def oauth_start(provider: str, request: Request):

        spec = PROVIDERS.get(provider)
        if spec is None:
            return RedirectResponse(
                "/login?error=" + quote("Unknown sign-in provider."),
                status_code=status.HTTP_302_FOUND,
            )

        settings = public_settings()
        if not settings[f"{provider}_configured"]:
            return RedirectResponse(
                "/login?error="
                + quote(
                    f"{spec['label']} sign-in is not configured yet. Add the "
                    f"{spec['label']} client ID and secret in the dashboard."
                ),
                status_code=status.HTTP_302_FOUND,
            )

        state = secrets.token_urlsafe(24)
        response = RedirectResponse(
            _authorize_url(provider, settings, _redirect_uri(request, settings, provider), state)
        )
        response.set_cookie(
            OAUTH_STATE_COOKIE,
            admin_auth.create_session_token({"state": state, "provider": provider}),
            httponly=True,
            samesite="lax",
            max_age=OAUTH_STATE_MAX_AGE,
            secure=request.url.scheme == "https",
        )
        return response

    @app.get("/auth/{provider}/callback")
    async def oauth_callback(
        provider: str,
        request: Request,
        code: str = "",
        state: str = "",
        error: str = "",
        error_description: str = "",
    ):

        spec = PROVIDERS.get(provider)
        if spec is None:
            return RedirectResponse(
                "/login?error=" + quote("Unknown sign-in provider."),
                status_code=status.HTTP_302_FOUND,
            )

        def fail(message: str) -> RedirectResponse:
            LOGGER.warning("OAuth sign-in with %s failed: %s", provider, message)
            return RedirectResponse(
                "/login?error=" + quote(message), status_code=status.HTTP_302_FOUND
            )

        if error:
            return fail(error_description or "The sign-in was cancelled.")

        pending = admin_auth.decode_admin_session(request.cookies.get(OAUTH_STATE_COOKIE))
        if not pending or pending.get("provider") != provider or pending.get("state") != state:
            return fail("The sign-in request expired. Please try again.")
        if not code:
            return fail("The provider did not return an authorization code.")

        settings = public_settings()
        if not settings[f"{provider}_configured"]:
            return fail(f"{spec['label']} sign-in is not configured yet.")

        try:
            access_token = await _exchange_code(
                provider, settings, code, _redirect_uri(request, settings, provider)
            )
            identity = await _fetch_identity(provider, access_token)
            payload = _upsert_oauth_user(provider, identity)
        except OAuthError as e:
            return fail(str(e))
        except aiohttp.ClientError as e:
            return fail(f"Could not reach {spec['label']}: {e}")
        except Exception:
            LOGGER.exception("Unexpected error during %s sign-in", provider)
            return fail("Unexpected error during sign-in. Please try again.")

        LOGGER.info(
            "Dashboard login SUCCESS via %s for %s from %s",
            provider,
            payload.get("email") or "(account)",
            admin_auth.client_ip(request),
        )
        response = RedirectResponse("/", status_code=status.HTTP_302_FOUND)
        admin_auth.set_session_cookie(
            response, request, admin_auth.create_session_token(payload)
        )
        response.delete_cookie(OAUTH_STATE_COOKIE)
        return response

    # ── settings API (behind the login gate) ────────────────────────────
    @app.get("/api/auth/settings", response_class=JSONResponse)
    async def api_auth_settings_get(request: Request):
        settings = public_settings()
        return JSONResponse(
            {
                "ok": True,
                "settings": settings,
                "providers": _provider_list(request, settings),
                "user_count": _db.count_web_users(),
                "password_login": bool(admin_auth.ADMIN_PASSWORD),
            }
        )

    @app.put("/api/auth/settings", response_class=JSONResponse)
    async def api_auth_settings_put(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(
                {"ok": False, "error": "Invalid JSON"}, status_code=400
            )

        for key in CONFIG_KEYS:
            if key not in payload:
                continue
            value = payload[key]
            if key == CFG_ALLOW_REGISTRATION:
                value = "true" if _as_bool(str(value), default=True) else "false"
            elif key == CFG_PUBLIC_BASE_URL:
                value = str(value or "").strip().rstrip("/")
                if value and not re.match(r"^https?://[^\s/]+", value):
                    return JSONResponse(
                        {
                            "ok": False,
                            "error": "The public URL must start with http:// or https://",
                        },
                        status_code=400,
                    )
            else:
                value = str(value or "").strip()
            _db.set_config(key, value)

        settings = public_settings()
        return JSONResponse(
            {
                "ok": True,
                "settings": settings,
                "providers": _provider_list(request, settings),
            }
        )
