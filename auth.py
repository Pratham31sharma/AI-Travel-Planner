"""Customer accounts for the travel planner.

Sign up / log in with Google, or with an email and a password.
Stored in ./data/users.sqlite (separate from the LangGraph checkpoints):
  users    - name, email, password hash, google id
  sessions - hashed login tokens (valid 30 days)
  plans    - which customer owns which plan_id, its last known status, and a hidden flag
             (a deleted plan is only hidden, so the daily limit cannot be dodged by deleting)
Uses only the Python standard library (scrypt for password hashing).
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
import urllib.parse
import urllib.request
from contextlib import closing

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from app.config import DATABASE_PATH

USERS_DB = os.path.join(os.path.dirname(DATABASE_PATH) or ".", "users.sqlite")
SESSION_SECONDS = 30 * 24 * 3600
MAX_PLANS_PER_DAY = 10          # protects your Groq quota
_failed: dict[str, list[float]] = {}

router = APIRouter()


def _q(sql, args=(), *, one=False, write=False):
    with closing(sqlite3.connect(USERS_DB)) as c:
        c.row_factory = sqlite3.Row
        cur = c.execute(sql, args)
        if write:
            c.commit()
            return cur.lastrowid
        return cur.fetchone() if one else cur.fetchall()


def init_db():
    os.makedirs(os.path.dirname(USERS_DB) or ".", exist_ok=True)
    for sql in (
        """CREATE TABLE IF NOT EXISTS users(
             id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE,
             phone TEXT UNIQUE, pw_hash TEXT NOT NULL, created_at REAL NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS sessions(
             token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires REAL NOT NULL)""",
        """CREATE TABLE IF NOT EXISTS plans(
             plan_id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, destination TEXT,
             start_date TEXT, end_date TEXT, created_at REAL NOT NULL)""",
    ):
        _q(sql, write=True)
    try:                                   # Google login support (safe on existing databases)
        _q("ALTER TABLE users ADD COLUMN google_sub TEXT", write=True)
    except sqlite3.OperationalError:
        pass
    _q("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google ON users(google_sub)", write=True)
    for col in ("status TEXT", "deleted INTEGER NOT NULL DEFAULT 0"):      # plan status + delete (safe on old databases)
        try:
            _q("ALTER TABLE plans ADD COLUMN " + col, write=True)
        except sqlite3.OperationalError:
            pass


# ---------- passwords ----------
def _hash_pw(password: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${h.hex()}"


def _check_pw(password: str, stored: str) -> bool:
    try:
        _, salt, h = stored.split("$")
        calc = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1)
        return hmac.compare_digest(calc.hex(), h)
    except Exception:
        return False


# ---------- email ----------
def _clean_email(raw: str) -> str:
    email = (raw or "").strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) or len(email) > 254:
        raise HTTPException(422, "Enter a valid email address.")
    return email


# ---------- sessions ----------
def _new_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    _q("INSERT INTO sessions VALUES(?,?,?)",
       (hashlib.sha256(token.encode()).hexdigest(), user_id, time.time() + SESSION_SECONDS), write=True)
    return token


def _public(u) -> dict:
    return {"id": u["id"], "name": u["name"], "email": u["email"]}


def _token_from(authorization: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def current_user(authorization: str | None = Header(None)) -> dict:
    token = _token_from(authorization)
    if token:
        row = _q("""SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id
                    WHERE s.token_hash = ? AND s.expires > ?""",
                 (hashlib.sha256(token.encode()).hexdigest(), time.time()), one=True)
        if row:
            return dict(row)
    raise HTTPException(401, "Please log in to continue.")


# ---------- plans owned by a customer ----------
def record_plan(plan_id: str, user_id: int, req, status: str | None = None) -> None:
    _q("INSERT OR IGNORE INTO plans(plan_id, user_id, destination, start_date, end_date, created_at, status) "
       "VALUES(?,?,?,?,?,?,?)",
       (plan_id, user_id, req.destination, req.start_date, req.end_date, time.time(), status), write=True)


def owns_plan(plan_id: str, user_id: int) -> bool:
    return _q("SELECT 1 FROM plans WHERE plan_id=? AND user_id=? AND deleted=0", (plan_id, user_id), one=True) is not None


def set_status(plan_id: str, status: str | None) -> None:
    if status:
        _q("UPDATE plans SET status=? WHERE plan_id=?", (status, plan_id), write=True)


def list_plans(user_id: int, limit: int = 20) -> list[dict]:
    rows = _q("""SELECT plan_id, destination, start_date, end_date, created_at, status
                 FROM plans WHERE user_id=? AND deleted=0 ORDER BY created_at DESC LIMIT ?""", (user_id, limit))
    return [dict(r) for r in rows]


def delete_plan(plan_id: str, user_id: int) -> bool:
    """Hide the plan from its owner. The row stays, so plans_today() still counts it."""
    if not owns_plan(plan_id, user_id):
        return False
    _q("UPDATE plans SET deleted=1 WHERE plan_id=? AND user_id=?", (plan_id, user_id), write=True)
    return True


def plans_today(user_id: int) -> int:
    row = _q("SELECT COUNT(*) AS n FROM plans WHERE user_id=? AND created_at>?",
             (user_id, time.time() - 86400), one=True)
    return row["n"]


# ---------- routes ----------
class SignupBody(BaseModel):
    name: str
    email: str
    password: str


class LoginBody(BaseModel):
    email: str
    password: str


def _google_client_id() -> str:
    return os.getenv("GOOGLE_CLIENT_ID", "").strip()


def _password_signup_allowed() -> bool:
    return not (_google_client_id() and os.getenv("ALLOW_PASSWORD_SIGNUP", "true").strip().lower() == "false")


def _verify_google(credential: str) -> dict:
    """Ask Google to verify the sign-in token, then check it was issued for OUR app."""
    cid = _google_client_id()
    if not cid:
        raise HTTPException(503, "Google login is not set up yet.")
    try:
        url = "https://oauth2.googleapis.com/tokeninfo?" + urllib.parse.urlencode({"id_token": credential})
        with urllib.request.urlopen(url, timeout=10) as r:
            info = json.load(r)
    except Exception:
        raise HTTPException(401, "Google sign-in failed. Please try again.")
    ok = (info.get("aud") == cid
          and info.get("iss") in ("accounts.google.com", "https://accounts.google.com")
          and str(info.get("email_verified")).lower() == "true"
          and info.get("sub") and info.get("email")
          and float(info.get("exp", 0)) > time.time())
    if not ok:
        raise HTTPException(401, "Google sign-in failed. Please try again.")
    return info


@router.get("/auth/config")
def auth_config():
    return {"google_client_id": _google_client_id() or None, "password_signup": _password_signup_allowed()}


class GoogleBody(BaseModel):
    credential: str


@router.post("/auth/google")
def google_login(body: GoogleBody):
    info = _verify_google(body.credential)
    sub, email = info["sub"], info["email"].lower()
    name = (info.get("name") or email.split("@")[0]).strip()[:80]
    user = _q("SELECT * FROM users WHERE google_sub=?", (sub,), one=True)
    if not user:
        old = _q("SELECT * FROM users WHERE email=?", (email,), one=True)
        if old:
            # Google has verified this email, so its real owner wins: link the account,
            # wipe any password set earlier by someone else, and sign out old sessions.
            _q("DELETE FROM sessions WHERE user_id=?", (old["id"],), write=True)
            _q("UPDATE users SET google_sub=?, pw_hash='', name=? WHERE id=?", (sub, name, old["id"]), write=True)
        else:
            _q("INSERT INTO users(name,email,pw_hash,created_at,google_sub) VALUES(?,?,?,?,?)",
               (name, email, "", time.time(), sub), write=True)
        user = _q("SELECT * FROM users WHERE google_sub=?", (sub,), one=True)
    return {"token": _new_session(user["id"]), "user": _public(user)}


@router.post("/auth/signup", status_code=201)
def signup(body: SignupBody):
    if not _password_signup_allowed():
        raise HTTPException(403, "Please sign up with Google.")
    name = body.name.strip()
    if not 1 <= len(name) <= 80:
        raise HTTPException(422, "Enter your name.")
    if not 8 <= len(body.password) <= 128:
        raise HTTPException(422, "Password must be 8 to 128 characters.")
    email = _clean_email(body.email)
    try:
        uid = _q("INSERT INTO users(name,email,pw_hash,created_at) VALUES(?,?,?,?)",
                 (name, email, _hash_pw(body.password), time.time()), write=True)
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with this email already exists. Log in instead.")
    user = _q("SELECT * FROM users WHERE id=?", (uid,), one=True)
    return {"token": _new_session(uid), "user": _public(user)}


@router.post("/auth/login")
def login(body: LoginBody):
    email = _clean_email(body.email)
    key = email
    now = time.time()
    recent = [t for t in _failed.get(key, []) if now - t < 300]
    if len(recent) >= 5:
        raise HTTPException(429, "Too many wrong attempts. Try again in a few minutes.")
    user = _q("SELECT * FROM users WHERE email=?", (email,), one=True)
    if not user or not _check_pw(body.password, user["pw_hash"]):
        _failed[key] = recent + [now]
        raise HTTPException(401, "Incorrect email or password.")
    _failed.pop(key, None)
    return {"token": _new_session(user["id"]), "user": _public(user)}


@router.get("/auth/me")
def me(user: dict = Depends(current_user)):
    return {"user": _public(user)}


@router.post("/auth/logout")
def logout(authorization: str | None = Header(None)):
    token = _token_from(authorization)
    if token:
        _q("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),), write=True)
    return {"ok": True}