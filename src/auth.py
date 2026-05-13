"""Authentication module — SQLite-backed, bcrypt-hashed, role-aware.

Roles:
  superadmin  — full access + user management panel
  user        — normal app access

Database: data/users.db (SQLite, created automatically on first run)

Superadmin account is seeded on first run using the constant below.
Change the password immediately after first login via the Admin Panel.
"""
from __future__ import annotations

import sqlite3
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import bcrypt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = Path("data/users.db")

SUPERADMIN_EMAIL    = "ijazfahad683@gmail.com"
SUPERADMIN_NAME     = "Fahad (Super Admin)"
SUPERADMIN_DEFAULT_PASSWORD = "Admin@2024!"   # must be changed after first login

ROLES = ["user", "superadmin"]


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    """Create tables and seed the superadmin if the DB is new."""
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                email       TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                password    TEXT    NOT NULL,
                role        TEXT    NOT NULL DEFAULT 'user',
                created_at  TEXT    NOT NULL,
                last_login  TEXT
            )
        """)

    # Seed superadmin only if no users exist yet
    with _conn() as con:
        row = con.execute("SELECT id FROM users WHERE email = ? COLLATE NOCASE",
                          (SUPERADMIN_EMAIL,)).fetchone()
        if not row:
            hashed = bcrypt.hashpw(
                SUPERADMIN_DEFAULT_PASSWORD.encode(), bcrypt.gensalt()
            ).decode()
            con.execute(
                "INSERT INTO users (name, email, password, role, created_at) VALUES (?,?,?,?,?)",
                (SUPERADMIN_NAME, SUPERADMIN_EMAIL, hashed, "superadmin",
                 datetime.now().isoformat()),
            )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()))


def _valid_password(pw: str) -> tuple[bool, str]:
    """Returns (ok, reason). Password must be 8+ chars with upper, lower, digit."""
    if len(pw) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", pw):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", pw):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"\d", pw):
        return False, "Password must contain at least one digit."
    return True, ""


# ---------------------------------------------------------------------------
# Auth operations
# ---------------------------------------------------------------------------
def register_user(name: str, email: str, password: str) -> tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    name  = name.strip()
    email = email.strip().lower()

    if not name:
        return False, "Name is required."
    if not _valid_email(email):
        return False, "Invalid email address."
    ok, reason = _valid_password(password)
    if not ok:
        return False, reason

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        with _conn() as con:
            con.execute(
                "INSERT INTO users (name, email, password, role, created_at) VALUES (?,?,?,?,?)",
                (name, email, hashed, "user", datetime.now().isoformat()),
            )
        return True, "Account created! You can now sign in."
    except sqlite3.IntegrityError:
        return False, "An account with that email already exists."


def login_user(email: str, password: str) -> tuple[bool, str, Optional[dict]]:
    """Verify credentials. Returns (success, message, user_dict | None)."""
    email = email.strip().lower()
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE", (email,)
        ).fetchone()

    if not row:
        return False, "No account found with that email.", None

    if not bcrypt.checkpw(password.encode(), row["password"].encode()):
        return False, "Incorrect password.", None

    # Update last_login
    with _conn() as con:
        con.execute("UPDATE users SET last_login = ? WHERE id = ?",
                    (datetime.now().isoformat(), row["id"]))

    user = {
        "id":         row["id"],
        "name":       row["name"],
        "email":      row["email"],
        "role":       row["role"],
        "created_at": row["created_at"],
    }
    return True, f"Welcome back, {row['name']}!", user


def change_password(user_id: int, old_pw: str, new_pw: str) -> tuple[bool, str]:
    """Change password — requires correct old password."""
    with _conn() as con:
        row = con.execute("SELECT password FROM users WHERE id = ?",
                          (user_id,)).fetchone()
    if not row:
        return False, "User not found."
    if not bcrypt.checkpw(old_pw.encode(), row["password"].encode()):
        return False, "Current password is incorrect."
    ok, reason = _valid_password(new_pw)
    if not ok:
        return False, reason
    hashed = bcrypt.hashpw(new_pw.encode(), bcrypt.gensalt()).decode()
    with _conn() as con:
        con.execute("UPDATE users SET password = ? WHERE id = ?", (hashed, user_id))
    return True, "Password changed successfully."


# ---------------------------------------------------------------------------
# Admin operations (superadmin only)
# ---------------------------------------------------------------------------
def list_users() -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT id, name, email, role, created_at, last_login FROM users ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def set_user_role(user_id: int, role: str) -> tuple[bool, str]:
    if role not in ROLES:
        return False, f"Invalid role. Must be one of: {ROLES}"
    with _conn() as con:
        con.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    return True, "Role updated."


def delete_user(user_id: int) -> tuple[bool, str]:
    with _conn() as con:
        # Protect the superadmin account from deletion
        row = con.execute("SELECT email FROM users WHERE id = ?", (user_id,)).fetchone()
        if row and row["email"].lower() == SUPERADMIN_EMAIL.lower():
            return False, "The superadmin account cannot be deleted."
        con.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return True, "User deleted."


def admin_reset_password(user_id: int, new_pw: str) -> tuple[bool, str]:
    """Superadmin force-reset any user's password (no old password required)."""
    ok, reason = _valid_password(new_pw)
    if not ok:
        return False, reason
    hashed = bcrypt.hashpw(new_pw.encode(), bcrypt.gensalt()).decode()
    with _conn() as con:
        con.execute("UPDATE users SET password = ? WHERE id = ?", (hashed, user_id))
    return True, "Password reset successfully."


def get_user_count() -> int:
    with _conn() as con:
        return con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
