from __future__ import annotations

import json
import random
from dataclasses import dataclass

import chess
import chess.svg

from src.stock_chat import call_groq_json

CHESS_SYSTEM_PROMPT = (
    "You are a chess engine playing as Black. You will be given the current "
    "position as FEN, the move history, and a list of every legal move in "
    "UCI notation. Pick exactly one move from that list — the one that gives "
    "you the best chance of winning. "
    'Respond with ONLY JSON in this exact shape: {"move": "<uci move from the list>"}. '
    "No commentary, no markdown, no extra keys."
)

MAX_LLM_RETRIES = 3


@dataclass
class MoveResult:
    move: chess.Move
    uci: str
    san: str
    from_llm: bool
    fell_back: bool = False


def new_board() -> chess.Board:
    return chess.Board()


def legal_moves_uci(board: chess.Board) -> list[str]:
    return [m.uci() for m in board.legal_moves]


def parse_human_move(board: chess.Board, move_text: str) -> chess.Move:
    """Parse SAN ('Nf3') or UCI ('g1f3'). Raises ValueError if unparseable or illegal."""
    move_text = move_text.strip()
    try:
        return board.parse_san(move_text)
    except ValueError:
        pass
    try:
        move = chess.Move.from_uci(move_text.lower())
    except ValueError:
        raise ValueError(f"Could not parse move '{move_text}'.")
    if move not in board.legal_moves:
        raise ValueError(f"'{move_text}' is not a legal move in this position.")
    return move


def _extract_move_uci(raw: str) -> str | None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    move = data.get("move") if isinstance(data, dict) else None
    return move.strip().lower() if isinstance(move, str) else None


def request_llm_move(board: chess.Board, model_id: str, history_san: list[str]) -> MoveResult:
    """Ask Groq for Black's move. Retries on illegal replies, falls back to a random legal move."""
    legal = legal_moves_uci(board)
    feedback = ""
    for _ in range(MAX_LLM_RETRIES):
        user_prompt = (
            f"FEN: {board.fen()}\n"
            f"Move history (SAN): {' '.join(history_san) or '(none yet)'}\n"
            f"Legal moves (UCI): {', '.join(legal)}\n"
            f"{feedback}"
            "Reply with one move from the legal moves list."
        )
        raw = call_groq_json(
            system_prompt=CHESS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=model_id,
        )
        move_uci = _extract_move_uci(raw)
        if move_uci and move_uci in legal:
            move = chess.Move.from_uci(move_uci)
            return MoveResult(move=move, uci=move_uci, san=board.san(move), from_llm=True)
        feedback = f"Your last reply ({raw[:80]!r}) was not one of the legal moves. "

    move = random.choice(list(board.legal_moves))
    return MoveResult(move=move, uci=move.uci(), san=board.san(move), from_llm=False, fell_back=True)


def board_svg(board: chess.Board, lastmove: chess.Move | None = None, size: int = 420) -> str:
    return chess.svg.board(board, lastmove=lastmove, size=size, flipped=False)


def game_status(board: chess.Board) -> str | None:
    """Human-readable end-of-game message, or None if the game is still in progress."""
    if not board.is_game_over():
        return None
    outcome = board.outcome()
    if outcome is None:
        return "Game over."
    if outcome.winner is None:
        reason = outcome.termination.name.replace("_", " ").title()
        return f"Draw — {reason}."
    winner = "White (you)" if outcome.winner else "Black (Groq)"
    return f"Checkmate — {winner} win."
