"""
Connect-4 interactivo — elige jugadores antes de cada partida.
Uso: python play.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")

from connect4.connect_state import ConnectState
from connect4.policy import Policy
from connect4.utils import find_importable_classes

# ── Colores ANSI ─────────────────────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"
CYAN   = "\033[96m"

AGENT_TIME = 60.0   # segundos por partida para cada agente

# ── Descubrir agentes en la carpeta groups/ ───────────────────────────────────
_agents: dict[str, type] = find_importable_classes("groups", Policy)
AGENT_NAMES = sorted(_agents.keys())   # ej. ["Group A", "Group B", "Group C"]


# ── Render ────────────────────────────────────────────────────────────────────
def render(state: ConnectState) -> None:
    print()
    print(f"  {DIM}1   2   3   4   5   6   7{RESET}")
    print(f"  {'─' * 27}")
    for row in range(state.ROWS):
        print("│ ", end="")
        for col in range(state.COLS):
            val = state.board[row, col]
            if val == -1:
                print(f"{RED}●{RESET}   ", end="")
            elif val == 1:
                print(f"{YELLOW}●{RESET}   ", end="")
            else:
                print(f"{DIM}·{RESET}   ", end="")
        print("│")
    print(f"  {'─' * 27}")
    print()


# ── Helpers de selección ──────────────────────────────────────────────────────
def pick_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        print(f"  Elige un número entre {lo} y {hi}.")


def pick_player(slot: str) -> tuple[str, Policy | None]:
    """
    Devuelve (nombre, instancia_o_None).
    None indica que ese slot es el jugador humano.
    """
    print(f"\n  {BOLD}Jugador {slot}{RESET}")
    print(f"    0 → {CYAN}Humano{RESET}")
    for i, name in enumerate(AGENT_NAMES, start=1):
        print(f"    {i} → {name}")

    choice = pick_int(f"  Opción [0-{len(AGENT_NAMES)}]: ", 0, len(AGENT_NAMES))

    if choice == 0:
        return "Humano", None
    name = AGENT_NAMES[choice - 1]
    cls  = _agents[name]
    try:
        policy = cls(total_time=AGENT_TIME)
    except TypeError:
        policy = cls()
    return name, policy


def color_label(player_val: int) -> str:
    return f"{RED}Rojo{RESET}" if player_val == -1 else f"{YELLOW}Amarillo{RESET}"


def ask_column(state: ConnectState) -> int:
    free = state.get_free_cols()
    while True:
        raw = input(f"  Tu turno — elige columna {[c + 1 for c in free]}: ").strip()
        if not raw.isdigit():
            print("  Ingresa un número.")
            continue
        col = int(raw) - 1
        if col not in free:
            print("  Columna no disponible, elige otra.")
            continue
        return col


# ── Lógica de partida ─────────────────────────────────────────────────────────
def play_game(
    red_name: str,   red_policy: Policy | None,
    yel_name: str,   yel_policy: Policy | None,
) -> None:
    """
    Corre una partida completa.
    Si policy es None, el humano elige la columna.
    """
    # Montar agentes
    if red_policy is not None:
        red_policy.mount()
    if yel_policy is not None:
        yel_policy.mount()

    policies = {-1: red_policy, 1: yel_policy}
    names    = {-1: red_name,   1: yel_name}

    print(f"\n  {color_label(-1)} {BOLD}{red_name}{RESET}  vs  "
          f"{color_label(1)} {BOLD}{yel_name}{RESET}\n")

    state = ConnectState()

    while not state.is_final():
        render(state)
        current = state.player
        policy  = policies[current]
        name    = names[current]

        if policy is None:
            col = ask_column(state)
        else:
            print(f"  {name} pensando…", end="", flush=True)
            t0  = time.perf_counter()
            col = policy.act(state.board)
            elapsed = time.perf_counter() - t0
            print(f"\r  {name} jugó columna {col + 1}  ({elapsed:.2f}s)          ")

        state = state.transition(col)

    render(state)

    winner = state.get_winner()
    if winner == 0:
        print(f"{BOLD}  Empate.{RESET}\n")
    else:
        winner_name  = names[winner]
        winner_color = color_label(winner)
        print(f"{BOLD}  Ganó {winner_color} — {winner_name}{RESET}\n")


# ── Menú principal ────────────────────────────────────────────────────────────
def main() -> None:
    print(f"\n{BOLD}  ══════════════════════════════{RESET}")
    print(f"{BOLD}       Connect-4 Interactivo    {RESET}")
    print(f"{BOLD}  ══════════════════════════════{RESET}")
    print(f"  {DIM}Agentes disponibles: {', '.join(AGENT_NAMES)}{RESET}")
    print(f"  {DIM}Presiona Ctrl+C para salir.{RESET}")

    while True:
        print(f"\n  {BOLD}─── Nueva partida ───{RESET}")
        red_name, red_policy = pick_player(f"{color_label(-1)} (Rojo, mueve primero)")
        yel_name, yel_policy = pick_player(f"{color_label(1)} (Amarillo)")

        if red_name == "Humano" and yel_name == "Humano":
            print(f"\n  {DIM}Modo: Humano vs Humano{RESET}")
        elif red_name == "Humano" or yel_name == "Humano":
            print(f"\n  {DIM}Modo: Humano vs Agente  |  Tiempo agente: {AGENT_TIME}s{RESET}")
        else:
            print(f"\n  {DIM}Modo: Agente vs Agente  |  Tiempo por agente: {AGENT_TIME}s{RESET}")

        play_game(red_name, red_policy, yel_name, yel_policy)

        raw = input("  ¿Jugar de nuevo? [s/n]: ").strip().lower()
        if raw != "s":
            print("\n  ¡Hasta luego!\n")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Juego interrumpido.\n")
