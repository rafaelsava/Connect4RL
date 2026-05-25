import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from groups.Rafa.compare_memory import evaluate_head_to_head
from groups.Rafa.policy import RafaNoMemoryPolicy, RafaRootUCBPolicy


def write_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_lines(rows: list[dict[str, float | int]], path: Path) -> None:
    times = sorted({float(row["total_time"]) for row in rows})
    priors = sorted({int(row["global_prior_visits"]) for row in rows})

    plt.figure(figsize=(8, 4.8))
    for prior in priors:
        subset = [row for row in rows if int(row["global_prior_visits"]) == prior]
        subset = sorted(subset, key=lambda row: float(row["total_time"]))
        plt.plot(
            [float(row["total_time"]) for row in subset],
            [float(row["memory_win_rate"]) for row in subset],
            marker="o",
            label=f"prior={prior}",
        )

    plt.xscale("log")
    plt.xticks(times, [str(time) for time in times])
    plt.ylim(0, 1.02)
    plt.xlabel("Tiempo por agente/partida (s)")
    plt.ylabel("Win rate de Rafa vs RafaNoQ")
    plt.title("Impacto de memoria global segun presupuesto online")
    plt.grid(alpha=0.25)
    plt.legend(title="global_prior_visits")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_heatmap(rows: list[dict[str, float | int]], path: Path) -> None:
    times = sorted({float(row["total_time"]) for row in rows})
    priors = sorted({int(row["global_prior_visits"]) for row in rows})
    matrix = np.full((len(priors), len(times)), np.nan)

    for row in rows:
        i = priors.index(int(row["global_prior_visits"]))
        j = times.index(float(row["total_time"]))
        matrix[i, j] = float(row["memory_win_rate"])

    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(times)), [str(time) for time in times])
    ax.set_yticks(range(len(priors)), [str(prior) for prior in priors])
    ax.set_xlabel("Tiempo por agente/partida (s)")
    ax.set_ylabel("global_prior_visits")
    ax.set_title("Win rate de Rafa con memoria")

    for i in range(len(priors)):
        for j in range(len(times)):
            value = matrix[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, label="Win rate")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep de parametros para comparar Rafa con Q global vs RafaNoQ."
    )
    parser.add_argument("--times", nargs="+", type=float, default=[0.1, 0.3, 1.0, 2.0])
    parser.add_argument("--prior-visits", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--games-per-side", type=int, default=20)
    parser.add_argument("--seed", type=int, default=911)
    parser.add_argument("--qtable", type=Path, default=RafaRootUCBPolicy.DEFAULT_QTABLE_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("analytics") / "memory_sweep",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []
    total_jobs = len(args.times) * len(args.prior_visits)
    job = 0

    print("Sweep Rafa con memoria vs RafaNoQ")
    print(f"Q-table: {args.qtable.resolve()}")
    print(f"Juegos por lado: {args.games_per_side}")
    print(f"Combinaciones: {total_jobs}\n")

    for total_time in args.times:
        for prior in args.prior_visits:
            job += 1
            memory_policy = RafaRootUCBPolicy(
                total_time=total_time,
                qtable_path=args.qtable,
                auto_save=False,
                global_prior_visits=prior,
            )
            no_memory_policy = RafaNoMemoryPolicy(total_time=total_time)

            metrics = evaluate_head_to_head(
                memory_policy,
                no_memory_policy,
                args.games_per_side,
                rng,
                log_every=max(1, args.games_per_side),
            )

            row: dict[str, float | int] = {
                "total_time": total_time,
                "global_prior_visits": prior,
                "games_total": args.games_per_side * 2,
            }
            row.update(metrics)
            rows.append(row)

            print(
                f"[{job}/{total_jobs}] time={total_time}s prior={prior} | "
                f"Q win={row['memory_win_rate']:.2f}, "
                f"NoQ win={row['no_memory_win_rate']:.2f}, draw={row['draw_rate']:.2f}",
                flush=True,
            )

    csv_path = output_dir / "memory_sweep.csv"
    write_csv(rows, csv_path)
    plot_lines(rows, output_dir / "memory_sweep_lines.png")
    plot_heatmap(rows, output_dir / "memory_sweep_heatmap.png")

    print(f"\nCSV: {csv_path}")
    print(f"Line plot: {output_dir / 'memory_sweep_lines.png'}")
    print(f"Heatmap: {output_dir / 'memory_sweep_heatmap.png'}")


if __name__ == "__main__":
    main()
