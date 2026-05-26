# Connect4RL - Agente Rafa

Guía rápida para usar y defender el agente de Rafael Salcedo.

## Archivo principal

El agente final está en `groups/Rafa/policy.py`.

La clase principal es `RafaImprovedPolicy`. Esa es la política que debe tomar el torneo cuando carga la carpeta `groups/Rafa`.

Idea del agente:

- Usa TBOPI: `trial-based online policy improvement`.
- Carga una Q-table global aprendida por self-play desde `groups/Rafa/rafa_q_values.pkl`.
- Usa esos Q-values como prior para una búsqueda local Root-UCB en cada turno.
- Hace rollouts heurísticos: intenta ganar, bloquear, evitar derrotas inmediatas y preferir columnas centrales.

El archivo `rafa_q_values.pkl` es un artefacto local grande y está ignorado por git.

## Versiones internas

En `groups/Rafa/policy.py` también están las versiones usadas para análisis:

- `RafaBasePolicy`: TBOPI sin memoria global y con rollouts aleatorios.
- `RafaQPolicy`: usa Q-table global, pero mantiene rollouts aleatorios.
- `RafaImprovedPolicy`: versión final con Q-table, rollouts heurísticos y selección robusta.

Esto permite explicar la evolución: base sin memoria, memoria global, y finalmente memoria más rollouts de mejor calidad.

## Entrenamiento

Entrenamiento por self-play con presupuesto por rollouts:

```powershell
python groups\Rafa\train_self_play.py --games 50000 --inner-rollouts 8 --workers 4 --global-prior-visits 10 --alpha 0.04 --gamma 0.995
```

Entrenamiento más parecido al torneo, con presupuesto por tiempo:

```powershell
python groups\Rafa\train_self_play.py --games 1000 --budget-mode time --training-total-time 60 --max-turn-time 5 --workers 1
```

Durante el entrenamiento se guarda progreso periódicamente en `groups/Rafa/rafa_q_values.pkl`.

## Comparaciones

Comparar Rafa contra RafaNoQ, RafaQ o RafaImproved en formato torneo:

```powershell
python groups\Rafa\compare_tournament_agents.py --include RafaQ RafaImproved --games-per-match 25 --total-time 60
```

Comparar un agente fijo contra otros:

```powershell
python groups\Rafa\compare_tournament_agents.py --target RafaImproved --opponents RafaQ RafaBase --games-per-match 25 --total-time 60
```

Los CSV se guardan en `groups/Rafa/analytics/tournament_style/`.

## Análisis para la entrega

Generar la validación mínima del reto: Random, self-play y curva corta de entrenamiento:

```powershell
python groups\Rafa\analyze_reto_requirements.py --games-vs-random 20 --self-play-games 20 --total-time 2 --training-checkpoints 0,50,100,250,500 --training-eval-games 10 --eval-total-time 0.3 --inner-rollouts 8
```

Sweep de hiperparámetros contra RafaBase:

```powershell
python groups\Rafa\sweep_versions.py --baseline RafaBase --agents RafaQ RafaImproved --games-per-side 10
```

Sweep directo de RafaImproved contra RafaQ:

```powershell
python groups\Rafa\sweep_versions.py --baseline RafaQ --agents RafaImproved --games-per-side 10 --output-dir groups\Rafa\analytics\version_sweep_vs_q
```

## Notebook y reporte

El notebook de análisis está en:

```text
groups/Rafa/entrega.ipynb
```

El reporte final en LaTeX está en:

```text
groups/Rafa/report/reporte_rafa.tex
```

Para recompilar el PDF:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory groups/Rafa/report groups/Rafa/report/reporte_rafa.tex
```

El PDF resultante queda en:

```text
groups/Rafa/report/reporte_rafa.pdf
```
