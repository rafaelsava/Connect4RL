# Connect-4: agente principal MCTS

El agente principal del proyecto se encuentra en:

```text
groups/Fermin_MCTS/policy.py
```

Su clase concreta es `Head`, una politica basada en Monte Carlo Tree Search
(MCTS) con seleccion UCB1, deteccion de victoria inmediata y bloqueo de
derrota inmediata antes de iniciar la busqueda.

## Requisitos

Ejecute los comandos desde la raiz del repositorio. El proyecto incluye un
entorno virtual local:

```bash
source env/bin/activate
```

## Uso directo del agente principal

```python
import numpy as np
from groups.Fermin_MCTS.policy import Head

agent = Head()
agent.mount(action_timeout=0.5)

board = np.zeros((6, 7), dtype=int)
column = agent.act(board)
print(column)
```

El tablero utiliza:

| Valor | Significado |
|---:|---|
| `0` | Celda libre |
| `-1` | Jugador que inicia |
| `1` | Segundo jugador |

`act(board)` retorna la columna elegida como entero entre `0` y `6`.

## Parametro de tiempo

El recurso numerico principal es el presupuesto por turno:

```python
agent.mount(action_timeout=0.5)
```

Valores menores reducen el costo computacional, pero tambien el numero de
simulaciones MCTS disponibles para tomar la decision.

## Ejecutar el torneo del proyecto

```bash
env/bin/python main.py
```

`main.py` descubre automaticamente las politicas dentro de `groups/` y
ejecuta el torneo. El agente principal se registra a partir de la carpeta
`Fermin_MCTS`.

## Evaluacion experimental

El script [benchmark_entrega.py](benchmark_entrega.py) carga directamente:

```text
groups/Fermin_MCTS/policy.py
groups/Fermin_MCTS_Time&C/policy.py
groups/Random/policy.py
```

Para generar resultados reales por presupuesto:

```bash
env/bin/python benchmark_entrega.py --games 10
```

Los resultados se exportan en:

```text
versus/benchmark_entrega.csv
versus/benchmark_entrega_partidas.csv
```

El notebook [entrega.ipynb](entrega.ipynb) contiene las graficas y el
analisis del agente base, la version alternativa y el oponente aleatorio.

## Archivos relevantes

| Archivo | Descripcion |
|---|---|
| `groups/Fermin_MCTS/policy.py` | Agente principal MCTS (`Head`) |
| `groups/Fermin_MCTS_Time&C/policy.py` | Variante experimental |
| `groups/Random/policy.py` | Agente aleatorio de control |
| `benchmark_entrega.py` | Generacion de resultados experimentales |
| `entrega.ipynb` | Analisis y visualizaciones |
