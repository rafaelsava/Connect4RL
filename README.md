# Connect4RL: politica final con MCTS

Este repositorio ejecuta partidas de Conecta 4 entre politicas ubicadas en
`groups/`. La politica de `groups/Final/policy.py` implementa un agente
`Head` basado en Monte Carlo Tree Search (MCTS) con UCB1, complementado con
reglas tacticas para jugar victorias inmediatas y evitar derrotas en el
siguiente turno.

El punto de entrada del proyecto es `main.py`. Este archivo busca
automaticamente todas las clases que heredan de `connect4.policy.Policy`
dentro de `groups/`, arma un torneo y reporta al campeon. Por esta razon,
`groups/Final/policy.py` no se corre directamente como un script: se carga al
ejecutar el torneo.

## Estructura relevante

```text
Connect4RL/
|-- main.py                    # Descubre politicas y ejecuta el torneo
|-- tournament.py              # Organiza rondas, juega partidas y guarda resultados
|-- connect4/
|   |-- connect_state.py       # Estado, movimientos legales y ganador del juego
|   |-- policy.py              # Interfaz base de las politicas
|   `-- utils.py               # Descubrimiento dinamico de politicas
|-- groups/
|   |-- Final/policy.py        # Politica Head (MCTS + reglas tacticas)
|   `-- Random/policy.py       # Rival aleatorio de ejemplo
`-- versus/                    # Archivos JSON producidos por los enfrentamientos
```

## Requisitos

- Python 3.12 o superior. `main.py` tambien carga la politica `Random`, que
  utiliza `typing.override`, disponible desde Python 3.12.
- `numpy`, para representar estados y escoger movimientos.
- `matplotlib`, importado por la visualizacion de `ConnectState`.
- `pydantic` 2.x, para serializar los resultados de cada enfrentamiento.

## Crear el entorno e instalar dependencias

Desde la raiz del repositorio:

```bash
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy matplotlib "pydantic>=2"
```

En Windows PowerShell, la activacion del entorno es:

```powershell
.\env\Scripts\Activate.ps1
```

Para comprobar que la politica final puede importarse:

```bash
python -c "from groups.Final.policy import Head; print(Head.__name__)"
```

La salida esperada es `Head`.

## Ejecutar la politica en el torneo

Con el entorno activo y desde la raiz del proyecto:

```bash
python main.py
```

`main.py` detecta al menos estas dos politicas:

- `Final`: la clase `Head` definida en `groups/Final/policy.py`.
- `Random`: la clase `OhYes` definida en `groups/Random/policy.py`.

El programa muestra los enfrentamientos, los ganadores de cada ronda y una
linea final con el campeon:

```text
Initial Matches: ...
Winners this round: ...
Champion: ...
```

Cada enfrentamiento tambien genera un archivo JSON en `versus/`, con los
tableros y acciones de las partidas disputadas.

La politica final busca durante hasta aproximadamente 2 segundos por turno y
mantiene un limite acumulado de 58 segundos por instancia. Por ello, ejecutar
el torneo completo puede tardar varios minutos dependiendo de la cantidad y
duracion de las partidas.

## Como funciona `groups/Final/policy.py`

### Interfaz `Head`

`Head` hereda de `Policy`, por lo que expone dos metodos usados por el
torneo:

- `mount(action_timeout=None)`: inicializa el tiempo usado por el agente y
  configura el presupuesto disponible por turno.
- `act(s)`: recibe el tablero como un arreglo `numpy`, decide una columna
  legal y devuelve su indice entre `0` y `6`.

El tablero usa la representacion de `ConnectState`: `0` es una celda vacia,
`-1` corresponde al jugador rojo y `1` al amarillo. La politica cuenta las
fichas del tablero para inferir a quien le corresponde jugar.

### Decisiones tacticas antes de buscar

Antes de iniciar MCTS, `act()` resuelve situaciones directas:

1. Si solo queda una columna legal, la devuelve inmediatamente.
2. `_winning_action()` comprueba si existe un movimiento que gane en el turno
   actual y lo juega.
3. `_actions_without_immediate_loss()` descarta movimientos que permitirian
   al rival ganar inmediatamente en el turno siguiente, siempre que exista
   alguna alternativa segura.

Esto evita gastar tiempo de busqueda en jugadas evidentes y reduce errores
tacticos de un rollout puramente aleatorio.

### Arbol MCTS y clase `Node`

Cada `Node` representa un estado alcanzado despues de realizar una accion y
almacena:

- `children`: hijos ya creados para acciones exploradas.
- `untried_actions`: columnas legales que aun no se han expandido.
- `visits`: numero de simulaciones que pasaron por el nodo.
- `value`: recompensa acumulada desde la perspectiva del jugador que acaba
  de mover.

Durante el presupuesto de tiempo asignado, la politica repite las cuatro
fases clasicas de MCTS:

1. **Seleccion**: `ucb1_child()` elige entre hijos expandidos equilibrando
   recompensa media y exploracion con UCB1.
2. **Expansion**: `expand()` crea un hijo a partir de una accion aun no
   probada.
3. **Simulacion**: `_rollout()` completa la partida usando movimientos
   aleatorios hasta encontrar victoria o empate.
4. **Retropropagacion**: `_backpropagate()` actualiza visitas y recompensas
   desde el nodo simulado hasta la raiz.

Al agotarse el tiempo, `best_action()` selecciona la accion mas visitada; si
hay empate practico, favorece la de mejor recompensa promedio.

### Control de tiempo

Las constantes principales de `Head` son:

| Constante | Valor | Proposito |
| --- | ---: | --- |
| `EXPLORATION` | `sqrt(2)` | Peso exploratorio de UCB1. |
| `GLOBAL_TIME_LIMIT` | `58.0` s | Maximo acumulado de busqueda del agente. |
| `TURN_TIME_LIMIT` | `2.0` s | Presupuesto normal para una accion. |
| `MIN_TURN_BUDGET` | `0.05` s | Margen minimo reservado para poder responder. |

Si el presupuesto global esta por agotarse, la politica deja de simular y
escoge una accion valida entre las previamente filtradas.

## Flujo completo de una ejecucion

1. `main.py` usa `find_importable_classes("groups", Policy)` para importar
   las politicas participantes.
2. `run_tournament()` construye los emparejamientos y ejecuta cada ronda.
3. Antes de cada partida, `play()` instancia las politicas y llama a
   `mount()`.
4. En cada turno, la politica activa recibe `state.board` y responde mediante
   `act()`.
5. `ConnectState.transition()` aplica la accion y alterna el jugador hasta
   que haya cuatro fichas conectadas o se llene el tablero.
6. El historial del enfrentamiento se guarda como JSON en `versus/` y el
   torneo continua hasta imprimir al campeon.