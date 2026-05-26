# Group B - Connect4RL

Este directorio contiene el agente final de Group B para Connect 4, los scripts usados para entrenarlo y evaluarlo, y los artefactos generados durante los experimentos.

## Contenido

- `policy.py`: implementacion del agente `FVMCAgent`.
- `train_agent.py`: script para continuar entrenamiento y guardar el modelo.
- `run_experiments.py`: ejecuta varias configuraciones de entrenamiento y guarda resultados comparativos.
- `compare_q_memory.py`: compara entrenar desde cero vs reutilizar `Q-values`.
- `entrega.ipynb`: notebook de analisis y graficas.
- `artifacts/`: resultados de experimentos y resumentes en CSV/TXT.

## Archivos necesarios para ejecutar

Para usar este agente se necesita:

- este directorio completo `groups/Group B/`
- la carpeta base del proyecto con el paquete `connect4/`
- Python 3
- `numpy`
- para el notebook: `pandas`, `matplotlib`, `seaborn`, `ipython`

Los resultados ya generados se encuentran en:

- `artifacts/experiment_results.csv`
- `artifacts/q_memory_comparison.csv`
- `artifacts/training_summary.txt`

El modelo `fvmc_model.pkl` se genera cuando se entrena o guarda el agente.

## Idea general del agente

El agente usa `First-Visit Monte Carlo` para aprender valores `Q` sobre pares `(estado, accion)`. Tambien aplica dos reglas tacticas directas en tiempo de juego:

- si tiene una jugada ganadora inmediata, la toma
- si el rival puede ganar en el siguiente turno, lo bloquea

Cuando no hay una jugada tactica inmediata, elige la accion con mayor valor `Q`.

## Guia breve de uso

Todos los comandos se ejecutan desde la raiz del repositorio.

### 1. Entrenar o continuar entrenamiento

Entrenamiento simple:

```bash
python "groups/Group B/train_agent.py" --episodes 1000 --mode random --epsilon 0.2
```

Entrenamiento con preset:

```bash
python "groups/Group B/train_agent.py" --preset self_heavy
```

Presets disponibles:

- `self_only`
- `self_heavy`
- `self_refine`

Al terminar, el script guarda:

- `groups/Group B/fvmc_model.pkl`
- `groups/Group B/artifacts/training_summary.txt`

### 2. Ejecutar experimentos comparativos

Para correr todos los experimentos:

```bash
python "groups/Group B/run_experiments.py"
```

Para listar experimentos disponibles:

```bash
python "groups/Group B/run_experiments.py" --list
```

Para correr solo algunos experimentos:

```bash
python "groups/Group B/run_experiments.py" --only exp_random_5k_eps020 exp_mixed_8k_12k
```

Salida principal:

- `groups/Group B/artifacts/experiment_results.csv`

### 3. Comparar guardar Q-values vs entrenar desde cero

```bash
python "groups/Group B/compare_q_memory.py"
```

Salida principal:

- `groups/Group B/artifacts/q_memory_comparison.csv`

### 4. Abrir el notebook de analisis

```bash
jupyter notebook "groups/Group B/entrega.ipynb"
```

El notebook usa los CSV de `artifacts/` para mostrar tablas y graficas.

## Notas

- `policy.py` esta preparado para cargar un modelo existente si `fvmc_model.pkl` ya existe.
- Si no se desea autoentrenamiento al cargar el agente, los scripts usan `auto_train_if_missing=False`.
- Las evaluaciones comparan el agente frente a `Random` y frente a `Self` segun la logica definida en los scripts de experimentos.
