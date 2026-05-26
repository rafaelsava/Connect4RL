# Agente MCTS Puro (Tongue) - Connect-4

Este directorio contiene la implementación del agente **Tongue**, una política de juego para Connect-4 basada en **Monte Carlo Tree Search (MCTS)** puro utilizando el criterio **UCB1** para la asignación inteligente del cómputo de búsqueda.

## 🚀 Características Principales

*   **Búsqueda Anytime:** Se adapta dinámicamente al tiempo disponible. El agente realiza tantas simulaciones como sea posible dentro del límite de tiempo asignado por turno.
*   **Filtros Tácticos Inmediatos:**
    *   **Victoria Inmediata:** Detecta de forma determinista si existe un movimiento que garantice la victoria en el turno actual y lo ejecuta sin realizar búsqueda.
    *   **Prevención de Derrota Inmediata:** Identifica si el oponente tiene una amenaza de victoria en su próximo turno y filtra las columnas para bloquear obligatoriamente dicha amenaza.
*   **Selección UCB1 Estándar:** Balancea la explotación (ramas con alto porcentaje de victoria) y la exploración (ramas poco visitadas) con una constante de exploración $C = \sqrt{2.0}$.

---

## 🛠️ Arquitectura del Código

El archivo [`policy.py`](groups/Fermin_MCTS/policy.py) se divide en dos componentes principales:

### 1. Clase `Node` (Nodo del Árbol)
Representa un estado del juego en el árbol de búsqueda de MCTS. Almacena:
*   `visits`: Cantidad de veces que se ha simulado una partida pasando por este nodo.
*   `value`: El valor acumulado de recompensa (+1 por victoria propia, -1 por victoria del rival, 0 por empate).
*   `children`: Diccionario con las acciones válidas y sus respectivos nodos descendientes.
*   `untried_actions`: Acciones legales que aún no han sido expandidas en el árbol.

### 2. Clase `Tongue` (Política de MCTS)
Implementa el ciclo principal de toma de decisiones heredando de la clase abstracta `Policy`.

---

## ⏱️ Configuración del Presupuesto de Tiempo

El agente gestiona estrictamente sus recursos numéricos para evitar descalificaciones por exceder el tiempo de juego:

*   `GLOBAL_TIME_LIMIT = 58.0` segundos: Límite de tiempo acumulado disponible para toda la partida (margen de seguridad sobre el límite estándar de 60s).
*   `TURN_TIME_LIMIT = 2.0` segundos: Tiempo de búsqueda asignado para resolver cada turno individualmente.
*   `MIN_TURN_BUDGET = 0.05` segundos: Presupuesto mínimo de seguridad. Si el tiempo global restante o el tiempo de turno cae por debajo de este umbral, el agente elige un movimiento seguro al azar para evitar descalificaciones tácticas.

---

## 🔄 Ciclo de Búsqueda MCTS de la Clase `Tongue`

En cada turno en el que no se detecten jugadas obvias de victoria o bloqueo inmediato, MCTS ejecuta en bucle las siguientes 4 fases hasta agotar su presupuesto de tiempo:

1.  **Selección (Selection):** Búsqueda recursiva desde la raíz del árbol a través de los nodos completamente expandidos. Para elegir el siguiente nodo hijo, se calcula la puntuación **UCB1**:
    $$\text{score} = \frac{v_i}{n_i} + C \times \sqrt{\frac{\ln(N)}{n_i}}$$
    *Donde $v_i$ es el valor acumulado del nodo hijo, $n_i$ sus visitas, $N$ las visitas del padre y $C = \sqrt{2.0}$ la constante de exploración.*
2.  **Expansión (Expansion):** Si el algoritmo llega a un nodo que no está completamente expandido (tiene acciones legales sin probar), selecciona una acción al azar, la ejecuta mediante una transición de estado y añade un nuevo nodo hijo al árbol.
3.  **Simulación (Simulation / Rollout):** Realiza una partida rápida ficticia (rollout) desde el estado del nuevo nodo hasta un estado terminal (victoria, derrota o empate) seleccionando movimientos aleatorios.
4.  **Retropropagación (Backpropagation):** Propaga el resultado de la partida simulada hacia la raíz actualizando de forma alternante las métricas `visits` y `value` de todos los nodos ancestros.

Al finalizar el límite de tiempo, el agente selecciona la acción que corresponde al hijo directo de la raíz que acumuló **mayor cantidad de visitas** (criterio de robustez).

---

## 💻 Guía de Uso Rápido

Para instanciar y utilizar esta política en un entorno Connect-4:

```python
import numpy as np
from groups.Fermin_MCTS.policy import Tongue

# 1. Instanciar la política
agente = Tongue()

# 2. Configurar la política (opcionalmente pasando un tiempo de turno límite diferente)
agente.mount(action_timeout=2.0)

# 3. Representar un tablero (ej. vacío de 6x7)
# 0 = Celda vacía, -1 = Fichas Rojas (Jugador 1), 1 = Fichas Amarillas (Jugador 2)
tablero = np.zeros((6, 7), dtype=int)

# 4. Solicitar la mejor acción al agente
columna_elegida = agente.act(tablero)
print(f"El agente Tongue decide jugar en la columna: {columna_elegida}")
```
