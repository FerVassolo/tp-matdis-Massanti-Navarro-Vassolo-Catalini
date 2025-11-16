# Trabajo Práctico: Los 100 Grafos Porteños

El Gobierno de la Ciudad de Buenos Aires necesita una herramienta capaz de analizar la red de transporte, la red
eléctrica y la red de distribución de agua de la ciudad. Tu tarea es implementar los algoritmos necesarios para realizar
distintas consultas sobre estas redes y ayudar a tomar decisiones.

Cada red se modelará como un grafo independiente, cuyos vértices serán los barrios porteños.

---

## Instalación

Este proyecto usa [uv](https://docs.astral.sh/uv/) para la gestión del entorno.

```bash
# Instalar uv (si no lo tienen)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Configurar el proyecto
uv sync
```

---

## Formato de los Grafos

### Grafo Eléctrico (simple)

Archivo: `resources/grafo_electrico_48.txt`

```
Barrio1 Barrio2
Barrio3 Barrio4
...
```

### Grafo Vial (ponderado - tiempo en minutos)

Archivo: `resources/grafo_vial_48.txt`

```
Barrio1 Barrio2 Tiempo
Barrio3 Barrio4 Tiempo
...
```

### Grafo Hídrico (simple)

Archivo: `resources/grafo_hidrico_48.txt`

```
Barrio1 Barrio2
Barrio3 Barrio4
...
```

---

## Implementación

Deben implementar `src/main.py` con las funciones que `run.py` necesita:

Además, deberán implementar las funciones de algoritmos necesarias para resolver cada tipo de consulta.

---

## Ejecución

Una vez implementado `src/main.py`, ejecutar:

```bash
# Con el ejemplo simple (8 barrios)
uv run run.py resources/ejemplo/ejemplo_electrico.txt resources/ejemplo/ejemplo_vial.txt resources/ejemplo/ejemplo_hidrico.txt resources/ejemplo/ejemplo_consultas.txt resources/ejemplo/ejemplo_respuestas.txt

# Con el ejemplo completo (48 barrios)
uv run run.py resources/ejemplo-48/grafo_electrico_48.txt resources/ejemplo-48/grafo_vial_48.txt resources/ejemplo-48/grafo_hidrico_48.txt resources/ejemplo-48/consultas.txt resources/ejemplo-48/respuestas.txt
```

---

## Módulo Output (Proporcionado)

El módulo `src/output.py` ya está implementado con funciones para formatear las salidas:

```python
from src.output import (
    format_componentes_conexos,
    format_orden_fallos,
    format_camino_minimo,
    format_simulacion_corte,
    format_ruta_recoleccion,
    format_plantas_asignadas,
    format_puentes_y_articulaciones
)
```

Use estas funciones para imprimir los resultados tal como se muestran en los ejemplos.

---

## Criterios de Evaluación

- Implementación correcta de todas las consultas
- Manejo correcto de casos borde (grafos desconectados, sin camino, etc.)
- Código limpio y bien documentado
- Uso correcto de los formatters proporcionados
- Funciona con los grafos de ejemplo y los de 48 barrios
- Funciona contra los grafos privados con los que se evaluaran las implementaciones

---

## Entrega

Enviar un archivo zip a rodrigo.pazos@ing.austral.edu.ar con el proyecto funcionando.
Ese proyecto debera soportar una ejecucion contra cualquier grafo y operacion válidos tal como están presentados en los
ejemplos