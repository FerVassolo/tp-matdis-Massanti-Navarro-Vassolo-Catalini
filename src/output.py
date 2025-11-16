"""
Módulo de formatters para presentación de resultados de análisis de grafos.
Proporciona funciones para formatear diferentes tipos de salidas de manera legible.
"""

def wrap_list(items, prefix="  ", max_width=72):
    """
    Envuelve una lista de items en múltiples líneas si es necesario.

    Args:
        items: Lista de strings a formatear
        prefix: Prefijo para cada línea
        max_width: Ancho máximo de línea

    Returns:
        String formateado con saltos de línea apropiados
    """
    lines = []
    line = prefix
    for j, item in enumerate(items):
        if len(line) + len(item) + 2 > max_width:
            lines.append(line)
            line = prefix
        line += item
        if j < len(items) - 1:
            line += ", "
    lines.append(line)
    return "\n".join(lines)


def format_componentes_conexos(componentes):
    """
    Formatea la salida de componentes conexos.

    Args:
        componentes: Lista de listas, cada una representa un componente conexo

    Returns:
        String formateado con los componentes conexos
    """
    output = []
    output.append("=" * 60)
    output.append("COMPONENTES CONEXOS - RED ELÉCTRICA")
    output.append("=" * 60)

    # Ordenar componentes: primero por tamaño (desc), luego lexicográficamente por primer nodo
    componentes_sorted = sorted(
        [sorted(comp) for comp in componentes],
        key=lambda c: (-len(c), c[0] if c else "")
    )

    for i, comp in enumerate(componentes_sorted, 1):
        output.append(f"Componente {i} ({len(comp)} nodos):")
        output.append(wrap_list(comp))

    output.append("")
    return "\n".join(output)


def format_orden_fallos(nodos_grados):
    """
    Formatea la salida de orden de fallos agrupado por grado.

    Args:
        nodos_grados: Lista de tuplas (nodo, grado)

    Returns:
        String formateado con el orden de fallos
    """
    from itertools import groupby

    output = []
    output.append("=" * 60)
    output.append("ORDEN DE FALLOS - RED ELÉCTRICA")
    output.append("=" * 60)
    output.append("Nodos ordenados por criticidad (menor grado = más crítico):")
    output.append("")

    # Asegurar orden determinista: por grado, luego alfabéticamente
    nodos_grados_sorted = sorted(nodos_grados, key=lambda x: (x[1], x[0]))

    for grado, nodos_grupo in groupby(nodos_grados_sorted, key=lambda x: x[1]):
        nodos_list = sorted([n[0] for n in nodos_grupo])
        output.append(f"Grado {grado} ({len(nodos_list)} nodos):")
        output.append(wrap_list(nodos_list))
        output.append("")

    output.append("")
    return "\n".join(output)


def format_camino_minimo(origen, destino, distancia, camino):
    """
    Formatea la salida de un camino mínimo.

    Args:
        origen: Nodo de origen
        destino: Nodo de destino
        distancia: Distancia total (float o inf)
        camino: Lista de nodos en el camino

    Returns:
        String formateado con el camino mínimo
    """
    output = []
    output.append("-" * 60)
    output.append(f"CAMINO MÍNIMO: {origen} → {destino}")
    output.append("-" * 60)

    if distancia == float('inf'):
        output.append("Resultado: NO HAY CAMINO DISPONIBLE")
    else:
        output.append(f"Distancia total: {distancia} minutos")
        output.append(f"Ruta: {' → '.join(camino)}")

    output.append("")
    return "\n".join(output)


def format_simulacion_corte(origen, destino, cortes, distancia, camino):
    """
    Formatea la salida de una simulación de corte.

    Args:
        origen: Nodo de origen
        destino: Nodo de destino
        cortes: Lista de nodos cortados
        distancia: Distancia total (float o inf)
        camino: Lista de nodos en el camino alternativo

    Returns:
        String formateado con la simulación de corte
    """
    output = []
    output.append("-" * 60)
    output.append(f"SIMULACIÓN DE CORTE: {origen} → {destino}")
    # Ordenar cortes alfabéticamente para determinismo
    output.append(f"Nodos cortados: {', '.join(sorted(cortes))}")
    output.append("-" * 60)

    if distancia == float('inf'):
        output.append("Resultado: NO HAY RUTA ALTERNATIVA")
    else:
        output.append(f"Distancia total: {distancia} minutos")
        output.append(f"Ruta alternativa: {' → '.join(camino)}")

    output.append("")
    return "\n".join(output)


def format_ruta_recoleccion(camino, nodos_por_linea=8):
    """
    Formatea la salida de una ruta de recolección.

    Args:
        camino: Lista de nodos en el camino de recolección
        nodos_por_linea: Cantidad de nodos a mostrar por línea

    Returns:
        String formateado con la ruta de recolección
    """
    output = []
    output.append("=" * 60)
    output.append("RUTA DE RECOLECCIÓN DE BASURA")
    output.append("=" * 60)
    output.append(f"Total de paradas: {len(camino)}")
    output.append(f"Inicio: {camino[0]}")
    output.append(f"Fin: {camino[-1]}")
    output.append("")
    output.append("Recorrido completo:")

    # Dividir en grupos para mejor legibilidad
    for i in range(0, len(camino), nodos_por_linea):
        grupo = camino[i:i+nodos_por_linea]
        line = f"  {' → '.join(grupo)}"
        if i + nodos_por_linea < len(camino):
            line += " →"
        output.append(line)

    output.append("")
    return "\n".join(output)


def format_plantas_asignadas(plantas, asignaciones):
    """
    Formatea la salida de plantas asignadas por barrio.

    Args:
        plantas: Lista de nombres de plantas
        asignaciones: Diccionario {barrio: planta_asignada}

    Returns:
        String formateado con las asignaciones
    """
    output = []
    output.append("=" * 60)
    output.append("PLANTAS DE AGUA ASIGNADAS POR BARRIO")
    output.append("=" * 60)
    # Ordenar plantas alfabéticamente para determinismo
    plantas_sorted = sorted(plantas)
    output.append(f"Plantas disponibles: {', '.join(plantas_sorted)}")
    output.append("")

    # Agrupar por planta, ordenar barrios alfabéticamente
    for planta in plantas_sorted:
        barrios = sorted([b for b, p in asignaciones.items() if p == planta])
        output.append(f"Planta {planta} ({len(barrios)} barrios):")
        output.append(wrap_list(barrios))
        output.append("")

    output.append("")
    return "\n".join(output)


def format_puentes_y_articulaciones(articulaciones, puentes):
    """
    Formatea la salida de puntos críticos (articulaciones y puentes).

    Args:
        articulaciones: Lista de nodos articulación
        puentes: Lista de tuplas (u, v) representando aristas puente

    Returns:
        String formateado con puntos críticos
    """
    output = []
    output.append("=" * 60)
    output.append("PUNTOS CRÍTICOS - RED HÍDRICA")
    output.append("=" * 60)
    output.append("")

    # Articulaciones - ordenar alfabéticamente
    articulaciones_sorted = sorted(articulaciones)
    output.append(f"ARTICULACIONES ({len(articulaciones_sorted)} nodos críticos):")
    output.append("(Nodos cuya eliminación desconecta el grafo)")
    output.append("")

    if articulaciones_sorted:
        output.append(wrap_list(articulaciones_sorted))
        output.append("")
    else:
        output.append("  Ninguna")
        output.append("")

    # Puentes - ordenar alfabéticamente
    puentes_sorted = sorted(puentes, key=lambda x: (x[0], x[1]))
    output.append(f"PUENTES ({len(puentes_sorted)} aristas críticas):")
    output.append("(Aristas cuya eliminación desconecta el grafo)")
    output.append("")

    if puentes_sorted:
        for u, v in puentes_sorted:
            output.append(f"  • {u} ↔ {v}")
        output.append("")
    else:
        output.append("  Ninguno")
        output.append("")

    return "\n".join(output)


def format_matriz_distancias(matriz):
    """
    Formatea una matriz de distancias (para SIMULAR_CORTE con matriz completa).

    Args:
        matriz: Diccionario anidado {origen: {destino: distancia}}

    Returns:
        String formateado con la matriz de distancias
    """
    output = []
    output.append("=" * 60)
    output.append("MATRIZ DE DISTANCIAS")
    output.append("=" * 60)
    output.append("")

    # Obtener todos los nodos ordenados
    nodos = sorted(matriz.keys())

    # Mostrar en formato de tabla compacta
    for origen in nodos:
        output.append(f"{origen}:")
        destinos_con_dist = [(dest, matriz[origen][dest])
                            for dest in sorted(matriz[origen].keys())
                            if matriz[origen][dest] != float('inf') and dest != origen]

        if destinos_con_dist:
            lineas = []
            for dest, dist in destinos_con_dist:
                lineas.append(f"{dest}:{dist}")
            output.append(f"  {', '.join(lineas)}")
        else:
            output.append("  (sin conexiones)")

    output.append("")
    return "\n".join(output)
