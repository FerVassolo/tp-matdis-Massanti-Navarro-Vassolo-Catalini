"""
Graph analysis module for Buenos Aires city networks.
Students must implement all functions marked with TODO.
"""

from src.output import (
    format_componentes_conexos,
    format_orden_fallos,
    format_camino_minimo,
    format_simulacion_corte,
    format_ruta_recoleccion,
    format_plantas_asignadas,
    format_puentes_y_articulaciones,
)

# -----------------------------
# Graph loading
# -----------------------------

def load_graph(path):
    """
    Load a simple graph from a file.

    Args:
        path: File path

    Returns:
        Adjacency dictionary {node: [neighbors]}
    """
    adjacency = {}
    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            if u not in adjacency:
                adjacency[u] = []
            if v not in adjacency:
                adjacency[v] = []
            if v not in adjacency[u]:
                adjacency[u].append(v)
            if u not in adjacency[v]:
                adjacency[v].append(u)
    # Keep deterministic traversal: sort neighbor lists
    for node in adjacency:
        adjacency[node].sort()
    return adjacency


def load_weighted_graph(path):
    """
    Load a weighted graph from a file.

    Args:
        path: File path

    Returns:
        Adjacency dictionary {node: [(neighbor, weight), ...]}
    """
    adjacency = {}
    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            u, v = parts[0], parts[1]
            try:
                weight = float(parts[2])
            except ValueError:
                # Skip malformed lines
                continue
            if u not in adjacency:
                adjacency[u] = []
            if v not in adjacency:
                adjacency[v] = []
            # Avoid duplicate undirected edge insertions
            if not any(n == v for n, _ in adjacency[u]):
                adjacency[u].append((v, weight))
            if not any(n == u for n, _ in adjacency[v]):
                adjacency[v].append((u, weight))
    # Deterministic neighbor order
    for node in adjacency:
        adjacency[node].sort(key=lambda t: (t[0]))
    return adjacency


# -----------------------------
# Algorithms
# -----------------------------

def _connected_components(graph):
    visited = set()
    components = []
    for node in sorted(graph.keys()):
        if node in visited:
            continue
        stack = [node]
        component = []
        visited.add(node)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _degree_order(graph):
    # Returns list of (node, degree)
    return [(node, len(graph.get(node, []))) for node in graph.keys()]


def _dijkstra_path(weighted_graph, origin, destination, blocked_nodes=None):
    import heapq

    if origin not in weighted_graph or destination not in weighted_graph:
        return float("inf"), []

    blocked = set(blocked_nodes or [])
    if origin in blocked or destination in blocked:
        return float("inf"), []

    # Build a quick neighbor function to respect blocked nodes
    def neighbors(u):
        for v, w in weighted_graph.get(u, []):
            if v not in blocked:
                yield v, w

    distances = {node: float("inf") for node in weighted_graph.keys()}
    previous = {}
    distances[origin] = 0.0
    heap = [(0.0, origin)]
    visited = set()

    while heap:
        dist_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == destination:
            break
        for v, w in neighbors(u):
            if v in visited:
                continue
            alt = dist_u + float(w)
            if alt < distances[v]:
                distances[v] = alt
                previous[v] = u
                heapq.heappush(heap, (alt, v))

    if distances[destination] == float("inf"):
        return float("inf"), []

    # Reconstruct path
    path = []
    cur = destination
    while cur is not None:
        path.append(cur)
        cur = previous.get(cur)
        if cur == origin:
            path.append(cur)
            break
    path.reverse()
    return distances[destination], path


def _bfs_all_pairs_unweighted(graph, origins):
    # Returns dict {origin: {node: dist}} for provided origins
    from collections import deque

    dists = {}
    for start in origins:
        dist = {node: float("inf") for node in graph.keys()}
        if start in graph:
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in graph.get(u, []):
                    if dist[v] == float("inf"):
                        dist[v] = dist[u] + 1
                        q.append(v)
        dists[start] = dist
    return dists


def _assign_plants(water_graph, plants):
    # Ensure plants exist in the graph as isolated nodes if missing
    for p in plants:
        if p not in water_graph:
            water_graph[p] = []
    # Compute shortest unweighted distances from each plant
    dists = _bfs_all_pairs_unweighted(water_graph, plants)
    assignment = {}
    for barrio in water_graph.keys():
        # Choose plant with min distance; tie-break lexicographically by plant name
        best_plant = None
        best_dist = float("inf")
        for p in plants:
            dist = dists[p].get(barrio, float("inf"))
            if dist < best_dist or (dist == best_dist and (best_plant is None or p < best_plant)):
                best_dist = dist
                best_plant = p
        assignment[barrio] = best_plant
    return assignment


def _tarjan_articulation_bridges(graph):
    # Tarjan algorithm for articulation points and bridges in an undirected graph
    index = {}
    low = {}
    time = 0
    parent = {}
    articulations = set()
    bridges = []

    def dfs(u):
        nonlocal time
        time += 1
        index[u] = time
        low[u] = time
        children = 0
        for v in sorted(graph.get(u, [])):
            if v not in index:
                parent[v] = u
                children += 1
                dfs(v)
                low[u] = min(low[u], low[v])
                # Articulation conditions
                if u not in parent and children > 1:
                    articulations.add(u)
                if u in parent and low[v] >= index[u]:
                    articulations.add(u)
                # Bridge condition
                if low[v] > index[u]:
                    bridges.append(tuple(sorted((u, v))))
            elif parent.get(u) != v:
                low[u] = min(low[u], index[v])

    for node in sorted(graph.keys()):
        if node not in index:
            dfs(node)
    # Deduplicate bridges
    bridges = sorted(set(bridges))
    return sorted(articulations), bridges


def _extract_nodes_in_braces(token):
    # token like "{A,B,C}" or "{Recoleta}"
    token = token.strip()
    if token.startswith("{") and token.endswith("}"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        # allow commas and/or spaces
        parts = [p for p in inner.replace(" ", "").split(",") if p]
        return parts
    return []


def _all_nodes_from_weighted_graph(g):
    # Return sorted list of nodes from weighted graph adjacency
    return sorted(g.keys())


def _shortest_path_nodes(weighted_graph, start, end):
    # Convenience wrapper to return only nodes of path
    dist, path = _dijkstra_path(weighted_graph, start, end)
    return dist, path


def _build_garbage_collection_route(weighted_graph):
    # Heuristic route: visit all nodes by repeatedly going to nearest unvisited via Dijkstra over the road graph.
    # Deterministic start: lexicographically smallest node
    unvisited = set(_all_nodes_from_weighted_graph(weighted_graph))
    if not unvisited:
        return []
    current = sorted(unvisited)[0]
    route = [current]
    unvisited.remove(current)
    while unvisited:
        # Find nearest unvisited by Dijkstra
        best_target = None
        best_dist = float("inf")
        best_path = []
        for candidate in sorted(unvisited):
            dist, path = _dijkstra_path(weighted_graph, current, candidate)
            if dist < best_dist:
                best_dist = dist
                best_target = candidate
                best_path = path
        if not best_path:
            # Disconnected case: start a new component from lexicographically smallest remaining
            current = sorted(unvisited)[0]
            route.append(current)
            unvisited.remove(current)
            continue
        # Append path excluding the first node (already at current)
        for node in best_path[1:]:
            route.append(node)
        current = best_target
        unvisited.remove(current)
    return route


def process_queries(queries_file, output_file, electric_graph, road_graph, water_graph):
    """
    Process queries from file and generate output.

    Args:
        queries_file: Path to queries file
        output_file: Path to output file
        electric_graph: Electric network graph
        road_graph: Road network graph
        water_graph: Water network graph
    """
    outputs = []

    with open(queries_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            command = parts[0].upper()

            if command == "COMPONENTES_CONEXOS":
                # Expected: COMPONENTES_CONEXOS ELECTRICA
                # Only ELECTRICA is supported, ignore the token
                comps = _connected_components(electric_graph)
                outputs.append(format_componentes_conexos(comps))

            elif command == "ORDEN_FALLOS":
                # Expected: ORDEN_FALLOS ELECTRICA
                grados = _degree_order(electric_graph)
                outputs.append(format_orden_fallos(grados))

            elif command == "CAMINO_MINIMO":
                # Expected: CAMINO_MINIMO Origen Destino
                if len(parts) < 3:
                    continue
                origen = parts[1]
                destino = parts[2]
                distancia, camino = _shortest_path_nodes(road_graph, origen, destino)
                outputs.append(format_camino_minimo(origen, destino, distancia, camino))

            elif command in ("CAMINO_MINIMO_SIMULAR_CORTE", "SIMULAR_CORTE"):
                # Expected: CAMINO_MINIMO_SIMULAR_CORTE {A,B,...} Origen Destino
                # or SIMULAR_CORTE {..} Origen Destino
                if len(parts) < 4:
                    continue
                cortes = _extract_nodes_in_braces(parts[1])
                origen = parts[2]
                destino = parts[3]
                distancia, camino = _dijkstra_path(road_graph, origen, destino, blocked_nodes=set(cortes))
                outputs.append(
                    format_simulacion_corte(origen, destino, cortes, distancia, camino)
                )

            elif command == "CAMINO_RECOLECCION_BASURA":
                # Build a city-wide collection route over road graph
                camino = _build_garbage_collection_route(road_graph)
                if not camino:
                    camino = []
                outputs.append(format_ruta_recoleccion(camino))

            elif command == "PLANTAS_ASIGNADAS":
                # Expected: PLANTAS_ASIGNADAS P1 P2 [P3 ...]
                if len(parts) < 2:
                    # No plants provided; still produce output with empty
                    plantas = []
                else:
                    plantas = parts[1:]
                # Ensure plants exist in water graph
                for p in plantas:
                    if p not in water_graph:
                        water_graph[p] = []
                asignaciones = _assign_plants(water_graph, plantas)
                outputs.append(format_plantas_asignadas(plantas, asignaciones))

            elif command == "PUENTES_Y_ARTICULACIONES":
                articulaciones, puentes = _tarjan_articulation_bridges(water_graph)
                outputs.append(format_puentes_y_articulaciones(articulaciones, puentes))

            else:
                # Unknown command: ignore or log; choose to ignore silently
                continue

    # Write all outputs to the output file
    with open(output_file, "w") as f:
        f.write("\n".join(outputs))
    return output_file
