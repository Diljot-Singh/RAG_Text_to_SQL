# src/schema/join_path.py
from collections import deque
from typing import List, Dict, Any, Optional

def shortest_path(graph: Dict[str, List[str]], start: str, goal: str) -> Optional[List[str]]:
    if start == goal:
        return [start]
    q = deque([(start, [start])])
    seen = {start}
    while q:
        node, path = q.popleft()
        for nxt in graph.get(node, []):
            if nxt in seen:
                continue
            if nxt == goal:
                return path + [nxt]
            seen.add(nxt)
            q.append((nxt, path + [nxt]))
    return None

def join_clause_for_edge(a: str, b: str, join_rules: List[Dict[str, Any]]) -> Optional[str]:
    # find a rule in either direction
    for r in join_rules:
        if r.get("left_table") == a and r.get("right_table") == b:
            return f"JOIN {b} ON {a}.{r['left_key']} = {b}.{r['right_key']}"
        if r.get("left_table") == b and r.get("right_table") == a:
            return f"JOIN {b} ON {b}.{r['left_key']} = {a}.{r['right_key']}"
    return None

def build_join_clauses(path: List[str], join_rules: List[Dict[str, Any]]) -> List[str]:
    clauses = []
    for i in range(len(path)-1):
        a, b = path[i], path[i+1]
        clause = join_clause_for_edge(a, b, join_rules)
        if clause:
            clauses.append(clause)
    return clauses
