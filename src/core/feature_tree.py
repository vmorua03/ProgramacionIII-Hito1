"""[HITO 1] Árbol Binario de Búsqueda (BST) para indexación por puntaje de Data Drift.

Sea n el número total de vectores, d el número de puntajes distintos
y h la altura del árbol. Cada puntaje distinto ocupa un nodo. Si el
orden de inserción produce árboles de altura logarítmica en promedio,
h es O(log d). En el peor caso, el árbol puede quedar como una cadena
y su altura es O(d). Los costos siguientes se expresan para un árbol
no vacío y suponen que comparar dos identificadores cuesta O(1).

insert
    Sea b la cantidad de registros que comparten el puntaje que se
    quiere insertar. El método baja por el árbol y puede revisar los
    b registros del nodo para encontrar el mismo feature_id.
    Su tiempo es O(log d + b) en promedio bajo la condición de altura
    indicada y O(n) en el peor caso. El recorrido iterativo usa O(1)
    de espacio auxiliar, sin contar el almacenamiento de los registros
    ni las reasignaciones internas de la lista al crecer.

search
    Sea k la cantidad de registros devueltos. El método baja por el
    árbol y copia las k referencias del nodo encontrado. Su tiempo
    es O(log d + k) en promedio y O(n) en el peor caso. Usa O(1)
    de espacio auxiliar en ambos casos, más O(k) para el resultado.

range_query
    Sea k la cantidad de registros devueltos. La poda evita visitar
    ramas que quedan fuera del intervalo. Su tiempo es O(log d + k)
    en promedio y O(n) en el peor caso. La pila de llamadas recursivas
    ocupa O(log d) en promedio y O(d) en el peor caso. La lista de
    resultados ocupa O(k) en ambos casos.

inorder_traversal
    Invoca range_query con el intervalo completo de 0.0 a 1.0.
    Recorre todos los nodos y devuelve los n vectores, así que tarda
    O(n) en promedio y en el peor caso. La pila de llamadas ocupa
    O(log d) en promedio y O(d) en el peor caso. La lista devuelta
    ocupa O(n) en ambos casos.

El almacenamiento del árbol es O(n + d), que se simplifica a O(n)
porque cada nodo contiene al menos un vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config.settings import FeatureVector


@dataclass
class BSTNode:
    """Agrupa los registros que comparten un mismo puntaje."""

    drift_score: float
    records: list[FeatureVector] = field(default_factory=list)
    left: Optional[BSTNode] = None
    right: Optional[BSTNode] = None


class DriftIndexBST:
    """Organiza vectores según su puntaje de drift."""

    def __init__(self) -> None:
        self.root: Optional[BSTNode] = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, record: FeatureVector) -> None:
        if not isinstance(record, FeatureVector):
            raise TypeError("El registro debe ser un FeatureVector.")

        score = record.drift_score
        if not 0.0 <= score <= 1.0:
            raise ValueError("El puntaje de drift debe estar entre 0.0 y 1.0.")

        if self.root is None:
            self.root = BSTNode(score, [record])
            self._size += 1
            return

        current = self.root
        while True:
            if score < current.drift_score:
                if current.left is None:
                    current.left = BSTNode(score, [record])
                    self._size += 1
                    return
                current = current.left
            elif score > current.drift_score:
                if current.right is None:
                    current.right = BSTNode(score, [record])
                    self._size += 1
                    return
                current = current.right
            else:
                for index, existing in enumerate(current.records):
                    if existing.feature_id == record.feature_id:
                        current.records[index] = record
                        return

                current.records.append(record)
                self._size += 1
                return

    def search(self, drift_score: float) -> list[FeatureVector]:
        if not 0.0 <= drift_score <= 1.0:
            raise ValueError("El puntaje de drift debe estar entre 0.0 y 1.0.")

        current = self.root
        while current is not None:
            if drift_score == current.drift_score:
                return current.records.copy()
            if drift_score < current.drift_score:
                current = current.left
            else:
                current = current.right

        return []

    def range_query(self, min_drift: float, max_drift: float) -> list[FeatureVector]:
        if not 0.0 <= min_drift <= max_drift <= 1.0:
            raise ValueError("El rango debe cumplir 0.0 <= mínimo <= máximo <= 1.0.")

        results: list[FeatureVector] = []

        def visit(node: Optional[BSTNode]) -> None:
            if node is None:
                return

            if node.drift_score > min_drift:
                visit(node.left)

            if min_drift <= node.drift_score <= max_drift:
                results.extend(node.records)

            if node.drift_score < max_drift:
                visit(node.right)

        visit(self.root)
        return results

    def inorder_traversal(self) -> list[FeatureVector]:
        return self.range_query(0.0, 1.0)

    def clear(self) -> None:
        self.root = None
        self._size = 0


BinarySearchTree = DriftIndexBST