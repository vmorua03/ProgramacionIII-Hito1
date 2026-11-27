"""[HITO 1] Tabla Hash con resolución de colisiones por encadenamiento (Separate Chaining).

Sea n el número de registros, m el número de buckets y alpha = n / m
el factor de carga. Sea L la longitud de la clave consultada. Para
analizar los recorridos de las cadenas, se supone que las claves
tienen longitud acotada y que compararlas cuesta O(1).

_hash
    Recorre los L caracteres de la clave para calcular el índice
    del bucket. Tarda O(L) en promedio y en el peor caso. El cálculo
    usa O(1) de espacio auxiliar, pero la validación con strip()
    puede crear una cadena temporal de hasta O(L).

put
    Calcula el índice y recorre la cadena para comprobar si el ID
    ya existe. Actualizar el valor o enlazar un nodo nuevo al inicio
    cuesta O(1) una vez terminada esa búsqueda.

search
    Calcula el índice y recorre la cadena hasta encontrar el ID
    o llegar al final. Devuelve la referencia al vector encontrado.

delete
    Calcula el índice y recorre la cadena guardando el nodo anterior.
    Una vez encontrado el ID, desvincula el nodo en O(1).

Con buena dispersión y claves de longitud acotada, put, search y
delete tardan O(1 + alpha) en promedio. Si alpha se mantiene acotado,
ese costo es O(1). En el peor caso, todos los registros caen en el
mismo bucket y cualquiera de las tres operaciones tarda O(n).

Los recorridos de put, search y delete usan O(1) de espacio auxiliar
en promedio y en el peor caso. Si se considera el costo de validar
claves de longitud variable, las tres operaciones pueden usar hasta
O(L) de espacio temporal al llamar a _hash. El almacenamiento total
de la tabla es O(m + n) por el arreglo de buckets y los nodos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config.settings import DEFAULT_HASH_CAPACITY, FeatureVector


@dataclass
class HashNode:
    """Nodo de la lista enlazada de un bucket."""

    key: str
    value: FeatureVector
    next: Optional[HashNode] = None


class FeatureHashTable:
    """Indexa vectores por feature_id mediante listas enlazadas."""

    def __init__(self, capacity: int = DEFAULT_HASH_CAPACITY) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("La capacidad de la tabla hash debe ser un entero positivo.")

        self.capacity = capacity
        self.buckets: list[Optional[HashNode]] = [None] * capacity
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def _hash(self, key: str) -> int:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("La clave debe ser una cadena con al menos un carácter no blanco.")

        index = 0
        for character in key:
            index = (index * 31 + ord(character)) % self.capacity
        return index

    def put(self, record: FeatureVector) -> None:
        if not isinstance(record, FeatureVector):
            raise TypeError("El registro debe ser un FeatureVector.")

        key = record.feature_id
        index = self._hash(key)
        current = self.buckets[index]

        while current is not None:
            if current.key == key:
                current.value = record
                return
            current = current.next

        self.buckets[index] = HashNode(key, record, self.buckets[index])
        self._size += 1

    def search(self, feature_id: str) -> Optional[FeatureVector]:
        index = self._hash(feature_id)
        current = self.buckets[index]

        while current is not None:
            if current.key == feature_id:
                return current.value
            current = current.next

        return None

    def delete(self, feature_id: str) -> bool:
        index = self._hash(feature_id)
        previous = None
        current = self.buckets[index]

        while current is not None:
            if current.key == feature_id:
                if previous is None:
                    self.buckets[index] = current.next
                else:
                    previous.next = current.next

                self._size -= 1
                return True

            previous = current
            current = current.next

        return False

    def load_factor(self) -> float:
        return self._size / self.capacity

    def clear(self) -> None:
        self.buckets = [None] * self.capacity
        self._size = 0


CustomHashTable = FeatureHashTable