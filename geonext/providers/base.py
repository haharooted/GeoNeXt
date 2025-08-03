from __future__ import annotations
from abc import ABC, abstractmethod

class Provider(ABC):
    """Common interface every provider must implement."""

    @abstractmethod
    def run(self, *, text: str) -> list[dict]:
        """Return GeoNeXt final JSON (already geocoded)"""
