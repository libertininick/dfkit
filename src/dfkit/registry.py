"""DataFrame registry grouping context and references."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

import polars as pl

from dfkit.context import DataFrameContext
from dfkit.identifier import DataFrameId
from dfkit.models import DataFrameReference


@dataclass
class DataFrameRegistry:
    """Groups a DataFrameContext and its associated references.

    This container holds the two data structures that are always
    modified together during DataFrame registration and state restoration.

    Attributes:
        context (DataFrameContext): The SQL-capable DataFrame registry.
    """

    context: DataFrameContext = field(default_factory=DataFrameContext)
    _references: dict[DataFrameId, DataFrameReference] = field(default_factory=dict)

    @property
    def references(self) -> MappingProxyType[DataFrameId, DataFrameReference]:
        """Read-only view of registered references."""
        return MappingProxyType(self._references)

    def register(self, reference: DataFrameReference, dataframe: pl.DataFrame | pl.LazyFrame) -> None:
        """Register a dataframe with its reference metadata.

        Updates both the SQL context and references dict together,
        keeping them in sync.

        Args:
            reference (DataFrameReference): The reference metadata for the dataframe.
            dataframe (pl.DataFrame | pl.LazyFrame): The dataframe to register.
        """
        self.context.register(reference.id, dataframe)
        self._references[reference.id] = reference

    def unregister(self, dataframe_id: DataFrameId) -> None:
        """Unregister a dataframe from both context and references.

        Args:
            dataframe_id (DataFrameId): The ID of the dataframe to unregister.

        Raises:
            KeyError: If the dataframe_id is not present in both stores.
        """
        if dataframe_id not in self.context or dataframe_id not in self._references:
            raise KeyError(dataframe_id)
        self.context.unregister(dataframe_id)
        del self._references[dataframe_id]
