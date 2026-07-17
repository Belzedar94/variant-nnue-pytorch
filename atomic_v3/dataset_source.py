"""Explicit, mutually exclusive AtomicNNUEV3 dataset-source selection.

There is deliberately no content sniffing.  Callers select either the strict
publication campaign path or the non-publication bootstrap path, and the
unselected contract is never attempted as a fallback.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union

from . import bootstrap_dataset, dataset
from .dataset import DatasetContractError, _require_path, _require_sha256


@dataclass(frozen=True)
class PublicationDatasetSource:
    campaign_path: Path
    receipt_path: Path
    expected_receipt_sha256: str


@dataclass(frozen=True)
class BootstrapDatasetSource:
    receipt_path: Path
    expected_receipt_sha256: str


def publication_source(
    campaign_path: Union[str, Path],
    receipt_path: Union[str, Path],
    expected_receipt_sha256: str,
) -> PublicationDatasetSource:
    return PublicationDatasetSource(
        _require_path("campaign_path", campaign_path),
        _require_path("publication_receipt_path", receipt_path),
        _require_sha256("expected_publication_receipt_sha256", expected_receipt_sha256),
    )


def bootstrap_source(
    receipt_path: Union[str, Path], expected_receipt_sha256: str
) -> BootstrapDatasetSource:
    return BootstrapDatasetSource(
        _require_path("bootstrap_receipt_path", receipt_path),
        _require_sha256("expected_bootstrap_receipt_sha256", expected_receipt_sha256),
    )


def create_selected_role_provider(
    role: Literal["train", "validation"],
    *,
    provider_factory: Callable[..., Any],
    publication: Optional[PublicationDatasetSource] = None,
    bootstrap: Optional[BootstrapDatasetSource] = None,
    **provider_options: Any,
) -> Any:
    """Invoke exactly one explicitly selected trust contract."""

    if (publication is None) == (bootstrap is None):
        raise DatasetContractError(
            "exactly one of publication or bootstrap dataset source is required"
        )
    if publication is not None:
        if not isinstance(publication, PublicationDatasetSource):
            raise TypeError("publication must be a PublicationDatasetSource")
        return dataset.create_role_provider(
            publication.campaign_path,
            publication.receipt_path,
            publication.expected_receipt_sha256,
            role,
            provider_factory=provider_factory,
            **provider_options,
        )
    if not isinstance(bootstrap, BootstrapDatasetSource):
        raise TypeError("bootstrap must be a BootstrapDatasetSource")
    return bootstrap_dataset.create_bootstrap_role_provider(
        bootstrap.receipt_path,
        bootstrap.expected_receipt_sha256,
        role,
        provider_factory=provider_factory,
        **provider_options,
    )


def add_dataset_source_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the frozen mutually-exclusive CLI source contract to ``parser``."""

    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--publication-source",
        nargs=3,
        metavar=("CAMPAIGN", "RECEIPT", "RECEIPT_SHA256"),
        help="select the publication campaign contract explicitly",
    )
    sources.add_argument(
        "--bootstrap-source",
        nargs=2,
        metavar=("RECEIPT", "RECEIPT_SHA256"),
        help="select the non-publication bootstrap contract explicitly",
    )


def dataset_source_from_namespace(
    arguments: argparse.Namespace,
) -> Union[PublicationDatasetSource, BootstrapDatasetSource]:
    publication_values = getattr(arguments, "publication_source", None)
    bootstrap_values = getattr(arguments, "bootstrap_source", None)
    if (publication_values is None) == (bootstrap_values is None):
        raise DatasetContractError(
            "exactly one CLI publication/bootstrap dataset source is required"
        )
    if publication_values is not None:
        if not isinstance(publication_values, (list, tuple)) or len(publication_values) != 3:
            raise DatasetContractError("--publication-source requires exactly three values")
        return publication_source(*publication_values)
    if not isinstance(bootstrap_values, (list, tuple)) or len(bootstrap_values) != 2:
        raise DatasetContractError("--bootstrap-source requires exactly two values")
    return bootstrap_source(*bootstrap_values)


def parse_dataset_source_args(
    argv: Optional[Sequence[str]] = None,
) -> Union[PublicationDatasetSource, BootstrapDatasetSource]:
    parser = argparse.ArgumentParser(description="Select one AtomicNNUEV3 dataset contract")
    add_dataset_source_arguments(parser)
    return dataset_source_from_namespace(parser.parse_args(argv))


__all__ = [
    "BootstrapDatasetSource",
    "PublicationDatasetSource",
    "add_dataset_source_arguments",
    "bootstrap_source",
    "create_selected_role_provider",
    "dataset_source_from_namespace",
    "parse_dataset_source_args",
    "publication_source",
]
