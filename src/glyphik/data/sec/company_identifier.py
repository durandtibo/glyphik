r"""Provide the :class:`CompanyIdentifier` dataclass used to identify a
company by its ticker symbol and SEC Central Index Key (CIK)."""

from __future__ import annotations

__all__ = ["CompanyIdentifier", "CompanyIdentifierHasher"]

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from coola.hashing import BaseHasher, HasherRegistry, get_default_registry, hash_string

from glyphik.data.sec.cik import fetch_ticker_from_cik
from glyphik.data.sec.ticker import fetch_cik_from_ticker

if TYPE_CHECKING:
    import edgar


@dataclass(frozen=True)
class CompanyIdentifier:
    """A single company identifier.

    Args:
        cik: The SEC Central Index Key (CIK).
        ticker: The stock ticker symbol (e.g. ``"AAPL"``).
    """

    cik: int
    ticker: str

    @classmethod
    def from_cik(cls, cik: int) -> CompanyIdentifier:
        """Build a :class:`CompanyIdentifier` by looking up the ticker
        for a CIK.

        Args:
            cik: The SEC Central Index Key (CIK) to look up.

        Returns:
            The resolved company identifier.

        Raises:
            ValueError: If no ticker can be found for ``cik``.
        """
        ticker = fetch_ticker_from_cik(cik)
        if ticker is None:
            msg = f"Cannot find ticker for CIK {cik}"
            raise ValueError(msg)
        return cls(cik=cik, ticker=ticker)

    @classmethod
    def from_ticker(cls, ticker: str) -> CompanyIdentifier:
        """Build a :class:`CompanyIdentifier` by looking up the CIK for
        a ticker.

        Args:
            ticker: The stock ticker symbol to look up.

        Returns:
            The resolved company identifier.

        Raises:
            ValueError: If no CIK can be found for ``ticker``.
        """
        cik = fetch_cik_from_ticker(ticker)
        if cik is None:
            msg = f"Cannot find CIK for ticker {ticker!r}"
            raise ValueError(msg)
        return cls(cik=cik, ticker=ticker)

    @classmethod
    def from_edgar_company(cls, company: edgar.Company) -> CompanyIdentifier:
        """Build a :class:`CompanyIdentifier` from an ``edgar.Company``
        object.

        If the ``edgar.Company`` does not expose a ticker directly, it is
        resolved from its CIK via :meth:`from_cik`.

        Args:
            company: The ``edgar.Company`` to build an identifier from.

        Returns:
            The resolved company identifier.

        Raises:
            ValueError: If ``company`` has no ticker and no ticker can be
                found for its CIK.
        """
        ticker = company.get_ticker()
        if ticker is None:
            return cls.from_cik(company.cik)
        return cls(cik=company.cik, ticker=ticker)

    def content_hash(self, length: int = 64) -> str:
        """Compute a stable hash of this identifier's content.

        The identifier is first serialized to a canonical JSON string
        (its fields sorted by key, so field declaration order does not
        affect the result), then hashed with BLAKE2b via
        :func:`coola.hashing.hash_string`.

        This is distinct from the built-in ``hash()`` / ``__hash__``
        that :func:`dataclasses.dataclass` generates automatically for
        frozen dataclasses: Python's string hashing is randomized per
        process (via ``PYTHONHASHSEED``) unless explicitly disabled, so
        ``hash(identifier)`` can differ between runs and must not be
        used for caching to disk, cache keys shared across processes,
        or persisted identifiers. ``content_hash`` is deterministic
        across processes, interpreter restarts, and machines, and is
        safe to use for those purposes.

        Args:
            length: The desired length of the returned hex string. Must
                be an even number between 2 and 128 inclusive, since
                each byte of the BLAKE2b digest encodes as two hex
                characters. Defaults to 64 (32-byte digest).

        Returns:
            A lowercase hexadecimal string of exactly ``length``
                characters, uniquely determined by ``cik`` and
                ``ticker``.

        Raises:
            ValueError: If ``length`` is not an even number between 2
                and 128 (propagated from
                :func:`coola.hashing.hash_string`).

        Example:
            ```pycon
            >>> from glyphik.data.sec import CompanyIdentifier
            >>> identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
            >>> len(identifier.content_hash())
            64

            ```
        """
        return hash_string(json.dumps(asdict(self), sort_keys=True), length=length)


class CompanyIdentifierHasher(BaseHasher[CompanyIdentifier]):
    r"""Hasher for :class:`~glyphik.data.sec.CompanyIdentifier` objects.

    This hasher delegates to
    :meth:`~glyphik.data.sec.CompanyIdentifier.content_hash`, which
    computes a hash from the identifier's ``cik`` and ``ticker``
    fields, so two identifiers with equal content produce the same
    hash regardless of object identity.

    Example:
        ```pycon
        >>> from glyphik.data.sec import CompanyIdentifier
        >>> from glyphik.data.sec.company_identifier import CompanyIdentifierHasher
        >>> from coola.hashing import HasherRegistry
        >>> registry = HasherRegistry()
        >>> hasher = CompanyIdentifierHasher()
        >>> hasher
        CompanyIdentifierHasher()
        >>> identifier = CompanyIdentifier(cik=320193, ticker="AAPL")
        >>> len(hasher.hash(identifier, registry=registry))
        64

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def hash(
        self,
        data: CompanyIdentifier,
        registry: HasherRegistry,  # noqa: ARG002
        length: int = 64,
    ) -> str:
        return data.content_hash(length=length)


get_default_registry().register(CompanyIdentifier, CompanyIdentifierHasher(), exist_ok=True)
