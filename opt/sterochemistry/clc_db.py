#!/usr/bin/env -S uv run python
"""
CLC-DB bulk downloader

Downloads all molecule SDF files and merges molecule properties into a single CSV.

Strategy:
- Page the public search API used by the site ("/search/molecules") to list molecules.
- For each molecule, fetch its SDF from the static path used by the web UI.
- Optionally iterate by category if the categories endpoint is available; otherwise, fetch all.

Outputs (under --out-dir):
- sdf/            (all .sdf files; one per CAS ID)
- molecules.csv   (merged properties in the same schema as the website's CSV)

Note: The site generates ZIPs client-side in the browser. This script mirrors that behavior
without trying to click buttons, using the same public endpoints visible in the JS.
"""

import argparse
import concurrent.futures as futures
import csv
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

# Public base URLs (axios base is not exposed directly; try a few sensible options)
API_BASE_CANDIDATES = [
    # Observed public API base
    "https://compbio.sjtu.edu.cn/api",
    # Fallback guesses based on site paths
    "https://compbio.sjtu.edu.cn/services/clc-db/api",
    "https://compbio.sjtu.edu.cn/services/clc-db",
    "https://compbio.sjtu.edu.cn/services/clc-db/api/v1",
]

# Static path for SDFs as used by the client JS
SDF_BASE = "https://compbio.sjtu.edu.cn/services/clc-db/static/all_sdfs"


@dataclass
class Molecule:
    data: dict[str, Any]

    @property
    def cas_id(self) -> str:
        return str(self.data.get("cas_id", "")).strip()

    @property
    def name(self) -> str:
        return str(self.data.get("name", "")).strip()

    @property
    def categories(self) -> list[str]:
        cats = self.data.get("category") or []
        if isinstance(cats, list):
            return [str(c.get("name", "")).strip() for c in cats if isinstance(c, dict)]
        return []

    def csv_row(self) -> dict[str, Any]:
        """Return a row dict with the same columns the website exports.

        Column order mirrors the client code seen in the website bundle.
        """
        d = self.data

        # Join helper
        def _join(items: list[dict[str, Any]] | None) -> str:
            if not items:
                return ""
            return ", ".join(
                str(it.get("name", "")).strip() for it in items if isinstance(it, dict)
            )

        def _fmt(x: Any, ndigits: int) -> str:
            try:
                return f"{float(x):.{ndigits}f}"
            except Exception:
                return ""

        def _int(x: Any) -> str:
            try:
                return str(int(x))
            except Exception:
                return ""

        return {
            "Name": d.get("name", ""),
            "CAS ID": d.get("cas_id", ""),
            "PubChem CID": d.get("pubchem_cid", ""),
            "Category": _join(d.get("category")),
            "URL": d.get("url", ""),
            "PubChem URL": d.get("pubchem_url", ""),
            "SMILES": d.get("smiles", ""),
            "Chirality": _join(d.get("chirality")),
            "Description": d.get("description", ""),
            "SMILES IUPAC": d.get("smiles_iupac", ""),
            "Molecule Formula": d.get("molecule_formula", ""),
            "Molecular Weight": _fmt(d.get("molecular_weight"), 3),
            "Heavy Atom Count": _int(d.get("heavy_atom_count")),
            "Ring Count": _int(d.get("ring_count")),
            "Hydrogen Bond Acceptor Count": _int(d.get("hydrogen_bond_acceptor_count")),
            "Hydrogen Bond Donor Count": _int(d.get("hydrogen_bond_donor_count")),
            "Rotatable Bond Count": _int(d.get("rotatable_bond_count")),
            "Zero-point correction": _fmt(d.get("zero_point_correction"), 6),
            "Thermal correction to Energy": _fmt(d.get("thermal_correction_energy"), 6),
            "Thermal correction to Enthalpy": _fmt(
                d.get("thermal_correction_enthalpy"), 6
            ),
            "Thermal correction to Gibbs Free Energy": _fmt(
                d.get("thermal_correction_gibbs"), 6
            ),
            "Sum of electronic and zero-point Energies": _fmt(
                d.get("sum_electronic_zero_point"), 6
            ),
            "Sum of electronic and thermal Energies": _fmt(
                d.get("sum_electronic_thermal_energy"), 6
            ),
            "Sum of electronic and thermal Enthalpies": _fmt(
                d.get("sum_electronic_thermal_enthalpy"), 6
            ),
            "Sum of electronic and thermal Free Energies": _fmt(
                d.get("sum_electronic_thermal_free_energy"), 6
            ),
            "HOMO Energy (eV)": _fmt(d.get("homo_energy"), 6),
            "LUMO Energy (eV)": _fmt(d.get("lumo_energy"), 6),
            "HOMO-LUMO Gap (eV)": _fmt(d.get("homo_lumo_gap"), 6),
        }


def _new_session(timeout: int = 30) -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=3, pool_connections=16, pool_maxsize=16)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(
        {
            "User-Agent": "clc-db-downloader/1.0 (+https://compbio.sjtu.edu.cn/services/clc-db)",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    # store a default timeout on the session for convenience
    s.request = _timeout_wrapper(s.request, timeout)
    return s


def _timeout_wrapper(request_fn, timeout_default: int):
    def wrapped(method, url, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout_default
        return request_fn(method, url, **kwargs)

    return wrapped


def _try_api_base(session: requests.Session) -> str | None:
    for base in API_BASE_CANDIDATES:
        logging.debug("Probing API base: %s", base)
        try:
            r = session.get(f"{base}/categories/")
            if r.ok and r.headers.get("content-type", "").startswith(
                "application/json"
            ):
                _ = r.json()
                logging.info("Discovered API base: %s", base)
                return base
        except Exception:
            logging.debug("Probe failed for %s", base, exc_info=True)
            continue
    return None


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    reraise=True,
    retry=retry_if_exception_type((requests.RequestException,)),
)
def _get_json(
    session: requests.Session, url: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    r = session.get(url, params=params)
    r.raise_for_status()
    return r.json()


def fetch_categories(session: requests.Session, api_base: str | None) -> list[str]:
    if not api_base:
        return []
    try:
        data = _get_json(session, f"{api_base}/categories/")
    except Exception:
        logging.warning("Fetching categories failed", exc_info=True)
        return []
    # Expecting list of {id, name}
    names = []
    if isinstance(data, list):
        names = [str(x.get("name", "")).strip() for x in data if isinstance(x, dict)]
    elif isinstance(data, dict) and "results" in data:
        names = [
            str(x.get("name", "")).strip()
            for x in data.get("results", [])
            if isinstance(x, dict)
        ]
    names = [n for n in names if n]
    logging.info("Fetched %d categories", len(names))
    return names


def iter_molecules(
    session: requests.Session,
    api_base: str | None,
    category: str | None = None,
    page_size: int = 30,
) -> Iterable[Molecule]:
    if not api_base:
        raise RuntimeError(
            "API base not discovered; cannot enumerate molecules reliably."
        )

    page = 1
    while True:
        params = {"page": page, "page_size": page_size}
        if category:
            params["category"] = category
        url = f"{api_base}/search/molecules"
        data = _get_json(session, url, params=params)
        results = data.get("results", []) if isinstance(data, dict) else []
        logging.info(
            "Fetched page %s (category=%s): %d results",
            page,
            category if category else "ALL",
            len(results),
        )
        for item in results:
            if isinstance(item, dict):
                yield Molecule(item)
        next_url = data.get("next") if isinstance(data, dict) else None
        if not next_url:
            logging.info("No next page; finished pagination for %s", category or "ALL")
            break
        page += 1


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    reraise=True,
    retry=retry_if_exception_type((requests.RequestException,)),
)
def download_sdf(
    session: requests.Session, cas_id: str, out_path: str
) -> tuple[str, bool, str | None]:
    url = f"{SDF_BASE}/{cas_id}.sdf"
    r = session.get(url)
    if r.status_code == 404:
        logging.warning("SDF not found (404) for CAS %s", cas_id)
        return (cas_id, False, "404")
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    logging.debug("Saved SDF %s -> %s", cas_id, out_path)
    return (cas_id, True, None)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(rows: list[dict[str, Any]], out_csv: str) -> None:
    if not rows:
        # write header-only CSV
        cols = Molecule({}).csv_row().keys()
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(cols))
            writer.writeheader()
        return

    # Use pandas for convenience and de-duplication on CAS ID
    df = pd.DataFrame(rows)
    # Drop duplicates by CAS ID keeping first occurrence
    if "CAS ID" in df.columns:
        df = df.drop_duplicates(subset=["CAS ID"], keep="first")
    df.to_csv(out_csv, index=False)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Download all molecules from CLC-DB and merge outputs."
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join("opt", "sterochemistry", "data", "clc_db"),
        help="Output directory for SDFs and CSV (default: opt/sterochemistry/data/clc_db)",
    )
    p.add_argument(
        "--by-category",
        action="store_true",
        help="Iterate through categories explicitly",
    )
    p.add_argument(
        "--workers", type=int, default=8, help="Concurrent download workers for SDFs"
    )
    p.add_argument("--page-size", type=int, default=30, help="API page size to use")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level (default: INFO)",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    out_dir = os.path.abspath(args.out_dir)
    sdf_dir = os.path.join(out_dir, "sdf")
    ensure_dir(sdf_dir)
    logging.info("Output directory: %s", out_dir)
    logging.info("SDF directory: %s", sdf_dir)

    session = _new_session()
    api_base = _try_api_base(session)
    if not api_base:
        logging.error("Could not discover API base endpoint; aborting.")
        return 2

    categories: list[str] = []
    if args.by_category:
        categories = fetch_categories(session, api_base)
        if not categories:
            logging.warning(
                "Categories endpoint not available; falling back to all molecules."
            )

    rows: list[dict[str, Any]] = []
    seen_cas: set[str] = set()

    def process_molecule(mol: Molecule) -> tuple[str, bool, str | None] | None:
        cas = mol.cas_id
        if not cas:
            return None
        out_path = os.path.join(sdf_dir, f"{cas}.sdf")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return (cas, True, None)
        try:
            return download_sdf(session, cas, out_path)
        except Exception as e:
            return (cas, False, str(e))

    # Iterate molecules (all or by category)
    if categories:
        iterables: list[tuple[str, Iterable[Molecule]]] = []
        for cat in categories:
            iterables.append(
                (
                    cat,
                    iter_molecules(
                        session, api_base, category=cat, page_size=args.page_size
                    ),
                )
            )
    else:
        iterables = [
            (
                "ALL",
                iter_molecules(
                    session, api_base, category=None, page_size=args.page_size
                ),
            )
        ]

    for label, it in iterables:
        logging.info("Collecting molecules for: %s", label)
        batch: list[Molecule] = list(it)
        logging.info("Found %d molecules for %s", len(batch), label)
        # merge rows
        for mol in batch:
            if mol.cas_id and mol.cas_id not in seen_cas:
                rows.append(mol.csv_row())
                seen_cas.add(mol.cas_id)

        # download SDFs concurrently
        results: list[tuple[str, bool, str | None]] = []
        with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for res in tqdm(
                ex.map(process_molecule, batch), total=len(batch), desc=f"SDF {label}"
            ):
                if res is not None:
                    results.append(res)
        total = len(results)
        ok = sum(1 for _, success, _ in results if success)
        missing = [cas for cas, ok_, _ in results if not ok_]
        logging.info(
            "Downloads finished for %s: %d ok, %d missing (of %d)",
            label,
            ok,
            len(missing),
            total,
        )
        if missing:
            logging.warning("Missing first few SDFs for %s: %s", label, missing[:5])

    # Write merged CSV
    out_csv = os.path.join(out_dir, "molecules.csv")
    write_csv(rows, out_csv)
    logging.info("Saved CSV: %s", out_csv)
    logging.info("SDF folder: %s", sdf_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
