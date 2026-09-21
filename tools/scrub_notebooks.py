#!/usr/bin/env python3
"""Nettoie les notebooks avant de les versionner.

- anonymise les chemins personnels présents dans les sorties (``C:\\Users\\alice\\...`` -> ``C:\\Users\\<user>\\...``,
  ``/home/alice/...`` -> ``/home/<user>/...``) ;
- option ``--strip-outputs`` : supprime toutes les sorties et les compteurs d'exécution ;
- option ``--check`` : ne modifie rien, échoue (code 1) s'il reste un chemin personnel ou un fichier > 5 Mo.

Utilisation (depuis la racine du dépôt) :

    python tools/scrub_notebooks.py                 # nettoie notebooks/*.ipynb
    python tools/scrub_notebooks.py --check         # vérifie seulement
    python tools/scrub_notebooks.py --strip-outputs # version sans sorties
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

USER_PATH = re.compile(r"(?i)((?:[a-z]:)?[\\/]+(?:users|home)[\\/]+)(?!<user>)([^\\/\s\"'<>|:*?]+)")
MAX_SIZE = 5 * 1024 * 1024


def scrub_text(s: str) -> str:
    return USER_PATH.sub(lambda m: m.group(1) + "<user>", s)


def scrub_obj(o):
    if isinstance(o, str):
        return scrub_text(o)
    if isinstance(o, list):
        return [scrub_obj(x) for x in o]
    if isinstance(o, dict):
        return {k: scrub_obj(v) for k, v in o.items()}
    return o


def has_personal_path(o) -> bool:
    if isinstance(o, str):
        return bool(USER_PATH.search(o))
    if isinstance(o, list):
        return any(has_personal_path(x) for x in o)
    if isinstance(o, dict):
        return any(has_personal_path(v) for v in o.values())
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", help="notebooks à traiter (défaut : notebooks/*.ipynb)")
    p.add_argument("--strip-outputs", action="store_true", help="supprime les sorties et les compteurs d'exécution")
    p.add_argument("--check", action="store_true", help="vérifie sans modifier ; code de sortie 1 en cas de problème")
    args = p.parse_args()

    files = [Path(x) for x in args.paths] or sorted(Path("notebooks").glob("*.ipynb"))
    if not files:
        print("Aucun notebook trouvé.")
        return 0

    problems = 0
    for f in files:
        nb = json.loads(f.read_text(encoding="utf-8"))
        size = f.stat().st_size
        if args.check:
            bad_path = has_personal_path(nb)
            too_big = size > MAX_SIZE
            status = "OK" if not (bad_path or too_big) else "PROBLÈME"
            detail = ", ".join(x for x, cond in [("chemin personnel", bad_path), (f"{size / 1e6:.1f} Mo > 5 Mo", too_big)] if cond)
            print(f"{status:9} {f} ({size / 1e6:.2f} Mo) {detail}")
            problems += int(bad_path or too_big)
            continue

        nb = scrub_obj(nb)
        if args.strip_outputs:
            for c in nb.get("cells", []):
                if c.get("cell_type") == "code":
                    c["outputs"], c["execution_count"] = [], None
        f.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"nettoyé   {f} ({f.stat().st_size / 1e6:.2f} Mo)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
