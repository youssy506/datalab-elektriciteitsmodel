"""
Python-versie van check-evidence.js uit de ECC-skill 'frontend-design-direction'
(Node.js is niet geïnstalleerd). Zelfde regels: controleert of het bewijsrecord compleet is,
NIET of de waarnemingen kloppen of de schermafbeeldingen goed zijn.

Uitvoeren: python reports/design/check_evidence.py reports/design/evidence.json
"""

import json
import os
import sys

STATUSSEN = ["loading", "empty", "error", "partial", "success", "permission"]


def tekst(w):
    return isinstance(w, str) and w.strip() != ""


def sleutel(viewport):
    return "".join(viewport.lower().split())


def controleer(rapport):
    if not isinstance(rapport, dict):
        return ["report must be an object"]
    fouten = []

    def check(naam, item, nvt_mag=False):
        if isinstance(item, dict) and item.get("status") == "pass" and tekst(item.get("evidence")):
            return []
        if nvt_mag and isinstance(item, dict) and item.get("status") == "not-applicable" and tekst(item.get("reason")):
            return []
        return [f"{naam} needs a passing observation" + (" or a not-applicable reason" if nvt_mag else "")]

    renders = rapport.get("renderedEvidence") if isinstance(rapport.get("renderedEvidence"), list) else []
    doelen = rapport.get("targetViewports") if isinstance(rapport.get("targetViewports"), list) else []
    if not tekst(rapport.get("contract")):
        fouten.append("contract is required")
    if not (doelen and all(tekst(d) for d in doelen)):
        fouten.append("targetViewports must contain nonempty viewport strings")
    for d in dict.fromkeys(sleutel(d) for d in doelen if tekst(d)):
        if not any(isinstance(r, dict) and tekst(r.get("viewport")) and sleutel(r["viewport"]) == d for r in renders):
            fouten.append(f"renderedEvidence is missing target viewport {d}")
    fouten += check("accessibility", rapport.get("accessibility"))
    fouten += check("responsive", rapport.get("responsive"))
    for s in STATUSSEN:
        fouten += check(f"contentStates.{s}", (rapport.get("contentStates") or {}).get(s), True)
    if not renders:
        fouten.append("renderedEvidence is required")
    for i, r in enumerate(renders):
        if not (isinstance(r, dict) and all(tekst(r.get(k)) for k in ("viewport", "artifact", "observation"))):
            fouten.append(f"renderedEvidence[{i}] needs viewport, artifact and observation")
    return fouten


def main(args):
    if len(args) != 1:
        print("Usage: python check_evidence.py <local-report.json>")
        return 1
    pad = os.path.abspath(args[0])
    with open(pad, encoding="utf-8") as f:
        try:
            rapport = json.load(f)
        except json.JSONDecodeError:
            print("Report must contain valid JSON")
            return 1
    fouten = controleer(rapport)
    if fouten:
        print("\n".join(fouten))
        return 1
    root = os.path.realpath(os.path.dirname(pad))
    for item in rapport["renderedEvidence"]:
        a = item["artifact"]
        if os.path.isabs(a) or ":" in a or ".." in a.replace("\\", "/").split("/"):
            print("Evidence artifacts must stay inside the report directory")
            return 1
        echt = os.path.realpath(os.path.join(root, a))
        if os.path.relpath(echt, root).startswith(".."):
            print("Evidence artifacts must stay inside the report directory")
            return 1
        volledig = os.path.join(root, a)
        if os.path.islink(volledig) or not os.path.isfile(volledig) or os.path.getsize(volledig) == 0:
            print("Evidence artifact must be a nonempty local file")
            return 1
    print("Evidence record complete; visual correctness still requires rendered inspection.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
