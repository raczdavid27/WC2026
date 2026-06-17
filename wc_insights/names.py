"""Shared team-name normalization.

Used to join names across feeds — ESPN, eloratings.net, and the martj42
historical results dataset all spell some countries differently. Everything
goes through norm() so lookups are consistent.
"""

import unicodedata

# Canonical forms (left = variant, right = canonical normalized name).
ALIASES = {
    "usa": "united states", "united states of america": "united states",
    "korea republic": "south korea", "korea dpr": "north korea",
    "ir iran": "iran", "iran islamic republic": "iran",
    "cote d ivoire": "ivory coast", "cabo verde": "cape verde",
    "czechia": "czech republic", "china pr": "china",
    "congo dr": "dr congo", "republic of ireland": "ireland", "turkiye": "turkey",
    "bosnia and herzegovina": "bosnia", "bosnia herzegovina": "bosnia",
    "north macedonia": "macedonia", "the gambia": "gambia",
    "kyrgyz republic": "kyrgyzstan", "chinese taipei": "taiwan",
    "curacao": "curacao", "st kitts and nevis": "saint kitts and nevis",
}


def norm(name):
    """Lowercase, strip accents/punctuation, collapse spaces, apply aliases."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = "".join(c if c.isalnum() or c == " " else " " for c in s)
    s = " ".join(s.split())
    return ALIASES.get(s, s)
