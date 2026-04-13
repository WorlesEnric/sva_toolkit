from __future__ import annotations

from sva_toolkit.formal.model import FormalProperty
from sva_toolkit.sva.emitter import emit_property_body


def normalize_property(prop: FormalProperty) -> FormalProperty:
    """Return a normalized copy of the property with cleaned body text."""

    if prop.ast is not None:
        body = emit_property_body(prop.ast.body)
    else:
        body = " ".join(prop.body.split()).rstrip("; ").strip()
    return prop.model_copy(update={"body": body})
