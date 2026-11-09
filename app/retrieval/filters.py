"""Validated exact-match filters for the existing chunk payload schema."""

import re

from qdrant_client import models


class InvalidFilterError(ValueError):
    """A filter has an unsupported field, value, or structure."""


def validate_filters(filters: dict | None) -> dict:
    """Return an independent normalized copy; missing payload fields never match."""
    if filters is None:
        return {}
    if not isinstance(filters, dict):
        raise InvalidFilterError("filters must be a dictionary or None")
    supported = {"filename", "format", "source_path", "page_number", "page_label", "metadata"}
    result = {}
    for key, value in filters.items():
        if key not in supported:
            raise InvalidFilterError("Unsupported filter; supported fields: " + ", ".join(sorted(supported)))
        if key == "page_number":
            if type(value) is not int or value <= 0:
                raise InvalidFilterError("page_number must be a positive integer")
        elif key == "metadata":
            if not isinstance(value, dict):
                raise InvalidFilterError("metadata must be a dictionary of string keys and values")
            copied = {}
            for name, item in value.items():
                if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                    raise InvalidFilterError("metadata keys must use letters, digits, and underscores, starting with a letter or underscore")
                if not isinstance(item, str) or not item.strip():
                    raise InvalidFilterError("metadata values must be non-empty strings")
                copied[name] = item
            value = copied
        else:
            if not isinstance(value, str) or not value.strip():
                raise InvalidFilterError(f"{key} must be a non-empty string")
            if key == "format":
                value = value.lower()
                if value not in {"pdf", "docx", "txt", "md", "markdown"}:
                    raise InvalidFilterError("format must be pdf, docx, txt, md, or markdown")
        result[key] = value
    return result


def build_qdrant_filter(filters: dict | None) -> models.Filter | None:
    """Combine exact matches with AND; page conditions share one location."""
    validated = validate_filters(filters)
    conditions = []
    location_conditions = []
    for key, value in validated.items():
        if key == "metadata":
            conditions.extend(models.FieldCondition(key=f"metadata.{name}",
                match=models.MatchValue(value=item)) for name, item in value.items())
        elif key in {"page_number", "page_label"}:
            location_conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        else:
            conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    if location_conditions:
        conditions.append(models.NestedCondition(nested=models.Nested(
            key="locations", filter=models.Filter(must=location_conditions))))
    return models.Filter(must=conditions) if conditions else None
