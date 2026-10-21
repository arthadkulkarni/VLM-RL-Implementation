import json
import re

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_BARE_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)


def parse_caption_entities(text):
    """Parse a VLM response expected to contain a fenced ```json block with
    {"caption": str, "entities": [{"name","type","description"}]}. Falls back
    to treating the raw text as the caption with no entities on any parse
    failure, mirroring the defensive extract_boxed_content style already used
    in vllm_service_init/start_vllm_server.py.
    """
    text = text or ""

    for pattern in (_JSON_BLOCK_RE, _BARE_JSON_RE):
        match = pattern.search(text)
        if not match:
            continue
        try:
            parsed = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue

        caption = str(parsed.get("caption", "")).strip()
        raw_entities = parsed.get("entities", [])
        if not isinstance(raw_entities, list):
            raw_entities = []

        entities = []
        for entity in raw_entities:
            if not isinstance(entity, dict):
                continue
            entities.append({
                "name": str(entity.get("name", "")).strip(),
                "type": str(entity.get("type", "")).strip(),
                "description": str(entity.get("description", "")).strip(),
            })

        return {"caption": caption or text.strip(), "entities": entities}

    return {"caption": text.strip(), "entities": []}
