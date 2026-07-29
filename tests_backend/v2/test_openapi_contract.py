from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import yaml

from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.schema import metadata
from src.backend_v2.storage.seeding import seed_system_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = PROJECT_ROOT / "openapi" / "v2.yaml"
GENERATED_TYPES = PROJECT_ROOT / "vue-frontend" / "src" / "api" / "generated" / "v2.ts"
V2_API_DIRECTORY = PROJECT_ROOT / "vue-frontend" / "src" / "api" / "v2"
HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete"})


def _document() -> dict[str, Any]:
    value = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _normalize_runtime_path(path: str) -> str:
    without_prefix = path.removeprefix("/api/v2")
    return re.sub(r"<(?:[^:>]+:)?([^>]+)>", r"{\1}", without_prefix)


def test_openapi_operations_match_the_runtime_route_set(tmp_path: Path) -> None:
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="openapi-contract",
                epoch_token="test-token",
                test_mode=True,
            ),
            engine=engine,
        )
    )
    try:
        runtime_operations = {
            (method.lower(), _normalize_runtime_path(rule.rule))
            for rule in app.url_map.iter_rules()
            if rule.rule.startswith("/api/v2/")
            and rule.rule != "/api/v2/openapi.json"
            for method in rule.methods - {"HEAD", "OPTIONS"}
        }
        document = _document()
        contract_operations = {
            (method, path)
            for path, path_item in document["paths"].items()
            for method in HTTP_METHODS
            if method in path_item
        }
        assert contract_operations == runtime_operations
    finally:
        app.extensions["saber_v2_runtime"].close()
        engine.dispose()


def test_openapi_contract_is_closed_and_all_local_refs_resolve() -> None:
    document = _document()
    assert document["openapi"].startswith("3.1.")
    for node in _walk(document):
        reference = node.get("$ref")
        if not isinstance(reference, str) or not reference.startswith("#/"):
            continue
        resolved: Any = document
        for token in reference[2:].split("/"):
            resolved = resolved[token.replace("~1", "/").replace("~0", "~")]
        assert resolved is not None


def test_every_operation_has_a_unique_operation_id_and_responses() -> None:
    document = _document()
    operation_ids: list[str] = []
    for path, path_item in document["paths"].items():
        for method in HTTP_METHODS:
            operation = path_item.get(method)
            if operation is None:
                continue
            operation_id = operation.get("operationId")
            assert isinstance(operation_id, str) and operation_id, (
                f"{method.upper()} {path} has no operationId"
            )
            assert operation.get("responses"), (
                f"{method.upper()} {path} has no responses"
            )
            operation_ids.append(operation_id)
    assert len(operation_ids) == len(set(operation_ids))


def test_public_operations_do_not_use_an_untyped_generic_object() -> None:
    document = _document()
    assert "GenericObject" not in document["components"]["schemas"]
    assert "GenericSuccess" not in document["components"]["responses"]
    serialized = yaml.safe_dump(document, sort_keys=False)
    assert "#/components/schemas/GenericObject" not in serialized


def test_required_backend_first_commands_are_explicit() -> None:
    document = _document()
    paths = document["paths"]
    assert {
        "/pages/{page_id}",
        "/pages/{page_id}/document/batch",
        "/pages/{page_id}/repairs",
        "/operations/{operation_id}",
        "/jobs/{job_id}/resume",
        "/jobs/{job_id}/continue",
        "/studio/chat/sessions/{session_id}/messages",
    } <= set(paths)

    schemas = document["components"]["schemas"]
    assert schemas["JobStatus"]["enum"]
    assert schemas["OperationStatus"]["enum"]
    assert set(schemas["StudioMessageCommand"]["required"]) >= {
        "baseSessionRevision",
    }


def test_mutating_commands_use_idempotency_keys() -> None:
    document = _document()
    for path, path_item in document["paths"].items():
        for method in ("post", "put", "patch", "delete"):
            operation = path_item.get(method)
            if operation is None:
                continue
            if operation.get("x-command-mode") == "transient":
                continue
            parameters = [*path_item.get("parameters", []), *operation.get("parameters", [])]
            refs = {parameter.get("$ref") for parameter in parameters}
            assert (
                "#/components/parameters/IdempotencyKey" in refs
            ), f"{method.upper()} {path} has no Idempotency-Key"


def test_typescript_client_is_generated_from_the_contract() -> None:
    generated = GENERATED_TYPES.read_text(encoding="utf-8")
    assert "This file was auto-generated by openapi-typescript." in generated
    assert '"/jobs/{job_id}/resume"' in generated


def test_v2_api_modules_do_not_declare_handwritten_v2_interfaces() -> None:
    declarations: list[str] = []
    pattern = re.compile(r"^\s*(?:export\s+)?interface\s+V2\w+", re.MULTILINE)
    for path in V2_API_DIRECTORY.glob("*.ts"):
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            declarations.append(f"{path.name}:{match.group(0).strip()}")
    assert declarations == []
