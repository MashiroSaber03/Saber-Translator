"""
Manga Insight 角色卡工坊回归检查脚本

运行方式:
    python tests_backend/manga_insight_character_cards_regression.py
"""

import asyncio
import copy
import io
import os
import shutil
import sys
import types
import uuid
import zipfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 某些最小环境未安装 PyYAML，这里注入最小桩避免导入失败。
try:
    import yaml  # type: ignore # noqa: F401
except ModuleNotFoundError:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda *_args, **_kwargs: {}
    yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
    sys.modules["yaml"] = yaml_stub


def _build_demo_card(name: str) -> dict:
    from src.core.manga_insight.character_cards.mappers import build_card_template

    character = {
        "name": name,
        "aliases": [],
        "description": f"{name} 的角色简介",
        "arc": "从迷茫到坚定",
        "first_appearance": 1,
        "relationships": [],
        "key_moments": [],
    }
    timeline_data = {"plot_threads": []}
    dialogues = [{"speaker": name, "text": "先做最重要的事。", "page": 1}]
    return build_card_template(character, "故事正在推进。", dialogues, timeline_data)


async def _cleanup_book(book_id: str) -> None:
    from src.core.manga_insight.storage import get_insight_storage_path

    shutil.rmtree(get_insight_storage_path(book_id), ignore_errors=True)


async def check_missing_timeline_candidates_error() -> None:
    """时间线缺失时，候选角色读取应报错。"""
    try:
        from src.core.manga_insight.character_cards import CharacterCardGenerator
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    book_id = f"cc_missing_timeline_{uuid.uuid4().hex[:8]}"
    generator = CharacterCardGenerator(book_id)
    try:
        await generator.get_candidates()
        raise AssertionError("缺失时间线时应抛出错误")
    except ValueError as exc:
        assert "时间线" in str(exc), f"错误信息应包含“时间线”，实际: {exc}"
    finally:
        await _cleanup_book(book_id)


async def check_alias_dialogue_extraction() -> None:
    """角色对话抽取应覆盖别名匹配。"""
    try:
        from src.core.manga_insight.character_cards import CharacterCardGenerator
        from src.core.manga_insight.storage import AnalysisStorage
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    book_id = f"cc_alias_dialogue_{uuid.uuid4().hex[:8]}"
    storage = AnalysisStorage(book_id)
    try:
        await storage.save_timeline({
            "mode": "enhanced",
            "characters": [
                {
                    "name": "主角",
                    "aliases": ["阿主", "Hero"],
                    "first_appearance": 1,
                    "description": "测试角色",
                    "arc": "成长",
                    "relationships": [],
                    "key_moments": [],
                }
            ],
        })

        await storage.save_page_analysis(1, {
            "dialogues": [
                {"speaker_name": "阿主", "text": "这是别名对话一"},
            ],
        })
        await storage.save_page_analysis(2, {
            "panels": [
                {
                    "dialogues": [
                        {"speaker": "hero", "text": "这是别名对话二"},
                    ],
                }
            ],
        })

        generator = CharacterCardGenerator(book_id)
        result = await generator.get_candidates()
        candidates = result.get("candidates", [])
        target = next((item for item in candidates if item.get("name") == "主角"), None)
        assert target is not None, "应包含主角候选项"
        assert target.get("dialogue_count") == 2, f"应识别 2 条别名对话，实际: {target}"
        assert target.get("sample_pages") == [1, 2], f"sample_pages 应包含 [1,2]，实际: {target.get('sample_pages')}"
    finally:
        await _cleanup_book(book_id)


async def check_v2_required_fields_validation() -> None:
    """V2 必填字段校验应有效。"""
    try:
        from src.core.manga_insight.character_cards.validator import validate_card_v2
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    card = _build_demo_card("校验角色")
    validation = validate_card_v2(card)
    assert validation.get("valid") is True, f"标准模板应校验通过，实际: {validation}"

    broken = copy.deepcopy(card)
    broken["data"].pop("first_mes", None)
    broken_validation = validate_card_v2(broken)
    assert broken_validation.get("valid") is False, "缺失必填字段时应校验失败"
    assert any("data.first_mes" in err for err in broken_validation.get("errors", [])), \
        f"应命中 first_mes 缺失，实际: {broken_validation}"


async def check_worldbook_schema_validation() -> None:
    """世界书条目结构校验应拒绝非法 key。"""
    try:
        from src.core.manga_insight.character_cards.validator import validate_character_book
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    valid_book = _build_demo_card("世界书角色")["data"]["character_book"]
    errors, _warnings = validate_character_book(valid_book)
    assert not errors, f"合法 worldbook 不应报错，实际: {errors}"

    invalid_book = {
        "entries": [
            {
                "uid": 1,
                "key": [],
                "content": "bad",
            }
        ]
    }
    errors, _warnings = validate_character_book(invalid_book)
    assert any(".key" in err for err in errors), f"非法 key 应报错，实际: {errors}"


async def check_extensions_schema_validation() -> None:
    """扩展区 schema 校验应拦截 regex/mvu/ui 结构错误。"""
    try:
        from src.core.manga_insight.character_cards.validator import validate_extensions
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    valid_extensions = _build_demo_card("扩展角色")["data"]["extensions"]
    errors, _warnings = validate_extensions(valid_extensions)
    assert not errors, f"合法扩展区不应报错，实际: {errors}"

    invalid_extensions = {
        "saber_tavern": {
            "regex_profiles": {},
            "mvu": {"variables": "bad"},
            "ui_manifest": [],
        }
    }
    errors, _warnings = validate_extensions(invalid_extensions)
    assert any("regex_profiles" in err for err in errors), f"应命中 regex_profiles 错误，实际: {errors}"
    assert any("mvu.variables" in err for err in errors), f"应命中 mvu.variables 错误，实际: {errors}"
    assert any("ui_manifest" in err for err in errors), f"应命中 ui_manifest 错误，实际: {errors}"


async def check_compile_input_validation() -> None:
    """编译接口核心逻辑应校验入参类型。"""
    try:
        from src.core.manga_insight.character_cards import CharacterCardGenerator
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    book_id = f"cc_compile_input_{uuid.uuid4().hex[:8]}"
    generator = CharacterCardGenerator(book_id)
    try:
        try:
            await generator.compile_cards([])  # type: ignore[arg-type]
            raise AssertionError("draft 为 list 时应报错")
        except ValueError as exc:
            assert "draft" in str(exc), f"应提示 draft 类型错误，实际: {exc}"

        try:
            await generator.compile_cards({"cards": []}, "abc")  # type: ignore[arg-type]
            raise AssertionError("character_names 为 string 时应报错")
        except ValueError as exc:
            assert "character_names" in str(exc), f"应提示 character_names 类型错误，实际: {exc}"
    finally:
        await _cleanup_book(book_id)


async def check_compile_malformed_cards_item() -> None:
    """编译时遇到非法 cards 项应返回可读错误而非崩溃。"""
    try:
        from src.core.manga_insight.character_cards import CharacterCardGenerator
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    book_id = f"cc_compile_cards_item_{uuid.uuid4().hex[:8]}"
    generator = CharacterCardGenerator(book_id)
    try:
        result = await generator.compile_cards({"cards": [123]})
        assert result.get("valid") is False, f"非法草稿项应编译失败，实际: {result}"
        assert any("草稿项格式错误" in err for err in result.get("errors", [])), \
            f"应返回草稿项格式错误，实际: {result.get('errors')}"
    finally:
        await _cleanup_book(book_id)


async def check_png_roundtrip() -> None:
    """PNG 写入后应可无损回读。"""
    try:
        from src.core.manga_insight.character_cards.png_codec import CharacterCardPngCodec
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    card = _build_demo_card("PNG角色")
    png_bytes = CharacterCardPngCodec.write_card_png(card, base_image_path=None, mirror_ccv3=True)
    ok, err = CharacterCardPngCodec.validate_roundtrip(card, png_bytes)
    assert ok is True, f"PNG 回读一致性校验失败: {err}"
    decoded = CharacterCardPngCodec.read_card_png(png_bytes)
    assert decoded == card, "PNG 解码结果应与原卡一致"


async def check_regex_profile_escape() -> None:
    """角色名包含正则特殊字符时，规则模板应可安全编译。"""
    try:
        import re
        from src.core.manga_insight.character_cards.mappers import build_regex_profiles
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    character_name = "A[Hero](v2)+?"
    profiles = build_regex_profiles(character_name)
    target = next((item for item in profiles if item.id == "name_normalization"), None)
    assert target is not None, "应生成 name_normalization 规则"
    try:
        re.compile(target.pattern)
    except re.error as exc:
        raise AssertionError(f"角色名转义后规则仍无法编译: {exc}") from exc


async def check_batch_export_filename_collision() -> None:
    """批量导出在文件名冲突时应稳定生成唯一文件名。"""
    try:
        from src.core.manga_insight.character_cards import CharacterCardGenerator
        from src.core.manga_insight.character_cards.png_codec import CharacterCardPngCodec
        from src.core.manga_insight.storage import AnalysisStorage
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    book_id = f"cc_batch_export_{uuid.uuid4().hex[:8]}"
    storage = AnalysisStorage(book_id)
    try:
        characters = ["A/B", "A:B"]
        await storage.save_timeline({
            "mode": "enhanced",
            "characters": [
                {
                    "name": "A/B",
                    "aliases": [],
                    "first_appearance": 1,
                    "description": "角色1",
                    "arc": "弧线1",
                    "relationships": [],
                    "key_moments": [],
                },
                {
                    "name": "A:B",
                    "aliases": [],
                    "first_appearance": 2,
                    "description": "角色2",
                    "arc": "弧线2",
                    "relationships": [],
                    "key_moments": [],
                },
            ],
        })

        draft = {
            "book_id": book_id,
            "style": "balanced",
            "cards": [
                {"character": "A/B", "card": _build_demo_card("A/B"), "source_stats": {}},
                {"character": "A:B", "card": _build_demo_card("A:B"), "source_stats": {}},
            ],
        }

        generator = CharacterCardGenerator(book_id)
        compile_result = await generator.compile_cards(draft)
        assert compile_result.get("valid") is True, f"编译应通过，实际: {compile_result}"

        zip_result = await generator.export_batch_zip(characters)
        zip_bytes = zip_result.get("zip_bytes", b"")
        assert isinstance(zip_bytes, (bytes, bytearray)) and len(zip_bytes) > 0, "应生成 zip 二进制内容"

        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            names = sorted(zf.namelist())
            assert names == ["A_B.png", "A_B_2.png"], f"冲突命名应避让，实际: {names}"

            decoded_names = set()
            for file_name in names:
                decoded = CharacterCardPngCodec.read_card_png(zf.read(file_name))
                decoded_names.add(decoded.get("data", {}).get("name", ""))
            assert decoded_names == {"A/B", "A:B"}, f"导出内容不应互相覆盖，实际: {decoded_names}"
    finally:
        await _cleanup_book(book_id)


async def check_batch_export_input_validation() -> None:
    """批量导出应校验角色名数组元素类型。"""
    try:
        from src.core.manga_insight.character_cards import CharacterCardGenerator
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MISSING_DEPENDENCY:{exc.name}") from exc

    book_id = f"cc_batch_input_{uuid.uuid4().hex[:8]}"
    generator = CharacterCardGenerator(book_id)
    try:
        try:
            await generator.export_batch_zip(["ok", 1])  # type: ignore[list-item]
            raise AssertionError("包含非字符串角色名时应报错")
        except ValueError as exc:
            assert "character_names[1]" in str(exc), f"应提示具体下标，实际: {exc}"
    finally:
        await _cleanup_book(book_id)


async def main() -> int:
    checks = [
        ("missing_timeline_candidates_error", check_missing_timeline_candidates_error),
        ("alias_dialogue_extraction", check_alias_dialogue_extraction),
        ("v2_required_fields_validation", check_v2_required_fields_validation),
        ("worldbook_schema_validation", check_worldbook_schema_validation),
        ("extensions_schema_validation", check_extensions_schema_validation),
        ("compile_input_validation", check_compile_input_validation),
        ("compile_malformed_cards_item", check_compile_malformed_cards_item),
        ("png_roundtrip", check_png_roundtrip),
        ("regex_profile_escape", check_regex_profile_escape),
        ("batch_export_filename_collision", check_batch_export_filename_collision),
        ("batch_export_input_validation", check_batch_export_input_validation),
    ]

    failed = 0
    skipped = 0
    for name, func in checks:
        try:
            await func()
            print(f"[PASS] {name}")
        except RuntimeError as exc:
            if str(exc).startswith("MISSING_DEPENDENCY:"):
                skipped += 1
                print(f"[SKIP] {name}: 缺少依赖 {str(exc).split(':', 1)[1]}")
            else:
                failed += 1
                print(f"[FAIL] {name}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {name}: {exc}")

    if failed > 0:
        print(f"\n角色卡回归失败: {failed}/{len(checks)} (跳过 {skipped})")
        return 1

    print(f"\n角色卡回归通过: {len(checks) - skipped}/{len(checks)} (跳过 {skipped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
