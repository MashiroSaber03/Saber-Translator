import unittest

from src.core.translation_constraints import (
    extract_glossary_candidates_from_payload,
    build_glossary_prompt,
    build_non_translate_guard_prompt,
    build_non_translate_prompt,
    collect_glossary_warnings,
    normalize_glossary_settings,
    normalize_non_translate_settings,
    protect_texts_with_non_translate,
    restore_texts_with_non_translate,
)


class TranslationConstraintsTests(unittest.TestCase):
    def test_extract_glossary_candidates_filters_invalid_and_duplicate_entries(self) -> None:
        candidates = extract_glossary_candidates_from_payload(
            [
                {"source": "Alice", "target": "爱丽丝"},
                {"source": "Alice", "target": "阿丽丝"},
                {"source": "Bob", "target": "鲍勃", "note": "角色"},
                {"source": "", "target": "空白"},
                {"source": "NoTarget", "target": ""},
                "invalid",
            ],
            existing_entries=[
                {"source": "Bob", "target": "旧鲍勃", "note": "已存在", "matchMode": "text"},
            ],
        )

        self.assertEqual(
            candidates,
            [
                {"source": "Alice", "target": "爱丽丝", "note": "", "matchMode": "text"},
            ],
        )

    def test_build_glossary_prompt_only_includes_matching_entries(self) -> None:
        settings = normalize_glossary_settings(
            {
                "enabled": True,
                "entries": [
                    {"source": "Alice", "target": "爱丽丝", "note": "主角", "matchMode": "text"},
                    {"source": r"drago\w+", "target": "巨龙", "note": "怪物", "matchMode": "regex"},
                    {"source": "Unused", "target": "不会出现", "note": "", "matchMode": "text"},
                ],
            }
        )

        prompt = build_glossary_prompt(
            settings,
            ["Alice and dragon are here", "ALICE meets another dragon"],
            target_language="zh",
        )

        self.assertIn("###术语表", prompt)
        self.assertIn("Alice|爱丽丝|主角", prompt)
        self.assertIn("dragon|巨龙|怪物", prompt)
        self.assertNotIn("Unused|不会出现", prompt)

    def test_build_non_translate_prompt_only_includes_matching_entries(self) -> None:
        settings = normalize_non_translate_settings(
            {
                "enabled": True,
                "entries": [
                    {"pattern": "<keep>", "note": "占位符", "matchMode": "text"},
                    {"pattern": r"\{[A-Z_]+\}", "note": "宏变量", "matchMode": "regex"},
                    {"pattern": "unused-marker", "note": "", "matchMode": "text"},
                ],
            }
        )

        prompt = build_non_translate_prompt(
            settings,
            ["hello <keep> world", "value is {PLAYER_NAME}"],
            target_language="zh",
        )

        self.assertIn("###禁翻表", prompt)
        self.assertIn("<keep>|占位符", prompt)
        self.assertIn("{PLAYER_NAME}|宏变量", prompt)
        self.assertNotIn("unused-marker", prompt)

    def test_non_translate_placeholders_round_trip(self) -> None:
        settings = normalize_non_translate_settings(
            {
                "enabled": True,
                "entries": [
                    {"pattern": "<keep>", "note": "", "matchMode": "text"},
                    {"pattern": r"\{[^}]+\}", "note": "", "matchMode": "regex"},
                ],
            }
        )
        originals = [
            "value <keep> {player_name} <keep>",
            "plain text",
        ]

        protected, mappings = protect_texts_with_non_translate(originals, settings)

        self.assertNotEqual(protected[0], originals[0])
        self.assertIn("__SABER_NTL_", protected[0])

        restored = restore_texts_with_non_translate(protected, mappings)
        self.assertEqual(restored, originals)

    def test_non_translate_guard_prompt_lists_placeholders(self) -> None:
        prompt = build_non_translate_guard_prompt(
            [
                [{"placeholder": "__SABER_NTL_0001__", "original": "<SFX_01>"}],
                [{"placeholder": "__SABER_NTL_0002__", "original": "{PLAYER_NAME}"}],
            ],
            target_language="zh",
        )

        self.assertIn("###占位符保护规则", prompt)
        self.assertIn("__SABER_NTL_0001__", prompt)
        self.assertIn("__SABER_NTL_0002__", prompt)

    def test_collect_glossary_warnings_reports_missing_expected_translation(self) -> None:
        settings = normalize_glossary_settings(
            {
                "enabled": True,
                "entries": [
                    {"source": "Alice", "target": "爱丽丝", "note": "", "matchMode": "text"},
                    {"source": r"dragon", "target": "巨龙", "note": "", "matchMode": "regex"},
                ],
            }
        )

        warnings = collect_glossary_warnings(
            settings,
            source_text="Alice and dragon are here",
            translated_text="她和龙在这里",
            image_index=3,
            bubble_index=5,
        )

        self.assertEqual(len(warnings), 2)
        self.assertEqual(warnings[0]["imageIndex"], 3)
        self.assertEqual(warnings[0]["bubbleIndex"], 5)
        self.assertEqual({warning["expectedTarget"] for warning in warnings}, {"爱丽丝", "巨龙"})


if __name__ == "__main__":
    unittest.main()
