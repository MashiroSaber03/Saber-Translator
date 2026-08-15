import json
import unittest
from pathlib import Path

from src.backend_v2.content.translation_constraints import (
    DEFAULT_AUTO_GLOSSARY_PROMPT,
)
from src.shared.constants import (
    BATCH_TRANSLATE_JSON_SYSTEM_TEMPLATE,
    BATCH_TRANSLATE_SYSTEM_TEMPLATE,
    DEFAULT_AI_VISION_OCR_JSON_PROMPT,
    DEFAULT_AI_VISION_OCR_PROMPT,
    DEFAULT_HQ_TRANSLATE_PROMPT,
    DEFAULT_PROMPT,
    DEFAULT_PROOFREADING_PROMPT,
    DEFAULT_TRANSLATE_JSON_PROMPT,
    DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT,
)
from src.shared.prompt_defaults import get_prompt_factory_defaults


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PromptFactoryDefaultsTests(unittest.TestCase):
    def test_backend_defaults_match_the_bundled_factory_resource(self) -> None:
        defaults_path = (
            PROJECT_ROOT / "src" / "shared" / "prompt_defaults_factory.json"
        )
        expected = json.loads(defaults_path.read_text(encoding="utf-8"))

        self.assertEqual(get_prompt_factory_defaults(), expected)
        self.assertEqual(DEFAULT_PROMPT, expected["singleNormal"])
        self.assertEqual(DEFAULT_TRANSLATE_JSON_PROMPT, expected["singleJson"])
        self.assertEqual(BATCH_TRANSLATE_SYSTEM_TEMPLATE, expected["batchNormal"])
        self.assertEqual(
            BATCH_TRANSLATE_JSON_SYSTEM_TEMPLATE,
            expected["batchJson"],
        )
        self.assertEqual(
            DEFAULT_AI_VISION_OCR_PROMPT,
            expected["aiVisionOcrNormal"],
        )
        self.assertEqual(
            DEFAULT_AI_VISION_OCR_JSON_PROMPT,
            expected["aiVisionOcrJson"],
        )
        self.assertEqual(DEFAULT_HQ_TRANSLATE_PROMPT, expected["hqTranslation"])
        self.assertEqual(DEFAULT_PROOFREADING_PROMPT, expected["proofreading"])
        self.assertEqual(DEFAULT_AUTO_GLOSSARY_PROMPT, expected["autoGlossary"])
        self.assertEqual(
            DEFAULT_WEB_IMPORT_EXTRACTION_PROMPT,
            expected["webImportExtraction"],
        )

    def test_factory_defaults_are_returned_as_copies(self) -> None:
        first = get_prompt_factory_defaults()
        first["singleNormal"] = "changed"

        self.assertNotEqual(
            first["singleNormal"],
            get_prompt_factory_defaults()["singleNormal"],
        )


if __name__ == "__main__":
    unittest.main()
