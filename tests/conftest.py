import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Tests must never hit real services, whatever is in backend/.env.
os.environ["GEMINI_API_KEY"] = "test-key"
os.environ["GEMINI_API_KEY_2"] = ""
os.environ["GEMINI_API_KEYS"] = ""
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = ""

from app.models import GlossaryTerm  # noqa: E402
from app.services.glossary import GlossaryMatcher  # noqa: E402

TERMS = [
    ("bail", "ضمانت", "criminal"),
    ("accused", "ملزم", "criminal"),
    ("witness", "گواہ", "court"),
    ("judgment", "فیصلہ", "court"),
    ("record of rights", "فرد ملکیت", "land"),
    ("rights", "حقوق", "constitutional"),
    ("High Court", "عدالت عالیہ", "court"),
    ("Pakistan Penal Code", "مجموعہ تعزیرات پاکستان", "statute"),
    ("party", "فریق", "civil"),
    ("summons", "سمن", "court"),
    ("record of rights", "جمع بندی", "land"),
]


@pytest.fixture
def terms() -> list[GlossaryTerm]:
    return [GlossaryTerm(id=i, term_en=en, term_ur=ur, category=cat) for i, (en, ur, cat) in enumerate(TERMS, 1)]


@pytest.fixture
def matcher(terms: list[GlossaryTerm]) -> GlossaryMatcher:
    return GlossaryMatcher(terms)
