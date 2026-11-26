"""Transparent, standalone compatibility matching for two Rueglim profiles.

The module intentionally contains the complete algorithm in one place.  It has
no database, web framework, ORM, registration or recommendation dependencies.

Production usage normally stores embeddings beforehand::

    result = compare_users(user_a, user_b, cross_fn=my_cross_score)

For local experiments embeddings may be generated on demand::

    result = compare_users(
        user_a,
        user_b,
        auto_generate_embeddings=True,
        embed_fn=embed,
        cross_fn=cross_score,
    )

The model wrappers at the bottom use ``sentence-transformers`` lazily.  Unit
tests and the demo inject tiny deterministic functions and download no models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, sqrt
from typing import Any, Callable, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Initial heuristic values.  They must be calibrated on real outcome data.
GROUP_WEIGHTS = {
    "structured": 0.30,
    "tags": 0.10,
    "embeddings": 0.25,
    "cross": 0.35,
}

MAX_TOTAL_BONUS = 0.05
MAX_TOTAL_PENALTY = 0.05

# All scalar values below are expected on a 0..100 scale.
STRUCTURED_SIMILARITY_TESTS = [
    # (test name, field, relative weight)
    ("home_outdoor", "home_outdoor", 1.0),
    ("calm_intense", "calm_intense", 1.0),
    ("planned_spontaneous", "planned_spontaneous", 1.0),
    ("private_social", "private_social", 1.0),
    ("closeness", "closeness", 1.0),
    ("messaging_frequency", "messaging_frequency", 1.0),
]

STRUCTURED_DIRECTIONAL_TESTS = [
    # (test name, need field, give field, relative weight)
    ("tactility", "need_tactility", "give_tactility", 1.0),
    ("initiative", "need_initiative", "give_initiative", 1.0),
    ("emotional_support", "need_emotional_support", "give_emotional_support", 1.0),
]

# Дополнительная совместимость для полей need/give, чтобы можно было использовать разные имена для одного и того же поля
STRUCTURED_ALIASES = {
    "need_tactility": ("need_tactility", "tactility"),
    "give_tactility": ("give_tactility", "tactility"),
    "need_initiative": ("need_initiative", "initiative_target"),
    "give_initiative": ("give_initiative", "initiative_self"),
    "need_emotional_support": ("need_emotional_support",),
    "give_emotional_support": ("give_emotional_support",),
}

CHILDREN_COMPATIBILITY = {
    ("definitely_yes", "definitely_yes"): 1.00,
    ("definitely_yes", "probably_yes"): 0.85,
    ("definitely_yes", "unsure"): 0.55,
    ("definitely_yes", "probably_not"): 0.20,
    ("definitely_yes", "definitely_not"): 0.00,
    ("probably_yes", "probably_yes"): 1.00,
    ("probably_yes", "unsure"): 0.75,
    ("probably_yes", "probably_not"): 0.45,
    ("probably_yes", "definitely_not"): 0.20,
    ("unsure", "unsure"): 1.00,
    ("unsure", "probably_not"): 0.75,
    ("unsure", "definitely_not"): 0.55,
    ("probably_not", "probably_not"): 1.00,
    ("probably_not", "definitely_not"): 0.85,
    ("definitely_not", "definitely_not"): 1.00,
}

CATEGORICAL_TESTS = [
    # (test name, field, compatibility matrix, relative weight)
    ("future_children", "future_children", CHILDREN_COMPATIBILITY, 1.5),
]

# Explicit semantic relationships.  Reverse order is handled automatically.
TAG_RELATEDNESS = {
    ("anime", "manga"): 0.75,
    ("anime", "japanese_media"): 0.60,
    ("manga", "japanese_media"): 0.70,
    ("programming", "technology"): 0.65,
    ("gym", "fitness"): 0.80,
    ("hiking", "nature"): 0.70,
    ("books", "literature"): 0.80,
}

TAG_FAMILIES = {
    "anime": "japanese_media",
    "manga": "japanese_media",
    "japanese_media": "japanese_media",
    "programming": "technology",
    "technology": "technology",
    "gym": "active_lifestyle",
    "fitness": "active_lifestyle",
    "running": "active_lifestyle",
    "hiking": "outdoors",
    "nature": "outdoors",
    "books": "reading",
    "literature": "reading",
}

RARE_INTERESTS = {"astronomy", "birdwatching", "calligraphy", "folklore"}

EMBEDDING_FIELDS = [
    ("relationship", 1.2),
    ("conflict", 1.2),
    ("care", 1.0),
    ("communication", 1.1),
    ("weekend", 0.8),
    ("emotional_needs", 1.0),
    ("intimacy", 0.8),
]

# Used only by generate_embeddings().  Several answers can be joined into one
# semantic area, but areas never get collapsed into one whole-profile vector.
SEMANTIC_TEXT_FIELDS = {
    "relationship": ("relationship_view", "future_vision", "desired_feeling"),
    "conflict": ("conflict_style", "conflict_immediate", "repair_style"),
    "care": ("care_need", "care_give", "care_target", "care_self"),
    "communication": ("communication_style", "communication_need", "communication_give"),
    "weekend": ("weekend", "usual_weekend", "ordinary_weekend"),
    "emotional_needs": ("emotional_need", "emotional_support", "desired_feeling"),
    "intimacy": ("intimacy", "closeness_description", "boundaries"),
}

# (name, left field, right field, mode, task prompt, relative weight)
# This is deliberately a small hand-written matrix: no all-pairs comparison.
CROSS_TESTS = [
    (
        "conflict",
        "conflict_style",
        "conflict_style",
        "symmetric",
        "Estimate how compatible the two people's conflict and repair styles are.",
        1.2,
    ),
    (
        "relationship",
        "relationship_view",
        "relationship_view",
        "symmetric",
        "Estimate how compatible their views of a good relationship are.",
        1.2,
    ),
    (
        "communication",
        "communication_style",
        "communication_style",
        "symmetric",
        "Estimate how compatible their everyday communication styles are.",
        1.0,
    ),
    (
        "weekend",
        "usual_weekend",
        "usual_weekend",
        "symmetric",
        "Estimate how compatible their preferred ways of spending a weekend are.",
        0.8,
    ),
    (
        "care",
        "care_need",
        "care_give",
        "directional",
        "Estimate how well the second person's behaviour satisfies the first person's stated care need.",
        1.2,
    ),
    (
        "communication_need_give",
        "communication_need",
        "communication_give",
        "directional",
        "Estimate how well the second person's communication behaviour satisfies the first person's need.",
        1.1,
    ),
    (
        "emotional_support",
        "emotional_need",
        "emotional_support",
        "directional",
        "Estimate how well the second person's support behaviour meets the first person's emotional need.",
        1.2,
    ),
    (
        "fear_behaviour",
        "what_hurts_me",
        "conflict_behaviour",
        "directional",
        "Estimate whether the second person's conflict behaviour avoids what emotionally hurts the first person. Higher means safer compatibility.",
        1.1,
    ),
    (
        "desired_actual",
        "ideal_relationship",
        "relationship_behaviour",
        "directional",
        "Estimate how well the second person's actual relationship behaviour matches the first person's ideal relationship.",
        1.1,
    ),
    (
        "date_lifestyle",
        "ideal_date",
        "usual_weekend",
        "directional",
        "Estimate how well the second person's usual lifestyle fits the first person's ideal date.",
        0.9,
    ),
]

ANSWER_ALIASES = {
    "conflict_style": ("conflict_style", "conflict_immediate", "repair_style"),
    "conflict_behaviour": ("conflict_behaviour", "conflict_immediate", "repair_style"),
    "relationship_view": ("relationship_view", "future_vision", "desired_feeling"),
    "communication_style": ("communication_style",),
    "usual_weekend": ("usual_weekend", "ordinary_weekend", "weekend"),
    "care_need": ("care_need", "care_target"),
    "care_give": ("care_give", "care_self"),
    "communication_need": ("communication_need",),
    "communication_give": ("communication_give",),
    "emotional_need": ("emotional_need", "desired_feeling"),
    "emotional_support": ("emotional_support", "care_self"),
    "what_hurts_me": ("what_hurts_me", "red_flags"),
    "ideal_relationship": ("ideal_relationship", "future_vision"),
    "relationship_behaviour": ("relationship_behaviour", "care_self", "about"),
    "ideal_date": ("ideal_date",),
}

RELATIONSHIP_GOAL_COMPATIBILITY = {
    frozenset(("serious_relationship", "long_term_partnership")),
    frozenset(("serious_relationship", "family_perspective")),
    frozenset(("long_term_partnership", "family_perspective")),
    frozenset(("long_term", "marriage")),
    frozenset(("long_term", "open_to_options")),
    frozenset(("casual", "open_to_options")),
}


# ---------------------------------------------------------------------------
# The four meaningful result/input structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HardPreferences:
    """Canonical hard constraints accepted by the matching engine.

    Questionnaire/UI values must be normalized into this structure before the
    matching call.  ``None`` means that the user has no hard restriction for
    that field; soft preferences do not belong here.
    """

    min_age: int = 18
    max_age: int = 99
    accepted_genders: frozenset[str] | None = None
    accepted_cities: frozenset[str] | None = None
    accepted_relationship_goals: frozenset[str] | None = None
    allowed_smoking: frozenset[str] | None = None
    allowed_alcohol: frozenset[str] | None = None
    allowed_nightlife: frozenset[str] | None = None
    allowed_existing_children: frozenset[bool] | None = None
    extra_allowed: dict[str, frozenset[Any]] = field(default_factory=dict)


@dataclass(slots=True)
class UserProfile:
    user_id: Any
    age: int | None = None
    gender: str | None = None
    city: str | None = None
    relationship_goal: str | None = None
    hard_preferences: HardPreferences = field(default_factory=HardPreferences)
    structured: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, dict[str, float]] = field(default_factory=dict)
    answers: dict[str, Any] = field(default_factory=dict)
    embeddings: dict[str, list[float]] = field(default_factory=dict)


@dataclass(slots=True)
class TestResult:
    name: str
    score: float | None
    passed: bool | None = None
    skipped: bool = False
    details: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MatchResult:
    eligible: bool
    hard_filters: list[TestResult]
    groups: dict[str, dict[str, Any]]
    bonuses: list[dict[str, Any]]
    penalties: list[dict[str, Any]]
    base_score: float
    bonus_total: float
    penalty_total: float
    final_score: float
    coverage: float


EmbeddingFunction = Callable[[str], Sequence[float]] # тип для функции, которая принимает строку и возвращает последовательность чисел (вектор)
CrossFunction = Callable[..., float] # тип для функции, которая принимает произвольные аргументы и возвращает число (скоринг совместимости)
_USE_DEFAULT_CROSS = object() # дефольтный объект для проверки, был ли передан cross_fn или нет


# ---------------------------------------------------------------------------
# Small mathematical primitives
# ---------------------------------------------------------------------------

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def scalar_similarity(a: float, b: float) -> float:
    """Quadratic similarity for values on the inclusive 0..100 scale."""
    difference = abs(float(a) - float(b)) / 100.0
    return clamp(1.0 - difference**2)


def geometric_mean(a: float, b: float) -> float:
    return sqrt(clamp(a) * clamp(b))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return ordinary cosine similarity in -1..1; reject malformed vectors."""

    if not a or not b or len(a) != len(b):
        raise ValueError("embedding vectors must be non-empty and have equal length")

    dot = sum(float(x) * float(y) for x, y in zip(a, b)) # вычисляем скалярное произведение двух векторов
    norm_a = sqrt(sum(float(x) ** 2 for x in a)) # вычисляем длину вектора a
    norm_b = sqrt(sum(float(y) ** 2 for y in b)) # вычисляем длину вектора b

    if norm_a == 0 or norm_b == 0:
        raise ValueError("embedding vectors must not be zero vectors")

    return clamp(dot / (norm_a * norm_b), -1.0, 1.0)


def embedding_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Map cosine similarity from -1..1 to the matching score scale 0..1."""
    return (cosine_similarity(a, b) + 1.0) / 2.0


def categorical_similarity(
    a: str,
    b: str,
    matrix: Mapping[tuple[str, str], float],
) -> float | None:
    if (a, b) in matrix:
        return clamp(float(matrix[(a, b)]))
    if (b, a) in matrix:
        return clamp(float(matrix[(b, a)]))
    return None


def _weighted_test_group(
    tests_with_weights: list[tuple[TestResult, float]],
    configured_weight: float,
) -> dict[str, Any]:
    total_weight = sum(weight for _, weight in tests_with_weights)
    available = [
        (test, weight)
        for test, weight in tests_with_weights
        if not test.skipped and test.score is not None
    ]
    available_weight = sum(weight for _, weight in available)
    score = (
        sum(float(test.score) * weight for test, weight in available) / available_weight
        if available_weight
        else None
    )
    return {
        "score": score,
        "weight": configured_weight,
        "effective_weight": 0.0,
        "weighted_score": 0.0,
        "coverage": available_weight / total_weight if total_weight else 0.0,
        "tests": [test for test, _ in tests_with_weights],
    }


# ---------------------------------------------------------------------------
# Hard filters -- always run first and always evaluate both directions
# ---------------------------------------------------------------------------

def _hard_result(name: str, passed: bool, details: str, **data: Any) -> TestResult:
    return TestResult(name, 1.0 if passed else 0.0, passed, False, details, data)


def _goal_accepts(owner: UserProfile, partner: UserProfile) -> bool:
    """Check if the owner's hard preferences accept the partner's relationship goal or if the goals are same"""
    accepted = owner.hard_preferences.accepted_relationship_goals
    if accepted is not None: # проверяем, есть ли у владельца профиля список целей отношений, которые он принимает
        return partner.relationship_goal in accepted
    if owner.relationship_goal == partner.relationship_goal: # если список целей отношений владельца профиля пуст, проверяем, совпадают ли цели отношений владельца и партнера
        return True
    return frozenset((owner.relationship_goal, partner.relationship_goal)) in RELATIONSHIP_GOAL_COMPATIBILITY


def _allows(allowed: frozenset[Any] | None, actual: Any) -> tuple[bool, str]:
    if allowed is None:
        return True, "no hard restriction"
    if actual is None:
        return False, "partner value is missing"
    return actual in allowed, f"allows {sorted(map(str, allowed))}; partner={actual!r}"


def run_hard_filters(a: UserProfile, b: UserProfile) -> list[TestResult]:
    tests: list[TestResult] = []

    tests.append(_hard_result("different_users", a.user_id != b.user_id, f"{a.user_id!r} vs {b.user_id!r}"))

    adults = a.age is not None and b.age is not None and a.age >= 18 and b.age >= 18
    tests.append(_hard_result("adulthood", adults, f"ages: A={a.age!r}, B={b.age!r}"))

    gender_data_present = a.gender is not None and b.gender is not None
    accepted_by_a = a.hard_preferences.accepted_genders
    accepted_by_b = b.hard_preferences.accepted_genders
    a_accepts = gender_data_present and (accepted_by_a is None or b.gender in accepted_by_a)
    b_accepts = gender_data_present and (accepted_by_b is None or a.gender in accepted_by_b)
    tests.append(
        _hard_result(
            "gender",
            bool(a_accepts and b_accepts),
            f"A accepts B={a_accepts}; B accepts A={b_accepts}",
            a_to_b=bool(a_accepts),
            b_to_a=bool(b_accepts),
        )
    )

    cities_a = a.hard_preferences.accepted_cities
    cities_b = b.hard_preferences.accepted_cities
    city_data_present = a.city is not None and b.city is not None
    a_accepts_city = city_data_present and (b.city in cities_a if cities_a is not None else a.city == b.city)
    b_accepts_city = city_data_present and (a.city in cities_b if cities_b is not None else a.city == b.city)
    tests.append(
        _hard_result(
            "city",
            bool(a_accepts_city and b_accepts_city),
            f"A={a.city!r}, B={b.city!r}; A accepts B={a_accepts_city}; B accepts A={b_accepts_city}",
            a_to_b=bool(a_accepts_city),
            b_to_a=bool(b_accepts_city),
        )
    )

    a_min_age = a.hard_preferences.min_age
    a_max_age = a.hard_preferences.max_age
    b_min_age = b.hard_preferences.min_age
    b_max_age = b.hard_preferences.max_age
    a_accepts_age = b.age is not None and a_min_age <= b.age <= a_max_age
    b_accepts_age = a.age is not None and b_min_age <= a.age <= b_max_age
    tests.append(
        _hard_result(
            "age_range",
            a_accepts_age and b_accepts_age,
            f"A accepts {a_min_age}..{a_max_age}, B age={b.age!r}: {a_accepts_age}; "
            f"B accepts {b_min_age}..{b_max_age}, A age={a.age!r}: {b_accepts_age}",
            a_to_b=a_accepts_age,
            b_to_a=b_accepts_age,
        )
    )

    goal_data_present = a.relationship_goal is not None and b.relationship_goal is not None
    goal_a_to_b = goal_data_present and _goal_accepts(a, b)
    goal_b_to_a = goal_data_present and _goal_accepts(b, a)
    tests.append(
        _hard_result(
            "relationship_goal",
            bool(goal_a_to_b and goal_b_to_a),
            f"A={a.relationship_goal!r}, B={b.relationship_goal!r}; "
            f"A accepts B={goal_a_to_b}; B accepts A={goal_b_to_a}",
            a_to_b=bool(goal_a_to_b),
            b_to_a=bool(goal_b_to_a),
        )
    )

    lifestyle_fields = [
        (
            "smoking",
            a.hard_preferences.allowed_smoking,
            b.structured.get("smoking"),
            b.hard_preferences.allowed_smoking,
            a.structured.get("smoking"),
        ),
        (
            "alcohol",
            a.hard_preferences.allowed_alcohol,
            b.structured.get("alcohol"),
            b.hard_preferences.allowed_alcohol,
            a.structured.get("alcohol"),
        ),
        (
            "nightlife",
            a.hard_preferences.allowed_nightlife,
            b.structured.get("nightlife"),
            b.hard_preferences.allowed_nightlife,
            a.structured.get("nightlife"),
        ),
        (
            "existing_children",
            a.hard_preferences.allowed_existing_children,
            b.structured.get("has_children"),
            b.hard_preferences.allowed_existing_children,
            a.structured.get("has_children"),
        ),
    ]
    for field_name, a_allowed, b_actual, b_allowed, a_actual in lifestyle_fields:
        a_ok, a_detail = _allows(a_allowed, b_actual)
        b_ok, b_detail = _allows(b_allowed, a_actual)
        tests.append(
            _hard_result(
                field_name,
                a_ok and b_ok,
                f"A→B: {a_detail}; B→A: {b_detail}",
                a_to_b=a_ok,
                b_to_a=b_ok,
            )
        )

    for owner, partner, direction in ((a, b, "a_to_b"), (b, a, "b_to_a")):
        for name, allowed in owner.hard_preferences.extra_allowed.items():
            actual = partner.structured.get(name)
            passed, detail = _allows(allowed, actual)
            tests.append(_hard_result(f"custom:{name}:{direction}", passed, detail))

    return tests


# ---------------------------------------------------------------------------
# Structured, tags and embeddings
# ---------------------------------------------------------------------------

def _structured_value(profile: UserProfile, field_name: str) -> Any:
    for candidate in STRUCTURED_ALIASES.get(field_name, (field_name,)):
        if candidate in profile.structured and profile.structured[candidate] is not None:
            return profile.structured[candidate]
    return None


def run_structured_tests(a: UserProfile, b: UserProfile) -> dict[str, Any]:
    weighted: list[tuple[TestResult, float]] = []

    for name, field_name, weight in STRUCTURED_SIMILARITY_TESTS:
        value_a, value_b = _structured_value(a, field_name), _structured_value(b, field_name)
        if value_a is None or value_b is None:
            result = TestResult(name, None, skipped=True, details="missing scalar value")
        else:
            score = scalar_similarity(value_a, value_b)
            result = TestResult(name, score, details=f"A={value_a}, B={value_b}", data={"a": value_a, "b": value_b})
        weighted.append((result, weight))

    for name, need_field, give_field, weight in STRUCTURED_DIRECTIONAL_TESTS:
        a_need = _structured_value(a, need_field)
        a_give = _structured_value(a, give_field)
        b_need = _structured_value(b, need_field)
        b_give = _structured_value(b, give_field)
        if None in (a_need, a_give, b_need, b_give):
            result = TestResult(name, None, skipped=True, details="missing need/give value")
        else:
            a_to_b = scalar_similarity(a_need, b_give)
            b_to_a = scalar_similarity(b_need, a_give)
            combined = geometric_mean(a_to_b, b_to_a)
            result = TestResult(
                name,
                combined,
                details=f"A→B={a_to_b:.3f}; B→A={b_to_a:.3f}; geometric mean={combined:.3f}",
                data={"a_to_b": a_to_b, "b_to_a": b_to_a, "combined": combined},
            )
        weighted.append((result, weight))

    for name, field_name, matrix, weight in CATEGORICAL_TESTS:
        value_a, value_b = _structured_value(a, field_name), _structured_value(b, field_name)
        score = None if value_a is None or value_b is None else categorical_similarity(value_a, value_b, matrix)
        if score is None:
            result = TestResult(name, None, skipped=True, details="missing value or unknown category pair")
        else:
            result = TestResult(name, score, details=f"A={value_a!r}, B={value_b!r}", data={"a": value_a, "b": value_b})
        weighted.append((result, weight))

    return _weighted_test_group(weighted, GROUP_WEIGHTS["structured"])


def relatedness(
    tag_a: str,
    tag_b: str,
    relationships: Mapping[tuple[str, str], float] = TAG_RELATEDNESS,
    families: Mapping[str, str] = TAG_FAMILIES,
) -> tuple[float, str]:
    if tag_a == tag_b:
        return 1.0, "exact"
    if (tag_a, tag_b) in relationships:
        return clamp(float(relationships[(tag_a, tag_b)])), "related"
    if (tag_b, tag_a) in relationships:
        return clamp(float(relationships[(tag_b, tag_a)])), "related"
    if tag_a in families and tag_b in families and families[tag_a] == families[tag_b]:
        return 0.55, "same_family"
    return 0.0, "unrelated"


def compare_tag_sets(
    tags_a: Mapping[str, float],
    tags_b: Mapping[str, float],
    relationships: Mapping[tuple[str, str], float] = TAG_RELATEDNESS,
    families: Mapping[str, str] = TAG_FAMILIES,
) -> TestResult:
    """Compare every related tag pair and sum their contributions."""

    clean_a = {
        tag: float(weight)
        for tag, weight in tags_a.items()
        if float(weight) > 0
    }
    clean_b = {
        tag: float(weight)
        for tag, weight in tags_b.items()
        if float(weight) > 0
    }

    if not clean_a or not clean_b:
        return TestResult(
            "tags",
            None,
            skipped=True,
            details="one or both tag sets are empty",
        )

    intersection = 0.0
    contributions: list[dict[str, Any]] = []

    for tag_a, weight_a in clean_a.items():
        for tag_b, weight_b in clean_b.items():
            relation, kind = relatedness(
                tag_a,
                tag_b,
                relationships,
                families,
            )

            contribution = relation * min(weight_a, weight_b)

            if contribution <= 0:
                continue

            intersection += contribution

            contributions.append(
                {
                    "a_tag": tag_a,
                    "b_tag": tag_b,
                    "kind": kind,
                    "relatedness": relation,
                    "contribution": contribution,
                }
            )

    union = sum(clean_a.values()) + sum(clean_b.values()) - intersection
    score = intersection / union if union > 0 else 0.0

    exact = [
        item
        for item in contributions
        if item["kind"] == "exact"
    ]

    related = [
        item
        for item in contributions
        if item["kind"] != "exact"
    ]

    return TestResult(
        "tags",
        clamp(score),
        details=f"fuzzy intersection={intersection:.3f}, union={union:.3f}",
        data={
            "exact_matches": exact,
            "related_matches": related,
            "contributions": contributions,
        },
    )


def run_tag_tests(
    a: UserProfile,
    b: UserProfile,
    relationships: Mapping[tuple[str, str], float] = TAG_RELATEDNESS,
    families: Mapping[str, str] = TAG_FAMILIES,
) -> dict[str, Any]:
    categories = sorted(set(a.tags) | set(b.tags) | {"interests"})
    weighted: list[tuple[TestResult, float]] = []
    for category in categories:
        result = compare_tag_sets(a.tags.get(category, {}), b.tags.get(category, {}), relationships, families)
        result.name = f"tags_{category}"
        weighted.append((result, 1.0))
    return _weighted_test_group(weighted, GROUP_WEIGHTS["tags"])


def run_embedding_tests(
    a_embeddings: Mapping[str, Sequence[float]],
    b_embeddings: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    weighted: list[tuple[TestResult, float]] = []
    for field_name, weight in EMBEDDING_FIELDS:
        vector_a, vector_b = a_embeddings.get(field_name), b_embeddings.get(field_name)
        if vector_a is None or vector_b is None:
            result = TestResult(
                f"embedding_{field_name}", None, skipped=True, details="missing embedding"
            )
        else:
            try:
                raw_cosine = cosine_similarity(vector_a, vector_b)
                score = clamp((raw_cosine + 1.0) / 2.0)
                result = TestResult(
                    f"embedding_{field_name}",
                    score,
                    details=f"cosine={raw_cosine:.3f}; normalized={score:.3f}",
                    data={"a_field": field_name, "b_field": field_name, "cosine": raw_cosine},
                )
            except (TypeError, ValueError) as error:
                result = TestResult(
                    f"embedding_{field_name}", None, skipped=True, details=f"invalid embedding: {error}"
                )
        weighted.append((result, weight))
    return _weighted_test_group(weighted, GROUP_WEIGHTS["embeddings"])


# ---------------------------------------------------------------------------
# Cross-encoder tests
# ---------------------------------------------------------------------------

def _answer(profile: UserProfile, field_name: str) -> str | None:
    values: list[str] = []
    for candidate in ANSWER_ALIASES.get(field_name, (field_name,)):
        value = profile.answers.get(candidate)
        if value is None or value == "":
            continue
        text = str(value).strip()
        if text and text not in values:
            values.append(text)
    return "\n".join(values) if values else None


def _cross_call(cross_fn: CrossFunction, text_a: str, text_b: str, task: str) -> float:
    value = cross_fn(text_a, text_b, task=task)
    return clamp(float(value))


def _contextual_task(task: str, left_label: str, right_label: str) -> str:
    return f"{left_label} / {right_label}. {task} Return compatibility from 0 to 1."


def run_cross_tests(a: UserProfile, b: UserProfile, cross_fn: CrossFunction | None) -> dict[str, Any]:
    weighted: list[tuple[TestResult, float]] = []
    for name, left_field, right_field, mode, task, weight in CROSS_TESTS:
        if cross_fn is None:
            result = TestResult(name, None, skipped=True, details="cross-encoder disabled")
        elif mode == "symmetric":
            text_a, text_b = _answer(a, left_field), _answer(b, right_field)
            if text_a is None or text_b is None:
                result = TestResult(name, None, skipped=True, details="missing answer")
            else:
                try:
                    score = _cross_call(
                        cross_fn,
                        text_a,
                        text_b,
                        _contextual_task(task, f"PERSON A {left_field}", f"PERSON B {right_field}"),
                    )
                    result = TestResult(
                        name,
                        score,
                        details=f"symmetric {left_field} × {right_field}",
                        data={"a_field": left_field, "b_field": right_field},
                    )
                except Exception as error:  # model failure becomes a skipped test, not false incompatibility
                    result = TestResult(name, None, skipped=True, details=f"cross-encoder error: {error}")
        else:
            a_need, b_give = _answer(a, left_field), _answer(b, right_field)
            b_need, a_give = _answer(b, left_field), _answer(a, right_field)
            if None in (a_need, b_give, b_need, a_give):
                result = TestResult(name, None, skipped=True, details="missing directional answer")
            else:
                try:
                    a_to_b = _cross_call(
                        cross_fn,
                        a_need,
                        b_give,
                        _contextual_task(task, f"PERSON A {left_field}", f"PERSON B {right_field}"),
                    )
                    b_to_a = _cross_call(
                        cross_fn,
                        b_need,
                        a_give,
                        _contextual_task(task, f"PERSON B {left_field}", f"PERSON A {right_field}"),
                    )
                    combined = geometric_mean(a_to_b, b_to_a)
                    result = TestResult(
                        name,
                        combined,
                        details=f"A→B={a_to_b:.3f}; B→A={b_to_a:.3f}; geometric mean={combined:.3f}",
                        data={"a_to_b": a_to_b, "b_to_a": b_to_a, "combined": combined},
                    )
                except Exception as error:
                    result = TestResult(name, None, skipped=True, details=f"cross-encoder error: {error}")
        weighted.append((result, weight))
    return _weighted_test_group(weighted, GROUP_WEIGHTS["cross"])


# ---------------------------------------------------------------------------
# Adjustments and final composition
# ---------------------------------------------------------------------------

def _exact_interest_names(tag_group: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for test in tag_group["tests"]:
        if test.name == "tags_interests" and not test.skipped:
            names.update(item["a_tag"] for item in test.data.get("exact_matches", []))
    return names


def calculate_adjustments(
    groups: Mapping[str, dict[str, Any]],
    rare_interests: set[str] = RARE_INTERESTS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bonuses: list[dict[str, Any]] = []
    penalties: list[dict[str, Any]] = []

    exact_interests = _exact_interest_names(groups["tags"])
    rare_shared = sorted(exact_interests & rare_interests)
    if rare_shared:
        bonuses.append(
            {"name": "shared_rare_interest", "value": 0.03, "details": f"shared: {', '.join(rare_shared)}"}
        )
    if len(exact_interests) >= 2:
        bonuses.append(
            {
                "name": "multiple_exact_interests",
                "value": min(0.02, 0.005 * len(exact_interests)),
                "details": f"{len(exact_interests)} exact interests",
            }
        )

    lifestyle_scores = [
        test.score
        for test in groups["structured"]["tests"]
        if test.name in {"home_outdoor", "calm_intense", "planned_spontaneous", "private_social"}
        and test.score is not None
    ]
    if lifestyle_scores and min(lifestyle_scores) < 0.50:
        penalties.append(
            {
                "name": "soft_lifestyle_mismatch",
                "value": 0.02,
                "details": f"lowest lifestyle similarity={min(lifestyle_scores):.3f}",
            }
        )

    children = next(
        (test for test in groups["structured"]["tests"] if test.name == "future_children"), None
    )
    if children and children.score is not None and children.score < 0.50:
        penalties.append(
            {
                "name": "children_preference_tension",
                "value": 0.02,
                "details": f"compatibility={children.score:.3f}",
            }
        )
    return bonuses, penalties


def finalize_score(
    base_score: float,
    bonuses: Iterable[Mapping[str, Any]],
    penalties: Iterable[Mapping[str, Any]],
    max_total_bonus: float = MAX_TOTAL_BONUS,
    max_total_penalty: float = MAX_TOTAL_PENALTY,
) -> tuple[float, float, float]:
    bonus_total = min(max_total_bonus, sum(max(0.0, float(item["value"])) for item in bonuses))
    penalty_total = min(max_total_penalty, sum(max(0.0, float(item["value"])) for item in penalties))
    return clamp(base_score + bonus_total - penalty_total), bonus_total, penalty_total


def _empty_groups(reason: str) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "score": None,
            "weight": weight,
            "effective_weight": 0.0,
            "weighted_score": 0.0,
            "coverage": 0.0,
            "tests": [],
            "details": reason,
        }
        for name, weight in GROUP_WEIGHTS.items()
    }


def compare_users(
    a: UserProfile,
    b: UserProfile,
    *,
    cross_fn: CrossFunction | None | object = _USE_DEFAULT_CROSS,
    auto_generate_embeddings: bool = False,
    embed_fn: EmbeddingFunction | None = None,
    tag_relationships: Mapping[tuple[str, str], float] = TAG_RELATEDNESS,
    tag_families: Mapping[str, str] = TAG_FAMILIES,
    rare_interests: set[str] = RARE_INTERESTS,
) -> MatchResult:
    """Compare two profiles and return a complete, explainable breakdown.

    By default cross tests use the local ``cross_score`` wrapper.  Pass
    ``cross_fn=None`` to skip them, or inject a compatible fake in tests.
    On-demand embedding generation is opt-in so production comparisons never
    unexpectedly recompute stored vectors.
    """
    hard_filters = run_hard_filters(a, b)
    eligible = all(test.passed is True for test in hard_filters)
    if not eligible:
        return MatchResult(
            eligible=False,
            hard_filters=hard_filters,
            groups=_empty_groups("not evaluated because a hard filter failed"),
            bonuses=[],
            penalties=[],
            base_score=0.0,
            bonus_total=0.0,
            penalty_total=0.0,
            final_score=0.0,
            coverage=0.0,
        )

    embeddings_a: Mapping[str, Sequence[float]] = a.embeddings
    embeddings_b: Mapping[str, Sequence[float]] = b.embeddings
    if auto_generate_embeddings:
        generator = embed_fn or embed
        embeddings_a = {**generate_embeddings(a, generator), **a.embeddings}
        embeddings_b = {**generate_embeddings(b, generator), **b.embeddings}

    resolved_cross_fn = cross_score if cross_fn is _USE_DEFAULT_CROSS else cross_fn
    groups = {
        "structured": run_structured_tests(a, b),
        "tags": run_tag_tests(a, b, tag_relationships, tag_families),
        "embeddings": run_embedding_tests(embeddings_a, embeddings_b),
        "cross": run_cross_tests(a, b, resolved_cross_fn),
    }

    available_groups = [group for group in groups.values() if group["score"] is not None]
    active_weight = sum(group["weight"] for group in available_groups)
    for group in groups.values():
        if group["score"] is not None and active_weight:
            group["effective_weight"] = group["weight"] / active_weight
            group["weighted_score"] = group["score"] * group["effective_weight"]
    base_score = sum(group["weighted_score"] for group in groups.values())

    bonuses, penalties = calculate_adjustments(groups, rare_interests)
    final_score, bonus_total, penalty_total = finalize_score(base_score, bonuses, penalties)

    coverage = sum(group["coverage"] * group["weight"] for group in groups.values())
    return MatchResult(
        eligible=True,
        hard_filters=hard_filters,
        groups=groups,
        bonuses=bonuses,
        penalties=penalties,
        base_score=base_score,
        bonus_total=bonus_total,
        penalty_total=penalty_total,
        final_score=final_score,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Embedding and cross-encoder wrappers (lazy, optional local dependency)
# ---------------------------------------------------------------------------

_embedding_model: Any = None
_cross_model: Any = None


def embed(text: str) -> list[float]:
    """Embed one text with a small multilingual sentence-transformer model."""
    global _embedding_model
    if not text or not text.strip():
        raise ValueError("cannot embed empty text")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "embed() requires the optional 'sentence-transformers' package; "
            "install it or pass a custom embed_fn"
        ) from error
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector = _embedding_model.encode(text, normalize_embeddings=True)
    return [float(value) for value in vector]


def generate_embeddings(
    profile: UserProfile,
    embed_fn: EmbeddingFunction = embed,
) -> dict[str, list[float]]:
    """Return only missing per-area embeddings; never creates one profile vector."""
    generated: dict[str, list[float]] = {}
    for semantic_field, answer_fields in SEMANTIC_TEXT_FIELDS.items():
        if semantic_field in profile.embeddings:
            continue
        parts = [str(profile.answers[field]).strip() for field in answer_fields if profile.answers.get(field)]
        text = "\n".join(part for part in parts if part)
        if text:
            generated[semantic_field] = [float(value) for value in embed_fn(text)]
    return generated


def cross_score(text_a: str, text_b: str, *, task: str) -> float:
    """Run a local reranker-style cross-encoder and normalize its logit to 0..1."""
    global _cross_model
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RuntimeError(
            "cross_score() requires the optional 'sentence-transformers' package; "
            "install it or pass a custom cross_fn"
        ) from error
    if _cross_model is None:
        _cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)

    query = f"TASK:\n{task}\n\nPERSON A:\n{text_a}"
    passage = f"PERSON B:\n{text_b}"
    raw = float(_cross_model.predict([(query, passage)])[0])
    # This model returns an unbounded relevance logit, not a probability.
    return 1.0 / (1.0 + exp(-max(-60.0, min(60.0, raw))))


# ---------------------------------------------------------------------------
# Human-readable report and editable demonstration
# ---------------------------------------------------------------------------

def format_match_report(result: MatchResult) -> str:
    lines = [
        f"MATCH SCORE: {result.final_score:.1%}",
        f"COVERAGE: {result.coverage:.1%}",
        f"ELIGIBLE: {'yes' if result.eligible else 'no'}",
        "",
        "HARD FILTERS",
    ]
    for test in result.hard_filters:
        marker = "✓" if test.passed else "✗"
        lines.append(f"{marker} {test.name:<28} {test.details}")

    for group_name in ("structured", "tags", "embeddings", "cross"):
        group = result.groups[group_name]
        score_text = "skipped" if group["score"] is None else f"{group['score']:.3f}"
        lines.extend(
            [
                "",
                f"{group_name.upper()}: {score_text}  (coverage {group['coverage']:.1%}, "
                f"effective weight {group['effective_weight']:.1%}, contribution {group['weighted_score']:.3f})",
            ]
        )
        for test in group["tests"]:
            if test.skipped:
                lines.append(f"  – {test.name:<30} skipped: {test.details}")
                continue
            lines.append(f"  {test.name:<32} {test.score:.3f}")
            if "a_to_b" in test.data:
                lines.append(f"    A → B {test.data['a_to_b']:.3f}; B → A {test.data['b_to_a']:.3f}")
            for contribution in test.data.get("contributions", []):
                lines.append(
                    f"    {contribution['a_tag']} ↔ {contribution['b_tag']} "
                    f"{contribution['relatedness']:.2f} ({contribution['kind']})"
                )

    lines.extend(["", "BONUSES"])
    lines.extend(
        [f"  {item['name']:<32} +{item['value']:.3f}  {item['details']}" for item in result.bonuses]
        or ["  none"]
    )
    lines.extend(["", "PENALTIES"])
    lines.extend(
        [f"  {item['name']:<32} -{item['value']:.3f}  {item['details']}" for item in result.penalties]
        or ["  none"]
    )
    lines.extend(
        [
            "",
            f"BASE: {result.base_score:.3f}",
            f"APPLIED BONUS: +{result.bonus_total:.3f} (cap {MAX_TOTAL_BONUS:.3f})",
            f"APPLIED PENALTY: -{result.penalty_total:.3f} (cap {MAX_TOTAL_PENALTY:.3f})",
            f"FINAL: {result.final_score:.3f}",
        ]
    )
    return "\n".join(lines)


def print_match_report(result: MatchResult) -> None:
    print(format_match_report(result))


def _demo_cross_score(text_a: str, text_b: str, *, task: str) -> float:
    """Deterministic stand-in used only by `python matching.py`."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    overlap = len(words_a & words_b) / max(1, len(words_a | words_b))
    return 0.62 + 0.30 * overlap


def _demo_profiles() -> tuple[UserProfile, UserProfile]:
    common_a = {
        "home_outdoor": 45,
        "calm_intense": 35,
        "planned_spontaneous": 55,
        "private_social": 40,
        "closeness": 75,
        "messaging_frequency": 70,
        "initiative_target": 65,
        "initiative_self": 55,
        "tactility": 80,
        "future_children": "probably_yes",
        "smoking": "never",
        "alcohol": "rarely",
        "nightlife": "rarely",
        "has_children": False,
    }
    common_b = {
        "home_outdoor": 55,
        "calm_intense": 40,
        "planned_spontaneous": 65,
        "private_social": 50,
        "closeness": 80,
        "messaging_frequency": 60,
        "initiative_target": 50,
        "initiative_self": 70,
        "tactility": 75,
        "future_children": "definitely_yes",
        "smoking": "never",
        "alcohol": "moderately",
        "nightlife": "sometimes",
        "has_children": False,
    }
    a = UserProfile(
        user_id="anna",
        age=29,
        gender="female",
        city="moscow",
        relationship_goal="serious_relationship",
        hard_preferences=HardPreferences(
            min_age=26,
            max_age=36,
            accepted_genders=frozenset({"male"}),
            allowed_smoking=frozenset({"never"}),
        ),
        structured=common_a,
        tags={"interests": {"anime": 1.0, "programming": 0.9, "astronomy": 0.8}},
        answers={
            "ordinary_weekend": "A walk, a film and a quiet dinner with friends.",
            "ideal_date": "A long walk and dinner where we can talk.",
            "care_target": "Stay nearby, listen and ask what would help.",
            "care_self": "I listen, hug and offer practical help.",
            "conflict_immediate": "Talk calmly after a short pause.",
            "repair_style": "Apologise directly and agree on a solution.",
            "future_vision": "A warm stable partnership with a family.",
            "desired_feeling": "Safe, heard and close.",
            "red_flags": "Silent punishment and shouting hurt me.",
        },
        embeddings={
            "relationship": [0.9, 0.2, 0.1],
            "conflict": [0.8, 0.1, 0.3],
            "care": [0.7, 0.5, 0.1],
            "weekend": [0.5, 0.7, 0.2],
            "emotional_needs": [0.8, 0.4, 0.1],
        },
    )
    b = UserProfile(
        user_id="boris",
        age=31,
        gender="male",
        city="moscow",
        relationship_goal="family_perspective",
        hard_preferences=HardPreferences(
            min_age=25,
            max_age=34,
            accepted_genders=frozenset({"female"}),
        ),
        structured=common_b,
        tags={"interests": {"manga": 0.9, "programming": 0.8, "astronomy": 1.0}},
        answers={
            "ordinary_weekend": "A walk, cooking dinner and a good film.",
            "ideal_date": "Coffee, a walk and an honest conversation.",
            "care_target": "Listen without fixing everything and stay close.",
            "care_self": "I stay nearby, listen and ask before giving advice.",
            "conflict_immediate": "Take a short pause, then discuss it calmly.",
            "repair_style": "Own my part and find a concrete solution.",
            "future_vision": "A stable partnership and children in the future.",
            "desired_feeling": "Trusted, calm and understood.",
            "red_flags": "Contempt and refusing to speak hurt me.",
        },
        embeddings={
            "relationship": [0.85, 0.25, 0.1],
            "conflict": [0.75, 0.15, 0.35],
            "care": [0.65, 0.55, 0.1],
            "weekend": [0.55, 0.65, 0.2],
            "emotional_needs": [0.75, 0.45, 0.1],
        },
    )
    return a, b


if __name__ == "__main__":
    demo_a, demo_b = _demo_profiles()
    print_match_report(compare_users(demo_a, demo_b, cross_fn=_demo_cross_score))
