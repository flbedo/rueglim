import unittest

from matching import (
    GROUP_WEIGHTS,
    HardPreferences,
    TestResult,
    UserProfile,
    compare_tag_sets,
    compare_users,
    cosine_similarity,
    embedding_similarity,
    finalize_score,
    generate_embeddings,
    geometric_mean,
    relatedness,
    run_cross_tests,
    run_hard_filters,
    scalar_similarity,
)


def preferences(user_id: str, **changes) -> HardPreferences:
    data = {
        "min_age": 25,
        "max_age": 35,
        "accepted_genders": frozenset({"male" if user_id == "a" else "female"}),
    }
    data.update(changes)
    return HardPreferences(**data)


def profile(user_id: str, **changes) -> UserProfile:
    data = {
        "user_id": user_id,
        "age": 30,
        "gender": "female" if user_id == "a" else "male",
        "city": "moscow",
        "relationship_goal": "serious_relationship",
        "hard_preferences": preferences(user_id),
        "structured": {
            "smoking": "never",
            "alcohol": "rarely",
            "nightlife": "rarely",
            "has_children": False,
        },
        "tags": {},
        "answers": {},
        "embeddings": {},
    }
    data.update(changes)
    return UserProfile(**data)


def hard_test(a: UserProfile, b: UserProfile, name: str) -> TestResult:
    return next(test for test in run_hard_filters(a, b) if test.name == name)


class HardFilterTests(unittest.TestCase):
    def test_age_passes_both_directions(self):
        test = hard_test(profile("a"), profile("b", age=34), "age_range")
        self.assertTrue(test.passed)
        self.assertTrue(test.data["a_to_b"])
        self.assertTrue(test.data["b_to_a"])

    def test_age_fails(self):
        result = compare_users(profile("a"), profile("b", age=40))
        self.assertFalse(result.eligible)
        self.assertEqual(result.final_score, 0.0)
        self.assertFalse(next(test for test in result.hard_filters if test.name == "age_range").passed)

    def test_age_is_mutual(self):
        a = profile("a", age=30, hard_preferences=preferences("a", min_age=20, max_age=50))
        b = profile("b", age=40, hard_preferences=preferences("b", min_age=35, max_age=45))
        test = hard_test(a, b, "age_range")
        self.assertTrue(test.data["a_to_b"])
        self.assertFalse(test.data["b_to_a"])
        self.assertFalse(test.passed)

    def test_city_mismatch(self):
        test = hard_test(profile("a"), profile("b", city="kazan"), "city")
        self.assertFalse(test.passed)

    def test_city_can_be_accepted_explicitly(self):
        a = profile(
            "a",
            hard_preferences=preferences("a", accepted_cities=frozenset({"kazan"})),
        )
        b = profile(
            "b",
            city="kazan",
            hard_preferences=preferences("b", accepted_cities=frozenset({"moscow"})),
        )
        self.assertTrue(hard_test(a, b, "city").passed)

    def test_hard_smoking_mismatch(self):
        a = profile(
            "a",
            hard_preferences=preferences("a", allowed_smoking=frozenset({"never"})),
        )
        b = profile("b")
        b.structured["smoking"] = "sometimes"
        self.assertFalse(hard_test(a, b, "smoking").passed)

    def test_absent_hard_smoking_preference_does_not_reject(self):
        a = profile("a")
        b = profile("b")
        b.structured["smoking"] = "sometimes"
        self.assertTrue(hard_test(a, b, "smoking").passed)

    def test_existing_children_hard_preference(self):
        a = profile(
            "a",
            hard_preferences=preferences("a", allowed_existing_children=frozenset({False})),
        )
        b = profile("b")
        b.structured["has_children"] = True
        self.assertFalse(hard_test(a, b, "existing_children").passed)

    def test_relationship_mismatch(self):
        a = profile("a", relationship_goal="long_term")
        b = profile("b", relationship_goal="casual")
        self.assertFalse(hard_test(a, b, "relationship_goal").passed)

    def test_gender_is_mutual(self):
        b = profile(
            "b",
            hard_preferences=preferences("b", accepted_genders=frozenset({"male"})),
        )
        test = hard_test(profile("a"), b, "gender")
        self.assertTrue(test.data["a_to_b"])
        self.assertFalse(test.data["b_to_a"])
        self.assertFalse(test.passed)

    def test_custom_constraints_run_in_both_directions(self):
        a = profile("a")
        a.hard_preferences.extra_allowed["likes_cats"] = frozenset({True})
        b = profile("b")
        b.structured["likes_cats"] = False
        result = run_hard_filters(a, b)
        test = next(test for test in result if test.name == "custom:likes_cats:a_to_b")
        self.assertFalse(test.passed)

    def test_expensive_functions_are_not_called_after_hard_failure(self):
        called = {"cross": False, "embed": False}

        def forbidden_cross(*args, **kwargs):
            called["cross"] = True
            raise AssertionError("must not run")

        def forbidden_embed(text):
            called["embed"] = True
            raise AssertionError("must not run")

        result = compare_users(
            profile("a"),
            profile("b", city="kazan"),
            cross_fn=forbidden_cross,
            auto_generate_embeddings=True,
            embed_fn=forbidden_embed,
        )
        self.assertFalse(result.eligible)
        self.assertEqual(called, {"cross": False, "embed": False})


class StructuredTests(unittest.TestCase):
    def test_equal_scalar_values(self):
        self.assertEqual(scalar_similarity(50, 50), 1.0)

    def test_strongly_different_scalar_values(self):
        self.assertEqual(scalar_similarity(0, 100), 0.0)

    def test_geometric_mean_penalizes_one_sided_score(self):
        combined = geometric_mean(1.0, 0.25)
        self.assertEqual(combined, 0.5)
        self.assertLess(combined, (1.0 + 0.25) / 2)

    def test_directional_need_give_keeps_both_directions(self):
        a = profile("a")
        b = profile("b")
        a.structured.update({"need_initiative": 100, "give_initiative": 100})
        b.structured.update({"need_initiative": 100, "give_initiative": 0})
        result = compare_users(a, b)
        test = next(test for test in result.groups["structured"]["tests"] if test.name == "initiative")
        self.assertEqual(test.data["a_to_b"], 0.0)
        self.assertEqual(test.data["b_to_a"], 1.0)
        self.assertEqual(test.score, 0.0)

    def test_children_matrix_is_not_binary_equality(self):
        a = profile("a")
        b = profile("b")
        a.structured["future_children"] = "definitely_yes"
        b.structured["future_children"] = "probably_yes"
        result = compare_users(a, b)
        test = next(test for test in result.groups["structured"]["tests"] if test.name == "future_children")
        self.assertEqual(test.score, 0.85)


class TagTests(unittest.TestCase):
    def test_exact_tag(self):
        result = compare_tag_sets({"anime": 1.0}, {"anime": 1.0})
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.data["exact_matches"][0]["a_tag"], "anime")

    def test_related_tag(self):
        relation, kind = relatedness("anime", "manga")
        self.assertEqual((relation, kind), (0.75, "related"))
        result = compare_tag_sets({"anime": 1.0}, {"manga": 1.0})
        self.assertGreater(result.score, 0.0)
        self.assertEqual(result.data["related_matches"][0]["relatedness"], 0.75)

    def test_same_family_tag(self):
        relation, kind = relatedness("anime", "japanese_media")
        # There is an explicit relation, which takes precedence over family.
        self.assertEqual((relation, kind), (0.60, "related"))
        relation, kind = relatedness("gym", "running")
        self.assertEqual((relation, kind), (0.55, "same_family"))

    def test_unrelated_tags(self):
        result = compare_tag_sets({"anime": 1.0}, {"gardening": 1.0})
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.data["contributions"], [])

    def test_empty_tags_are_skipped_not_zero(self):
        result = compare_tag_sets({}, {"anime": 1.0})
        self.assertTrue(result.skipped)
        self.assertIsNone(result.score)


class EmbeddingTests(unittest.TestCase):
    def test_invalid_vectors_are_rejected(self):
        for left, right in (([], []), ([1], [1, 2]), ([0, 0], [1, 0])):
            with self.subTest(left=left, right=right):
                with self.assertRaises(ValueError):
                    cosine_similarity(left, right)

    def test_orthogonal_embedding_group_score(self):
        result = compare_users(
            profile("a", embeddings={"conflict": [1, 0]}),
            profile("b", embeddings={"conflict": [0, 1]}),
            cross_fn=None,
        )
        self.assertEqual(result.groups["embeddings"]["score"], 0.5)

    def test_cosine_for_equal_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 2], [1, 2]), 1.0)
        self.assertAlmostEqual(embedding_similarity([1, 2], [1, 2]), 1.0)

    def test_cosine_for_orthogonal_vectors(self):
        self.assertEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertEqual(embedding_similarity([1, 0], [0, 1]), 0.5)

    def test_cosine_for_opposite_vectors(self):
        self.assertEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)
        self.assertEqual(embedding_similarity([1, 0], [-1, 0]), 0.0)

    def test_generate_only_missing_semantic_embeddings(self):
        user = profile(
            "a",
            answers={"future_vision": "Stable partnership", "ordinary_weekend": "A long walk"},
            embeddings={"relationship": [9.0]},
        )
        seen = []

        def fake_embed(text):
            seen.append(text)
            return [float(len(text))]

        generated = generate_embeddings(user, fake_embed)
        self.assertNotIn("relationship", generated)
        self.assertIn("weekend", generated)
        self.assertEqual(len(seen), 1)

    def test_missing_embedding_is_skipped(self):
        a = profile("a", embeddings={"conflict": [1, 0]})
        b = profile("b", embeddings={})
        result = compare_users(a, b)
        test = next(test for test in result.groups["embeddings"]["tests"] if test.name == "embedding_conflict")
        self.assertTrue(test.skipped)
        self.assertIsNone(test.score)


class CrossEncoderTests(unittest.TestCase):
    @staticmethod
    def fake_cross(text_a, text_b, *, task):
        if "weak" in text_a or "weak" in text_b:
            return 0.25
        if "excellent" in text_a or "excellent" in text_b:
            return 1.0
        return 0.81

    def test_symmetric_cross_test(self):
        a = profile("a", answers={"conflict_style": "calm"})
        b = profile("b", answers={"conflict_style": "calm"})
        group = run_cross_tests(a, b, self.fake_cross)
        test = next(test for test in group["tests"] if test.name == "conflict")
        self.assertEqual(test.score, 0.81)
        self.assertFalse(test.skipped)

    def test_directional_cross_keeps_both_scores(self):
        a = profile("a", answers={"care_need": "excellent need", "care_give": "weak give"})
        b = profile("b", answers={"care_need": "ordinary need", "care_give": "excellent give"})
        group = run_cross_tests(a, b, self.fake_cross)
        test = next(test for test in group["tests"] if test.name == "care")
        self.assertEqual(test.data["a_to_b"], 1.0)
        self.assertEqual(test.data["b_to_a"], 0.25)
        self.assertEqual(test.score, 0.5)

    def test_missing_cross_answer_is_skipped(self):
        group = run_cross_tests(profile("a"), profile("b"), self.fake_cross)
        self.assertTrue(all(test.skipped for test in group["tests"]))
        self.assertIsNone(group["score"])
        self.assertEqual(group["coverage"], 0.0)

    def test_cross_task_contains_context(self):
        tasks = []

        def recording_cross(text_a, text_b, *, task):
            tasks.append(task)
            return 0.5

        a = profile("a", answers={"care_need": "need a", "care_give": "give a"})
        b = profile("b", answers={"care_need": "need b", "care_give": "give b"})
        run_cross_tests(a, b, recording_cross)
        care_tasks = [task for task in tasks if "care need" in task]
        self.assertEqual(len(care_tasks), 2)
        self.assertIn("PERSON A care_need", care_tasks[0])


class MissingDataAndFinalScoreTests(unittest.TestCase):
    def test_missing_structured_value_does_not_become_zero(self):
        result = compare_users(profile("a"), profile("b"))
        test = next(test for test in result.groups["structured"]["tests"] if test.name == "home_outdoor")
        self.assertTrue(test.skipped)
        self.assertIsNone(test.score)

    def test_partial_data_reduces_coverage_but_keeps_score(self):
        a = profile("a")
        b = profile("b")
        a.structured["home_outdoor"] = 50
        b.structured["home_outdoor"] = 50
        result = compare_users(a, b)
        self.assertEqual(result.groups["structured"]["score"], 1.0)
        self.assertGreater(result.groups["structured"]["coverage"], 0.0)
        self.assertLess(result.groups["structured"]["coverage"], 1.0)

    def test_group_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(GROUP_WEIGHTS.values()), 1.0)

    def test_bonus_and_penalty_caps(self):
        bonuses = [{"value": 0.04}, {"value": 0.04}]
        penalties = [{"value": 0.03}, {"value": 0.04}]
        final, bonus_total, penalty_total = finalize_score(0.50, bonuses, penalties)
        self.assertEqual(bonus_total, 0.05)
        self.assertEqual(penalty_total, 0.05)
        self.assertEqual(final, 0.50)

    def test_final_score_is_clamped(self):
        self.assertEqual(finalize_score(0.99, [{"value": 1}], [])[0], 1.0)
        self.assertEqual(finalize_score(0.01, [], [{"value": 1}])[0], 0.0)

    def test_breakdown_sums_to_base_and_final(self):
        structured_a = {
            "home_outdoor": 50,
            "future_children": "probably_yes",
            "smoking": "never",
            "alcohol": "rarely",
            "nightlife": "rarely",
            "has_children": False,
        }
        structured_b = dict(structured_a)
        a = profile(
            "a",
            structured=structured_a,
            tags={"interests": {"astronomy": 1, "anime": 1}},
            embeddings={"conflict": [1, 0]},
            answers={"conflict_style": "calm and fair"},
        )
        b = profile(
            "b",
            structured=structured_b,
            tags={"interests": {"astronomy": 1, "anime": 1}},
            embeddings={"conflict": [1, 0]},
            answers={"conflict_style": "calm and fair"},
        )
        result = compare_users(a, b, cross_fn=lambda x, y, *, task: 0.8)
        summed = sum(group["weighted_score"] for group in result.groups.values())
        self.assertAlmostEqual(summed, result.base_score)
        raw_bonus = sum(item["value"] for item in result.bonuses)
        raw_penalty = sum(item["value"] for item in result.penalties)
        expected = max(0.0, min(1.0, result.base_score + min(0.05, raw_bonus) - min(0.05, raw_penalty)))
        self.assertAlmostEqual(expected, result.final_score)

    def test_no_available_groups_has_zero_score_and_coverage(self):
        result = compare_users(profile("a"), profile("b"))
        self.assertEqual(result.base_score, 0.0)
        self.assertEqual(result.final_score, 0.0)
        self.assertEqual(result.coverage, 0.0)


if __name__ == "__main__":
    unittest.main()
