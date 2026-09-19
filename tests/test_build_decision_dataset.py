import unittest

from scripts.build_decision_dataset import family_split_map


class DecisionSplitTests(unittest.TestCase):
    def test_train_covers_each_intent_and_families_are_disjoint(self):
        families = [
            "v2:shop_a:compare_products",
            "v2:shop_b:compare_products",
            "v2:mail_a:reply_to_email",
            "v2:mail_b:reply_to_email",
            "v2:repo_a:inspect_repository",
            "v2:repo_b:inspect_repository",
            "v2:noise_a:dismiss_noise",
            "v2:noise_b:dismiss_noise",
            "v2:search_a:research_topic",
            "v2:search_b:research_topic",
        ]
        mapping = family_split_map(families)
        train_labels = {family.rsplit(":", 1)[-1] for family, split in mapping.items() if split == "train"}
        self.assertEqual(train_labels, {family.rsplit(":", 1)[-1] for family in families})
        self.assertEqual(len(mapping), len(set(mapping)))
        for split in {"dev", "calibration", "test", "ood"}:
            self.assertTrue(any(value == split for value in mapping.values()))


if __name__ == "__main__":
    unittest.main()
