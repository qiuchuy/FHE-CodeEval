import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


class BenchmarkLayoutTests(unittest.TestCase):
    def test_reference_files_exist(self):
        self.assertTrue(list(BENCHMARKS.glob("tier*/*/*/torch_/ref.py")))

    def test_references_have_valid_layout_and_entry_point(self):
        references = sorted(BENCHMARKS.rglob("ref.py"))

        for reference in references:
            with self.subTest(reference=reference.relative_to(ROOT)):
                relative = reference.relative_to(BENCHMARKS)
                self.assertEqual(len(relative.parts), 5)
                self.assertRegex(relative.parts[0], r"^tier[1-9][0-9]*$")
                self.assertEqual(relative.parts[-2:], ("torch_", "ref.py"))

                source = reference.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(reference))
                kernels = [
                    node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "torch_kernel"
                ]

                self.assertEqual(len(kernels), 1)
                self.assertIsInstance(kernels[0], ast.FunctionDef)


if __name__ == "__main__":
    unittest.main()
