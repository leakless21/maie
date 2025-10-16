"""Validate import patterns across the repository.

Checks:
- No in-function imports (heuristic)
- Optional: import ordering can be enforced by isort task already
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def find_in_function_imports(path: Path) -> List[Tuple[Path, int, str]]:
    results: List[Tuple[Path, int, str]] = []
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        return results

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return results

    class ImportVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    results.append((path, child.lineno, "import inside function"))
            self.generic_visit(node)

    ImportVisitor().visit(tree)
    return results


def iter_py_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        for path in root.rglob("*.py"):
            yield path


def main(argv: List[str] | None = None) -> int:
    roots = [Path("src"), Path("tests")]
    issues: List[str] = []
    for file_path in iter_py_files(roots):
        for path, lineno, msg in find_in_function_imports(file_path):
            issues.append(f"{path}:{lineno}: {msg}")
    if issues:
        print("\n".join(issues))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())



