"""Replace print() calls with loguru logger calls across Python files.
Usage: scripts/replace_prints_with_loguru.py [--dry-run] [paths...]
"""

import argparse
import ast
import os
import sys
from typing import List, Tuple

from loguru import logger


class PrintToLogger(ast.NodeTransformer):

    def __init__(self):
        self.in_except = False
        self.found = False

    def visit_ExceptHandler(self, node):
        prev = self.in_except
        self.in_except = True
        self.generic_visit(node)
        self.in_except = prev
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            level = "info"
            if self.in_except:
                level = "exception"
            elif node.args:
                first = node.args[0]
                try:
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        txt = first.value.lower()
                    elif isinstance(first, ast.JoinedStr):
                        parts = []
                        for v in first.values:
                            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                                parts.append(v.value)
                        txt = " ".join(parts).lower()
                    else:
                        txt = ""
                except Exception:
                    txt = ""
                if any(
                    (
                        k in txt
                        for k in (
                            "error",
                            "failed",
                            "exception",
                            "traceback",
                            "missing",
                        )
                    )
                ):
                    level = "error"
                elif any((k in txt for k in ("warn", "warning", "⚠️"))):
                    level = "warning"
                elif "debug" in txt or txt.startswith("debug"):
                    level = "debug"
                else:
                    level = "info"
            new_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="logger", ctx=ast.Load()),
                    attr=level,
                    ctx=ast.Load(),
                ),
                args=node.args,
                keywords=node.keywords,
            )
            self.found = True
            return ast.copy_location(new_call, node)
        return self.generic_visit(node)


def has_loguru_import(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and getattr(node, "module", "") == "loguru":
            for alias in node.names:
                if alias.name == "logger":
                    return True
    return False


def insert_loguru_import(tree: ast.Module) -> ast.Module:
    import_node = ast.ImportFrom(
        module="loguru", names=[ast.alias(name="logger", asname=None)], level=0
    )
    insert_at = 0
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        insert_at = 1
    for i, node in enumerate(tree.body[insert_at : insert_at + 10], start=insert_at):
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
            insert_at = i
            break
        insert_at = i + 1
    tree.body.insert(insert_at, import_node)
    return tree


def process_file(path: str, dry_run: bool = True) -> Tuple[bool, str]:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return (False, f"syntax error: {e}")
    transformer = PrintToLogger()
    new_tree = transformer.visit(tree)
    if not transformer.found:
        return (False, "no prints")
    if not has_loguru_import(new_tree):
        new_tree = insert_loguru_import(new_tree)
    try:
        new_src = ast.unparse(new_tree)
    except Exception as e:
        return (False, f"unparse failed: {e}")
    if src == new_src:
        return (False, "no change")
    if dry_run:
        return (True, "would modify")
    bak = path + ".bak"
    with open(bak, "w", encoding="utf-8") as f:
        f.write(src)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_src)
    return (True, f"modified (backup {bak})")


def should_skip(filename: str, exclude_dirs: List[str]) -> bool:
    parts = filename.split(os.sep)
    for d in exclude_dirs:
        if d in parts:
            return True
    return False


def find_py_files(paths: List[str], exclude_dirs: List[str]) -> List[str]:
    collected = []
    for base in paths:
        if os.path.isfile(base) and base.endswith(".py"):
            if not should_skip(base, exclude_dirs):
                collected.append(base)
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for fn in files:
                if fn.endswith(".py"):
                    full = os.path.join(root, fn)
                    if not should_skip(full, exclude_dirs):
                        collected.append(full)
    return collected


def main(argv: List[str]):
    parser = argparse.ArgumentParser(
        description="Replace print() with loguru logger calls."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["src", "tests", "examples", "scripts", "main.py"],
        help="Files or directories to process (default: project code paths)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show changes without writing",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually modify files (implies --dry-run False)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[".git", "__pycache__", "venv", ".venv", "build", ".pixi"],
        help="Directories to exclude",
    )
    args = parser.parse_args(argv)
    dry = not args.apply
    files = find_py_files(args.paths, args.exclude)
    if not files:
        logger.info("No python files found")
        return 0
    modified = []
    for f in files:
        ok, msg = process_file(f, dry_run=dry)
        if ok:
            modified.append((f, msg))
            logger.info(f"{f}: {msg}")
    logger.info(f"Processed {len(files)} files, {len(modified)} would be/modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
