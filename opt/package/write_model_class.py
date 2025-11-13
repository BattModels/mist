import ast
import inspect
from isort import code as isort_code
from pathlib import Path
from typing import Iterable, Type, Optional, List, Tuple, Set, Dict
from types import ModuleType
from jinja2 import Environment, FileSystemLoader, StrictUndefined


def get_source_code(obj) -> str:
    return inspect.getsource(obj).strip()


def get_module_and_source(obj) -> Tuple[Optional[ModuleType], Optional[str]]:
    m = inspect.getmodule(obj)
    if not m:
        return None, None
    try:
        return m, inspect.getsource(m)
    except OSError:
        path = getattr(m, "__file__", None)
        if path and path.endswith(".py"):
            return m, Path(path).read_text(encoding="utf-8")
        return m, None


def render_template(template_path: Path, **ctx) -> str:
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_path.name).render(**ctx)


def is_top_level_def(n: ast.AST) -> bool:
    return isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


def is_import(n: ast.AST) -> bool:
    return isinstance(n, (ast.Import, ast.ImportFrom))


def is_relative_import(n: ast.ImportFrom) -> bool:
    return isinstance(n, ast.ImportFrom) and (n.level or 0) > 0


def top_level_names(src: str) -> List[str]:
    t = ast.parse(src)
    return [n.name for n in t.body if is_top_level_def(n)]


def find_def_node(src: str, name: str) -> Optional[ast.AST]:
    t = ast.parse(src)
    for n in t.body:
        if is_top_level_def(n) and n.name == name:
            return n
    return None


def _alias(a: ast.alias) -> str:
    return f"{a.name}{' as ' + a.asname if a.asname else ''}"


def _reconstruct_import(n: ast.Import) -> str:
    return "import " + ", ".join(_alias(a) for a in n.names)


def _reconstruct_from(n: ast.ImportFrom) -> str:
    return f"from {n.module or ''} import " + ", ".join(_alias(a) for a in n.names)


def extract_abs_imports(src: str) -> Set[str]:
    t = ast.parse(src)
    out: List[str] = []
    for n in t.body:
        if is_import(n):
            if is_relative_import(n):  # skip relatives
                continue
            seg = ast.get_source_segment(src, n)
            out.append(
                seg.strip()
                if seg
                else (
                    _reconstruct_import(n)
                    if isinstance(n, ast.Import)
                    else _reconstruct_from(n)
                )
            )
        elif is_top_level_def(n):
            break
    return set(out)


def extract_imports_from_class(cls: Type) -> Set[str]:
    m, src = get_module_and_source(cls)
    return extract_abs_imports(src or "") if src else set()


def sort_dedupe_imports(imports: Set[str]) -> str:
    return isort_code("\n".join(sorted(imports))).strip()


def extract_def_source(src: str, module: ModuleType, name: str) -> Optional[str]:
    n = find_def_node(src, name)
    if n:
        seg = ast.get_source_segment(src, n)
        if seg:
            return seg.strip()
    obj = getattr(module, name, None)
    if inspect.isclass(obj) or inspect.isfunction(obj):
        return get_source_code(obj)
    return None


def extract_additional_defs(model_class: Type, exclude: Set[str]) -> List[str]:
    m, src = get_module_and_source(model_class)
    if not (m and src):
        return []
    out: List[str] = []
    for name in top_level_names(src):
        if name in exclude:
            continue
        s = extract_def_source(src, m, name)
        if s:
            out.append(s)
    return out


def resolve_relative_module_name(
    base_mod: ModuleType, node: ast.ImportFrom
) -> Optional[str]:
    pkg = base_mod.__package__ or ""
    level = node.level or 0
    if level > 1:
        parts = pkg.split(".")
        if len(parts) >= level:
            pkg = ".".join(parts[: -(level - 1)])
        else:
            return None
    if node.module and pkg:
        return f"{pkg}.{node.module}"
    return node.module or pkg


def relative_import_nodes(src: str) -> List[ast.ImportFrom]:
    t = ast.parse(src)
    return [n for n in t.body if is_relative_import(n)]


def defs_from_dependency_module(
    mod: ModuleType, visited: Set[str]
) -> Tuple[List[Type], List[str]]:
    name = mod.__name__
    if name in visited:
        return [], []
    visited.add(name)

    _, src = get_module_and_source(mod)
    if not src:
        return [], []

    classes: List[Type] = []
    functions: List[str] = []
    for n in top_level_names(src):
        obj = getattr(mod, n, None)
        if inspect.isclass(obj) and obj.__module__ == name:
            classes.append(obj)
        elif inspect.isfunction(obj) and obj.__module__ == name:
            functions.append(get_source_code(obj))
    return classes, functions


def collect_relative_dependencies(
    cls: Type, seen: Set[str], seen_mods: Set[str]
) -> Tuple[List[Type], List[str]]:
    key = f"{cls.__module__}.{cls.__name__}"
    if key in seen:
        return [], []
    seen.add(key)

    m, src = get_module_and_source(cls)
    if not (m and src):
        return [], []

    dep_classes: List[Type] = []
    dep_functions: List[str] = []
    for node in relative_import_nodes(src):
        mod_name = resolve_relative_module_name(m, node)
        if not mod_name:
            continue
        dep_mod = __import__(mod_name, fromlist=[""])
        cls_list, fn_list = defs_from_dependency_module(dep_mod, seen_mods)
        dep_classes.extend(cls_list)
        dep_functions.extend(fn_list)
        for dc in cls_list:
            c2, f2 = collect_relative_dependencies(dc, seen, seen_mods)
            dep_classes.extend(c2)
            dep_functions.extend(f2)
    return dep_classes, dep_functions


def collect_all_relative_dependencies(
    classes: List[Type],
) -> Tuple[List[Type], List[str]]:
    seen: Set[str] = set()
    seen_mods: Set[str] = set()
    all_c: List[Type] = []
    all_f: List[str] = []
    for c in classes:
        cc, ff = collect_relative_dependencies(c, seen, seen_mods)
        all_c.extend(cc)
        all_f.extend(ff)
    return all_c, all_f


def def_name_and_kind(src: str) -> Optional[Tuple[str, str]]:
    t = ast.parse(src)
    if not t.body:
        return None
    n = t.body[0]
    if isinstance(n, ast.ClassDef):
        return n.name, "class"
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return n.name, "function"
    return None


def dedupe_defs_by_name(segs: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for s in segs:
        nk = def_name_and_kind(s)
        if nk:
            name, _ = nk
            if name in seen:
                continue
            seen.add(name)
        out.append(s)
    return out


def deps_used_names(src: str, avail: Set[str]) -> Set[str]:
    """Find structural dependencies (base classes, decorators, class-level attrs)."""
    t = ast.parse(src)
    used: Set[str] = set()

    # Only look at top-level nodes
    for node in t.body:
        if isinstance(node, ast.ClassDef):
            # Check base classes (inheritance)
            for b in node.bases:
                if isinstance(b, ast.Name) and b.id in avail:
                    used.add(b.id)

            # Check decorators
            for d in node.decorator_list:
                if isinstance(d, ast.Name) and d.id in avail:
                    used.add(d.id)

            # Check class-level attributes
            for item in node.body:
                # Only check direct assignments, not function defs
                if isinstance(item, ast.Assign):
                    for name_node in ast.walk(item):
                        if isinstance(name_node, ast.Name) and name_node.id in avail:
                            used.add(name_node.id)
                elif isinstance(item, ast.AnnAssign):
                    # Type-annotated assignments
                    if item.value:
                        for name_node in ast.walk(item.value):
                            if (
                                isinstance(name_node, ast.Name)
                                and name_node.id in avail
                            ):
                                used.add(name_node.id)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Check function decorators
            for d in node.decorator_list:
                if isinstance(d, ast.Name) and d.id in avail:
                    used.add(d.id)

    return used


def topo_sort_sources(sources: List[str]) -> List[str]:
    name_to_src: Dict[str, str] = {}
    for s in sources:
        nk = def_name_and_kind(s)
        if nk:
            n, _ = nk
            if n not in name_to_src:
                name_to_src[n] = s

    avail = set(name_to_src.keys())
    deps: Dict[str, Set[str]] = {
        n: deps_used_names(src, avail) - {n} for n, src in name_to_src.items()
    }
    indeg = {n: len(d) for n, d in deps.items()}
    Q = sorted([n for n, d in indeg.items() if d == 0])
    order: List[str] = []

    while Q:
        n = Q.pop(0)
        order.append(n)
        for m in list(deps.keys()):
            if n in deps[m]:
                deps[m].discard(n)
                indeg[m] -= 1
                if indeg[m] == 0 and m not in order and m not in Q:
                    Q.append(m)
        Q.sort()

    if len(order) != len(avail):
        return list(name_to_src.values()) + [
            s for s in sources if def_name_and_kind(s) is None
        ]

    sorted_srcs = [name_to_src[n] for n in order]
    unnamed = [s for s in sources if def_name_and_kind(s) is None]
    return sorted_srcs + unnamed


class _TypeHintStripper(ast.NodeTransformer):
    """
    Remove type hints from functions and non-dataclass class bodies.
    """

    def __init__(self) -> None:
        super().__init__()
        self._dataclass_depth = 0

    @staticmethod
    def _is_dataclass_classdef(node: ast.ClassDef) -> bool:
        for d in node.decorator_list:
            if isinstance(d, ast.Name) and d.id == "dataclass":
                return True
            if isinstance(d, ast.Attribute) and d.attr == "dataclass":
                return True
        return False

    @staticmethod
    def _strip_args(a: ast.arguments) -> ast.arguments:
        for arg in a.posonlyargs + a.args + a.kwonlyargs:
            arg.annotation = None
        if a.vararg:
            a.vararg.annotation = None
        if a.kwarg:
            a.kwarg.annotation = None
        return a

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        is_dc = self._is_dataclass_classdef(node)
        if is_dc:
            self._dataclass_depth += 1
        self.generic_visit(node)
        if is_dc:
            self._dataclass_depth -= 1
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.returns = None
        node.args = self._strip_args(node.args)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.returns = None
        node.args = self._strip_args(node.args)
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if self._dataclass_depth > 0:
            return self.generic_visit(node)
        if node.value is None:
            return None
        return ast.Assign(targets=[node.target], value=node.value, type_comment=None)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.annotation = None
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node.type_comment = None
        return self.generic_visit(node)

    def visit_For(self, node: ast.For) -> ast.AST:
        node.type_comment = None
        return self.generic_visit(node)

    def visit_While(self, node: ast.While) -> ast.AST:
        node.type_comment = None
        return self.generic_visit(node)

    def visit_With(self, node: ast.With) -> ast.AST:
        node.type_comment = None
        return self.generic_visit(node)


def strip_type_hints_src(src: str) -> str:
    if not src.strip():
        return src
    tree = ast.parse(src, type_comments=True)
    tree = _TypeHintStripper().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def assemble_dependency_sources(
    additional: List[str], dep_classes: List[Type], dep_functions: List[str]
) -> str:
    parts: List[str] = []
    parts.extend(strip_type_hints_src(s) for s in additional)
    parts.extend(strip_type_hints_src(get_source_code(c)) for c in dep_classes)
    parts.extend(strip_type_hints_src(s) for s in dep_functions)
    parts = dedupe_defs_by_name(parts)
    parts = topo_sort_sources(parts)
    return "\n\n".join(parts)


def write_modeling_module(
    save_dir: Path,
    template_path: Path,
    *,
    config_class: Type,
    model_class: Type,
    dep_classes: Iterable[Type],
    model_type_aliases: dict | None = None,
    module_basename: str,
) -> Path:
    all_classes = [config_class, model_class, *dep_classes]
    imports = set().union(*(extract_imports_from_class(c) for c in all_classes))

    rel_dep_classes, rel_dep_functions = collect_all_relative_dependencies(all_classes)
    dep_classes_all = list(dep_classes) + rel_dep_classes
    for c in rel_dep_classes:
        imports.update(extract_imports_from_class(c))

    additional = extract_additional_defs(
        model_class, exclude={config_class.__name__, model_class.__name__}
    )
    imports_src = sort_dedupe_imports(imports)
    deps_src = assemble_dependency_sources(
        additional, dep_classes_all, rel_dep_functions
    )

    config_src = strip_type_hints_src(get_source_code(config_class))
    model_src = strip_type_hints_src(get_source_code(model_class))

    text = render_template(
        template_path,
        model_type_aliases=model_type_aliases or {},
        imports_src=imports_src,
        deps_src=deps_src,
        config_src=config_src,
        model_src=model_src,
    )
    out = save_dir / f"{module_basename}.py"
    out.write_text(text)
    return out
