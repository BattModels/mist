import ast
import inspect
from pathlib import Path
from typing import Iterable, Type, Optional, List, Tuple, Set
from types import ModuleType
from jinja2 import Environment, FileSystemLoader, StrictUndefined


def get_source_code(obj) -> str:
    """Get the source code of a Python object (class or function)."""
    return inspect.getsource(obj).strip()


def get_module_for_object(obj) -> Optional[ModuleType]:
    """Get the module that contains the given object."""
    return inspect.getmodule(obj)


def read_module_source_from_file(module: ModuleType) -> Optional[str]:
    """Fallback: read module source from its file path."""
    path = getattr(module, "__file__", None)
    if path and path.endswith(".py"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except (OSError, IOError):
            return None
    return None


def read_module_source_code(module: ModuleType) -> Optional[str]:
    """Read the source code of a module from inspect or file system."""
    try:
        return inspect.getsource(module)
    except OSError:
        return read_module_source_from_file(module)


def get_module_and_source(obj) -> Tuple[Optional[ModuleType], Optional[str]]:
    """Return (module, source_text) for the module containing `obj`."""
    module = get_module_for_object(obj)
    if not module:
        return None, None
    source = read_module_source_code(module)
    return module, source


def render_template(template_path: Path, **ctx) -> str:
    """Render a Jinja2 template with the given context."""
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_path.name).render(**ctx)


def parse_ast_tree(source_code: str) -> ast.Module:
    """Parse source code into an AST tree."""
    return ast.parse(source_code)


def is_top_level_definition(node: ast.AST) -> bool:
    """Check if an AST node is a top-level function or class definition."""
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


def is_import_statement(node: ast.AST) -> bool:
    """Check if an AST node is an import statement."""
    return isinstance(node, (ast.Import, ast.ImportFrom))


def is_relative_import(node: ast.ImportFrom) -> bool:
    """Check if an ImportFrom node is a relative import."""
    return isinstance(node, ast.ImportFrom) and (node.level or 0) > 0


def get_top_level_definition_names(source_code: str) -> List[str]:
    """Return names of top-level definitions (class/function) in file order."""
    tree = parse_ast_tree(source_code)
    names: List[str] = []
    for node in tree.body:
        if is_top_level_definition(node):
            names.append(node.name)
    return names


def find_definition_node_by_name(source_code: str, name: str) -> Optional[ast.AST]:
    """Find a top-level definition node by name in the AST."""
    tree = parse_ast_tree(source_code)
    for node in tree.body:
        if is_top_level_definition(node) and node.name == name:
            return node
    return None


def format_import_alias(alias: ast.alias) -> str:
    """Format an import alias as a string (e.g., 'foo as bar' or 'foo')."""
    return f"{alias.name}{' as ' + alias.asname if alias.asname else ''}"


def reconstruct_import_statement(node: ast.Import) -> str:
    """Reconstruct an Import statement as source code."""
    specs = [format_import_alias(a) for a in node.names]
    return "import " + ", ".join(specs)


def reconstruct_import_from_statement(node: ast.ImportFrom) -> str:
    """Reconstruct an ImportFrom statement as source code."""
    specs = [format_import_alias(a) for a in node.names]
    return f"from {node.module or ''} import " + ", ".join(specs)


def extract_import_statement_text(source_code: str, node: ast.AST) -> str:
    """Extract the source text for an import statement, with fallback reconstruction."""
    text = ast.get_source_segment(source_code, node)
    if text:
        return text.strip()

    # Fallback: reconstruct the import statement
    if isinstance(node, ast.Import):
        return reconstruct_import_statement(node)
    elif isinstance(node, ast.ImportFrom):
        return reconstruct_import_from_statement(node)
    return ""


def extract_absolute_imports_from_source(source_code: str) -> Set[str]:
    """Extract absolute (non-relative) import statements from source code."""
    tree = parse_ast_tree(source_code)
    imports: List[str] = []

    for node in tree.body:
        if is_import_statement(node):
            # Skip relative imports
            if is_relative_import(node):
                continue

            text = extract_import_statement_text(source_code, node)
            if text:
                imports.append(text)
        elif is_top_level_definition(node):
            # Stop at first definition (only process import block at top)
            break

    return set(imports)


def extract_imports_from_class(cls: Type) -> Set[str]:
    """Extract absolute import statements from the module containing a class."""
    module, source = get_module_and_source(cls)
    if not source:
        return set()
    return extract_absolute_imports_from_source(source)


def extract_source_for_definition(
    source_code: str, module: ModuleType, name: str
) -> Optional[str]:
    """
    Extract source code for a definition by name, with fallback to inspect.

    First tries AST source segment extraction, then falls back to
    inspect.getsource on the live object.
    """
    # Try AST-based extraction
    node = find_definition_node_by_name(source_code, name)
    if node:
        segment = ast.get_source_segment(source_code, node)
        if segment and segment.strip():
            return segment.strip()

    # Fallback to inspect
    obj = getattr(module, name, None)
    if obj and (inspect.isclass(obj) or inspect.isfunction(obj)):
        try:
            return inspect.getsource(obj).strip()
        except OSError:
            pass

    return None


def extract_additional_definitions_from_module(
    model_class: Type, exclude_names: Set[str]
) -> List[str]:
    """
    Extract all top-level definitions from model's module except excluded names.

    Preserves file order and handles AST extraction failures gracefully.
    """
    module, source_code = get_module_and_source(model_class)
    if not module or not source_code:
        return []

    extracted_sources: List[str] = []
    for name in get_top_level_definition_names(source_code):
        if name in exclude_names:
            continue

        source = extract_source_for_definition(source_code, module, name)
        if source:
            extracted_sources.append(source)

    return extracted_sources


def resolve_relative_import_module_name(
    base_module: ModuleType, import_node: ast.ImportFrom
) -> Optional[str]:
    """Resolve the full module name for a relative import."""
    package = base_module.__package__ or ""
    level = import_node.level or 0

    if level > 1:
        parts = package.split(".")
        if len(parts) >= level:
            package = ".".join(parts[: -(level - 1)])
        else:
            return None

    if import_node.module and package:
        return f"{package}.{import_node.module}"
    return import_node.module or package


def import_object_from_module(module_name: str, object_name: str) -> Optional[object]:
    """Import and return an object from a module by name."""
    try:
        imported_module = __import__(module_name, fromlist=[object_name])
        return getattr(imported_module, object_name)
    except (ImportError, AttributeError, ValueError):
        return None


def extract_relative_import_nodes(source_code: str) -> List[ast.ImportFrom]:
    """Extract all relative import nodes from source code."""
    tree = parse_ast_tree(source_code)
    return [node for node in tree.body if is_relative_import(node)]


def get_all_definitions_from_dependency_module(
    module: ModuleType, visited_modules: Set[str]
) -> Tuple[List[Type], List[str]]:
    """
    Extract ALL classes and functions from a dependency module.

    This ensures we don't miss any helper functions or classes that
    might be needed by the imported definitions.
    """
    module_name = module.__name__
    if module_name in visited_modules:
        return [], []
    visited_modules.add(module_name)

    _, source_code = get_module_and_source(module)
    if not source_code:
        return [], []

    classes: List[Type] = []
    functions: List[str] = []

    for name in get_top_level_definition_names(source_code):
        obj = getattr(module, name, None)
        if not obj:
            continue

        if inspect.isclass(obj):
            # Only add if it's defined in this module (not imported)
            if obj.__module__ == module_name:
                classes.append(obj)
        elif inspect.isfunction(obj):
            # Only add if it's defined in this module (not imported)
            if obj.__module__ == module_name:
                try:
                    functions.append(get_source_code(obj))
                except OSError:
                    pass

    return classes, functions


def collect_dependencies_from_relative_imports(
    cls: Type, visited: Set[str], visited_modules: Set[str]
) -> Tuple[List[Type], List[str]]:
    """
    Collect ALL classes and functions from modules with relative imports.

    Recursively follows relative imports and gathers all definitions
    from each dependency module (not just explicitly imported items).
    Returns (list of classes, list of function source codes).
    """
    cls_key = f"{cls.__module__}.{cls.__name__}"
    if cls_key in visited:
        return [], []
    visited.add(cls_key)

    module, source_code = get_module_and_source(cls)
    if not module or not source_code:
        return [], []

    dep_classes: List[Type] = []
    dep_functions: List[str] = []

    for import_node in extract_relative_import_nodes(source_code):
        module_name = resolve_relative_import_module_name(module, import_node)
        if not module_name:
            continue

        # Import the dependency module
        try:
            dep_module = __import__(module_name, fromlist=[""])
        except (ImportError, ValueError):
            continue

        # Get ALL definitions from this dependency module
        module_classes, module_functions = get_all_definitions_from_dependency_module(
            dep_module, visited_modules
        )
        dep_classes.extend(module_classes)
        dep_functions.extend(module_functions)

        # Recursively process each class from the dependency module
        for dep_class in module_classes:
            nested_classes, nested_functions = (
                collect_dependencies_from_relative_imports(
                    dep_class, visited, visited_modules
                )
            )
            dep_classes.extend(nested_classes)
            dep_functions.extend(nested_functions)

    return dep_classes, dep_functions


def collect_all_dependencies_from_module(
    module_path: str,
) -> Tuple[List[Type], List[str]]:
    """
    Collect ALL classes and functions from a module file.

    This ensures that when we import from a dependency module, we get
    all its definitions, not just the ones explicitly imported.
    """
    try:
        # Read the module source
        with open(module_path, "r", encoding="utf-8") as f:
            source_code = f.read()
    except (OSError, IOError):
        return [], []

    # Import the module to get live objects
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_temp_module", module_path)
        if not spec or not spec.loader:
            return [], []
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return [], []

    classes: List[Type] = []
    functions: List[str] = []

    for name in get_top_level_definition_names(source_code):
        obj = getattr(module, name, None)
        if inspect.isclass(obj):
            classes.append(obj)
        elif inspect.isfunction(obj):
            try:
                functions.append(get_source_code(obj))
            except OSError:
                pass

    return classes, functions


def deduplicate_classes_by_name(classes: List[Type]) -> List[Type]:
    """Remove duplicate classes, keeping first occurrence of each name."""
    seen_names: Set[str] = set()
    unique_classes: List[Type] = []

    for cls in classes:
        if cls.__name__ not in seen_names:
            seen_names.add(cls.__name__)
            unique_classes.append(cls)

    return unique_classes


def filter_duplicate_class_definitions(
    source_segments: List[str], existing_class_names: Set[str]
) -> List[str]:
    """
    Filter out class definitions that duplicate existing classes.

    Keeps function definitions and non-duplicate classes.
    """
    filtered: List[str] = []

    for segment in source_segments:
        try:
            node = parse_ast_tree(segment).body[0]
            if isinstance(node, ast.ClassDef) and node.name in existing_class_names:
                continue  # Skip duplicate class
        except Exception:
            pass  # Keep segment if we can't parse it

        filtered.append(segment)

    return filtered


def collect_all_imports(classes: List[Type]) -> Set[str]:
    """Collect all absolute imports from a list of classes."""
    all_imports: Set[str] = set()
    for cls in classes:
        all_imports.update(extract_imports_from_class(cls))
    return all_imports


def collect_all_relative_dependencies(
    classes: List[Type],
) -> Tuple[List[Type], List[str]]:
    """
    Collect all classes and functions from relative imports of given classes.

    Returns (all dependency classes, all dependency function sources).
    """
    visited: Set[str] = set()
    visited_modules: Set[str] = set()
    all_dep_classes: List[Type] = []
    all_dep_functions: List[str] = []

    for cls in classes:
        dep_classes, dep_functions = collect_dependencies_from_relative_imports(
            cls, visited, visited_modules
        )
        all_dep_classes.extend(dep_classes)
        all_dep_functions.extend(dep_functions)

    return all_dep_classes, all_dep_functions


def assemble_dependency_sources(
    additional_definitions: List[str],
    dependency_classes: List[Type],
    dependency_functions: List[str],
) -> str:
    """Assemble all dependency sources into a single string."""
    sources: List[str] = []
    sources.extend(additional_definitions)
    sources.extend(get_source_code(cls) for cls in dependency_classes)
    sources.extend(dependency_functions)
    return "\n\n".join(sources)


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
    """
    Write a complete modeling module with all dependencies.
    """
    all_classes = [config_class, model_class, *dep_classes]

    all_imports = collect_all_imports(all_classes)

    initial_dep_classes = list(dep_classes)
    rel_dep_classes, rel_dep_functions = collect_all_relative_dependencies(all_classes)
    combined_dep_classes = initial_dep_classes + rel_dep_classes

    # Add imports from relative dependencies
    for cls in rel_dep_classes:
        all_imports.update(extract_imports_from_class(cls))

    unique_dep_classes = deduplicate_classes_by_name(combined_dep_classes)
    seen_class_names = {cls.__name__ for cls in unique_dep_classes}

    additional_definitions = extract_additional_definitions_from_module(
        model_class, exclude_names={model_class.__name__, config_class.__name__}
    )
    filtered_additional = filter_duplicate_class_definitions(
        additional_definitions, seen_class_names
    )
    imports_src = "\n".join(sorted(all_imports)) if all_imports else ""
    deps_src = assemble_dependency_sources(
        filtered_additional, unique_dep_classes, rel_dep_functions
    )

    text = render_template(
        template_path,
        model_type_aliases=model_type_aliases or {},
        imports_src=imports_src,
        config_src=get_source_code(config_class),
        model_src=get_source_code(model_class),
        deps_src=deps_src,
    )

    output_path = save_dir / f"{module_basename}.py"
    output_path.write_text(text)
    return output_path
