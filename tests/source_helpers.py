"""Locate migrated method source without importing ROS or constructing Qt.

Legacy orchestration tests execute a method with explicit fake dependencies.
Unqualifying migration references preserves that dependency-injection seam.
"""
import ast
import json
from eai_app.paths import PROJECT_ROOT


def method_tree(class_name, name, unqualify=False):
    registry = json.loads((PROJECT_ROOT/'eai_app/application/viewer_registry.json').read_text())
    mapping = next(v for v in registry.values() if isinstance(v, dict) and f'{class_name}.{name}' in v)
    module = mapping[f'{class_name}.{name}']
    tree = ast.parse((PROJECT_ROOT / (module.replace('.', '/') + '.py')).read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    if unqualify:
        class Dependencies(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id.startswith('_eai_'):
                    return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
                return self.generic_visit(node)
        method = Dependencies().visit(method)
    return ast.fix_missing_locations(method)
