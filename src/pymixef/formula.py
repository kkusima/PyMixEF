"""Safe R-like formula parsing and deterministic design-matrix compilation.

The parser implements fixed effects, ``*`` interactions, ``:`` explicit
interactions, ``/`` nesting, correlated ``|`` and independent ``||`` random
blocks.  Predictor expressions are interpreted by a small AST evaluator; Python
``eval`` and arbitrary attribute access are never used.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations, product
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data import AuditedData, ColumnarData, DataAudit, audit_data, is_missing
from .errors import FormulaError
from .ir import FixedEffectIR, LikelihoodIR, ModelIR, RandomEffectIR

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _split_top_level(text: str, separators: str) -> list[tuple[str, str]]:
    """Split on selected one-character operators outside parentheses."""

    output: list[tuple[str, str]] = []
    depth = 0
    quote: str | None = None
    escaped = False
    start = 0
    preceding = "+"
    for position, character in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise FormulaError("Formula contains an unmatched closing parenthesis.")
        elif depth == 0 and character in separators:
            token = text[start:position].strip()
            if token:
                output.append((preceding, token))
            preceding = character
            start = position + 1
    if quote is not None or depth != 0:
        raise FormulaError("Formula contains an unterminated quote or parenthesis.")
    token = text[start:].strip()
    if token:
        output.append((preceding, token))
    return output


def _split_operator(text: str, operator: str) -> list[str]:
    """Split on a possibly multi-character operator at top level."""

    depth = 0
    quote: str | None = None
    escaped = False
    parts: list[str] = []
    start = 0
    position = 0
    while position < len(text):
        character = text[position]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            position += 1
            continue
        if character in {"'", '"'}:
            quote = character
            position += 1
            continue
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif depth == 0 and text.startswith(operator, position):
            parts.append(text[start:position].strip())
            start = position + len(operator)
            position += len(operator)
            continue
        position += 1
    parts.append(text[start:].strip())
    return parts


def _join_interaction(parts: tuple[str, ...] | list[str]) -> str:
    return ":".join(parts)


def _expand_symbolic_term(term: str) -> tuple[str, ...]:
    """Expand one R-style ``*`` or ``/`` term."""

    star = _split_operator(term, "*")
    if len(star) > 1:
        if any(not item for item in star):
            raise FormulaError(f"Malformed interaction term {term!r}.")
        return tuple(
            _join_interaction(list(choice))
            for width in range(1, len(star) + 1)
            for choice in combinations(star, width)
        )
    nested = _split_operator(term, "/")
    if len(nested) > 1:
        if any(not item for item in nested):
            raise FormulaError(f"Malformed nesting term {term!r}.")
        return tuple(_join_interaction(nested[:width]) for width in range(1, len(nested) + 1))
    return (term.strip(),)


def _parse_additive(expression: str) -> tuple[bool, tuple[str, ...]]:
    intercept = True
    terms: list[str] = []
    removed: set[str] = set()
    for operator, raw in _split_top_level(expression, "+-"):
        compact = raw.replace(" ", "")
        if compact in {"0", "-1"}:
            if operator == "+":
                intercept = False
            continue
        if compact == "1":
            intercept = operator != "-"
            continue
        expanded = _expand_symbolic_term(raw)
        if operator == "-":
            removed.update(expanded)
            terms = [item for item in terms if item not in removed]
        else:
            for item in expanded:
                if item not in terms and item not in removed:
                    terms.append(item)
    return intercept, tuple(terms)


def _top_level_pipe(content: str) -> tuple[int, bool] | None:
    depth = 0
    quote: str | None = None
    position = 0
    while position < len(content):
        character = content[position]
        if quote:
            if character == quote and (position == 0 or content[position - 1] != "\\"):
                quote = None
        elif character in {"'", '"'}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "|" and depth == 0:
            independent = position + 1 < len(content) and content[position + 1] == "|"
            return position, independent
        position += 1
    return None


def _extract_random(rhs: str) -> tuple[str, tuple[tuple[str, str, bool], ...]]:
    spans: list[tuple[int, int, str, str, bool]] = []
    stack: list[int] = []
    quote: str | None = None
    for position, character in enumerate(rhs):
        if quote:
            if character == quote and (position == 0 or rhs[position - 1] != "\\"):
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "(":
            stack.append(position)
        elif character == ")":
            if not stack:
                raise FormulaError("Formula contains an unmatched closing parenthesis.")
            start = stack.pop()
            if not stack:  # only a complete outer expression can be a random block
                content = rhs[start + 1 : position]
                found = _top_level_pipe(content)
                if found is not None:
                    pipe, independent = found
                    width = 2 if independent else 1
                    left = content[:pipe].strip()
                    group = content[pipe + width :].strip()
                    if not left or not group:
                        raise FormulaError("Random-effect terms require both design and group.")
                    spans.append((start, position + 1, left, group, not independent))
    if stack:
        raise FormulaError("Formula contains an unmatched opening parenthesis.")
    fixed_chars = list(rhs)
    random: list[tuple[str, str, bool]] = []
    for start, end, left, group, correlated in spans:
        random.append((left, group, correlated))
        for position in range(start, end):
            fixed_chars[position] = " "
    fixed = "".join(fixed_chars)
    # Random spans leave redundant plus signs; the additive parser safely ignores them.
    return fixed, tuple(random)


def _expand_group(group: str) -> tuple[str, ...]:
    nested = _split_operator(group, "/")
    if len(nested) > 1:
        if any(not item for item in nested):
            raise FormulaError(f"Malformed grouping nesting {group!r}.")
        return tuple(_join_interaction(nested[:width]) for width in range(1, len(nested) + 1))
    if len(_split_operator(group, "+")) > 1 or len(_split_operator(group, "*")) > 1:
        raise FormulaError(
            "Grouping expressions use ':' for combinations and '/' for nesting; "
            "'+' and '*' are ambiguous."
        )
    return (group.strip(),)


@dataclass(frozen=True, slots=True)
class RandomTerm:
    """One resolved random-effects block."""

    terms: tuple[str, ...]
    group: str
    correlated: bool = True
    intercept: bool = True
    source: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        if not self.group:
            raise FormulaError("Random-effect grouping expressions cannot be empty.")

    @property
    def operator(self) -> str:
        return "|" if self.correlated else "||"

    @property
    def term_names(self) -> tuple[str, ...]:
        return (("Intercept",) if self.intercept else ()) + self.terms

    def to_dict(self) -> dict[str, Any]:
        return {
            "terms": list(self.terms),
            "group": self.group,
            "correlated": self.correlated,
            "intercept": self.intercept,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class FormulaSpec:
    """Parsed, data-independent formula semantics."""

    response: str
    fixed_terms: tuple[str, ...]
    random_terms: tuple[RandomTerm, ...] = ()
    intercept: bool = True
    source: str | None = None

    @property
    def formula(self) -> str | None:
        return self.source

    def to_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "fixed_terms": list(self.fixed_terms),
            "random_terms": [item.to_dict() for item in self.random_terms],
            "intercept": self.intercept,
            "source": self.source,
        }

    def to_ir(self, *, family: str = "gaussian") -> ModelIR:
        """Compile data-independent semantics into the shared model IR."""

        fixed = [FixedEffectIR(name="Intercept", expression="1")] if self.intercept else []
        fixed.extend(FixedEffectIR(name=item, expression=item) for item in self.fixed_terms)
        random = tuple(
            RandomEffectIR(
                terms=item.term_names,
                group=item.group,
                correlated=item.correlated,
                covariance="unstructured" if item.correlated else "diagonal",
            )
            for item in self.random_terms
        )
        return ModelIR(
            source="formula",
            formula=self.source,
            response=self.response,
            family=family,
            fixed_effects=tuple(fixed),
            random_effects=random,
            likelihoods=(LikelihoodIR(response=self.response, family=family),),
        )

    def explain(self) -> str:
        """Return a deterministic, data-independent mathematical summary."""

        fixed = ["1"] if self.intercept else []
        fixed.extend(self.fixed_terms)
        lines = [
            f"Response: {self.response}",
            f"Fixed predictor: {self.response} ~ {' + '.join(fixed) if fixed else '0'}",
        ]
        for item in self.random_terms:
            lines.append(
                "Random block: "
                f"({' + '.join(item.term_names)} {item.operator} {item.group}); "
                f"covariance={'unstructured' if item.correlated else 'diagonal'}"
            )
        return "\n".join(lines)


def parse_formula(formula: str) -> FormulaSpec:
    """Parse an R-like mixed-effects formula without touching data."""

    if not isinstance(formula, str) or not formula.strip():
        raise FormulaError("Formula must be a non-empty string.")
    tilde = _split_operator(formula, "~")
    if len(tilde) != 2:
        raise FormulaError("Formula must contain exactly one top-level '~'.")
    response, rhs = (item.strip() for item in tilde)
    if not response or not rhs:
        raise FormulaError("Formula requires both a response and a right-hand side.")
    try:
        response_tree = ast.parse(response, mode="eval")
    except SyntaxError as exc:
        raise FormulaError("Response is not a valid column name.") from exc
    if not isinstance(response_tree.body, ast.Name):
        raise FormulaError(
            "The response must be a single data column.",
            code="FORMULA-RESPONSE-001",
        )
    fixed_rhs, raw_random = _extract_random(rhs)
    # Strip operators left adjacent to removed random terms.
    fixed_rhs = fixed_rhs.strip()
    while "++" in fixed_rhs:
        fixed_rhs = fixed_rhs.replace("++", "+")
    fixed_rhs = fixed_rhs.strip("+ ")
    intercept, fixed_terms = _parse_additive(fixed_rhs or "1")
    random_terms: list[RandomTerm] = []
    for left, group_expression, correlated in raw_random:
        random_intercept, terms = _parse_additive(left)
        if not random_intercept and not terms:
            raise FormulaError("A random-effects block cannot have an empty design.")
        for group in _expand_group(group_expression):
            random_terms.append(
                RandomTerm(
                    terms=terms,
                    group=group,
                    correlated=correlated,
                    intercept=random_intercept,
                    source=f"({left} {'|' if correlated else '||'} {group_expression})",
                )
            )
    return FormulaSpec(
        response=response,
        fixed_terms=fixed_terms,
        random_terms=tuple(random_terms),
        intercept=intercept,
        source=formula.strip(),
    )


_FUNCTION_NAMES = {
    "I",
    "C",
    "abs",
    "center",
    "cos",
    "exp",
    "log",
    "log1p",
    "poly",
    "scale",
    "sin",
    "sqrt",
    "standardize",
}


def _expression_names(expression: str) -> set[str]:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise FormulaError(
            f"Invalid predictor expression {expression!r}.",
            source_location=expression,
        ) from exc
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _FUNCTION_NAMES:
            names.add(node.id)
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.Dict, ast.ListComp)):
            raise FormulaError(
                f"Unsafe syntax in predictor expression {expression!r}.",
                code="FORMULA-UNSAFE-001",
                remediation="Use documented column names and safe transform functions.",
            )
    return names


@dataclass(slots=True)
class _ExpressionValue:
    values: NDArray[Any]
    names: tuple[str, ...]
    force_categorical: bool = False


class _SafeEvaluator:
    def __init__(
        self,
        data: ColumnarData,
        *,
        level_source: ColumnarData | None = None,
    ) -> None:
        self.data = data
        self.level_source = level_source or data

    def evaluate(self, expression: str) -> _ExpressionValue:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise FormulaError(f"Invalid predictor expression {expression!r}.") from exc
        return self._visit(tree.body, expression)

    def _visit(self, node: ast.AST, source: str) -> _ExpressionValue:
        if isinstance(node, ast.Name):
            if node.id not in self.data:
                raise FormulaError(
                    f"Formula refers to absent column {node.id!r}.",
                    code="FORMULA-SYMBOL-001",
                    details={"available": list(self.data.column_names)},
                )
            return _ExpressionValue(np.asarray(self.data[node.id]), (node.id,))
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise FormulaError("Only numeric constants are allowed in predictors.")
            return _ExpressionValue(
                np.full(self.data.n_rows, float(node.value)),
                (repr(node.value),),
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = self._numeric(self._visit(node.operand, source), source)
            sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
            return _ExpressionValue(sign * operand.values, (source,))
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
        ):
            left = self._numeric(self._visit(node.left, source), source).values
            right = self._numeric(self._visit(node.right, source), source).values
            operation = {
                ast.Add: np.add,
                ast.Sub: np.subtract,
                ast.Mult: np.multiply,
                ast.Div: np.divide,
                ast.Pow: np.power,
            }[type(node.op)]
            with np.errstate(all="ignore"):
                values = operation(left, right)
            if not np.all(np.isfinite(values)):
                raise FormulaError(
                    f"Expression {source!r} produced non-finite values.",
                    code="FORMULA-TRANSFORM-DOMAIN-001",
                )
            return _ExpressionValue(np.asarray(values), (source,))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTION_NAMES:
                raise FormulaError(
                    "Only documented safe transform functions may be called.",
                    code="FORMULA-UNSAFE-001",
                )
            if node.keywords:
                raise FormulaError("Keyword arguments are not supported in formula transforms.")
            function = node.func.id
            if function == "poly":
                if len(node.args) != 2 or not isinstance(node.args[1], ast.Constant):
                    raise FormulaError("poly() requires a predictor and integer degree.")
                base = self._numeric(self._visit(node.args[0], source), source).values.reshape(-1)
                degree = node.args[1].value
                if not isinstance(degree, int) or degree < 1 or degree > 8:
                    raise FormulaError("poly() degree must be an integer from 1 through 8.")
                centered = base - np.mean(base)
                values = np.column_stack([centered**power for power in range(1, degree + 1)])
                names = tuple(
                    f"poly({ast.unparse(node.args[0])},{power})" for power in range(1, degree + 1)
                )
                return _ExpressionValue(values, names)
            if len(node.args) != 1:
                raise FormulaError(f"{function}() requires exactly one argument.")
            argument = self._visit(node.args[0], source)
            if function == "C":
                argument.force_categorical = True
                return argument
            numeric = self._numeric(argument, source).values
            if function == "I":
                values = numeric
            elif function == "abs":
                values = np.abs(numeric)
            elif function == "log":
                values = np.log(numeric)
            elif function == "log1p":
                values = np.log1p(numeric)
            elif function == "exp":
                values = np.exp(numeric)
            elif function == "sqrt":
                values = np.sqrt(numeric)
            elif function == "sin":
                values = np.sin(numeric)
            elif function == "cos":
                values = np.cos(numeric)
            elif function == "center":
                values = numeric - np.mean(numeric, axis=0)
            elif function in {"scale", "standardize"}:
                centered = numeric - np.mean(numeric, axis=0)
                deviation = np.std(centered, axis=0, ddof=0)
                if np.any(deviation == 0):
                    raise FormulaError(
                        f"{function}() cannot scale a constant predictor.",
                        code="FORMULA-CONSTANT-SCALE-001",
                    )
                values = centered / deviation
            else:  # pragma: no cover - exhaustiveness guard
                raise AssertionError(function)
            if not np.all(np.isfinite(values)):
                raise FormulaError(
                    f"Expression {source!r} produced non-finite values.",
                    code="FORMULA-TRANSFORM-DOMAIN-001",
                )
            return _ExpressionValue(np.asarray(values), (source,))
        raise FormulaError(
            f"Unsupported or unsafe syntax in predictor expression {source!r}.",
            code="FORMULA-UNSAFE-001",
        )

    @staticmethod
    def _numeric(value: _ExpressionValue, source: str) -> _ExpressionValue:
        try:
            values = np.asarray(value.values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise FormulaError(f"Expression {source!r} requires numeric inputs.") from exc
        return _ExpressionValue(values, value.names)


def _stable_levels(
    values: NDArray[Any],
    declared: tuple[Any, ...] = (),
) -> tuple[Any, ...]:
    if declared:
        return tuple(declared)
    unique: dict[tuple[str, str], Any] = {}
    for raw in values:
        if is_missing(raw):
            continue
        value = raw.item() if isinstance(raw, np.generic) else raw
        unique.setdefault((type(value).__name__, repr(value)), value)
    return tuple(unique[key] for key in sorted(unique))


def _declared_levels(data: ColumnarData, name: str) -> tuple[Any, ...]:
    for schema in data.schema:
        if schema.name == name and schema.categorical:
            return schema.levels
    return ()


def _source_levels(
    expression: str,
    raw_name: str,
    evaluator: _SafeEvaluator,
    analysis_values: NDArray[Any],
) -> tuple[Any, ...]:
    declared = _declared_levels(evaluator.level_source, raw_name)
    if declared:
        return declared
    if raw_name in evaluator.level_source:
        source_values = np.asarray(evaluator.level_source[raw_name])
    elif evaluator.level_source is evaluator.data:
        source_values = analysis_values
    else:
        source_values = np.asarray(
            _SafeEvaluator(evaluator.level_source).evaluate(expression).values
        )
    return _stable_levels(source_values)


def _is_categorical_atom(expression: str, evaluator: _SafeEvaluator) -> bool:
    """Return whether an atom uses factor rather than numeric model-matrix coding."""

    evaluated = evaluator.evaluate(expression)
    values = np.asarray(evaluated.values)
    return (
        evaluated.force_categorical
        or values.dtype.kind in {"O", "U", "S", "b"}
        or bool(_declared_levels(evaluator.data, expression))
    )


def _raw_factor_name(expression: str) -> str:
    if expression.startswith("C(") and expression.endswith(")"):
        return expression[2:-1].strip()
    return expression


def _record_contrast_coding(
    coding: dict[str, str],
    term: str,
    factor_levels: Mapping[str, tuple[Any, ...]],
    treatment_atoms: frozenset[str],
) -> None:
    """Record full, treatment, or term-dependent coding without hiding either."""

    for atom in _split_operator(term, ":"):
        name = _raw_factor_name(atom)
        if name not in factor_levels:
            continue
        selected = "treatment" if atom in treatment_atoms else "full"
        previous = coding.get(name)
        if previous is None:
            coding[name] = selected
        elif previous != selected:
            coding[name] = "term-dependent"


def _contrast_plan(
    terms: tuple[str, ...],
    evaluator: _SafeEvaluator,
    *,
    intercept: bool,
) -> tuple[frozenset[str], ...]:
    """Choose R-compatible full/treatment coding for each symbolic term.

    A categorical interaction spans one component for every subset of its
    factors.  Components already supplied by lower-order terms use treatment
    contrasts; factors needed to retain otherwise absent components use full
    indicator coding.  Processing lower-order terms first also reproduces the
    conventional no-intercept rule that the first factor supplies all levels.
    Numeric atoms are mandatory components rather than factors, so this logic
    also handles factor-by-numeric interactions without dropping a slope.
    """

    atoms_by_term = tuple(tuple(dict.fromkeys(_split_operator(term, ":"))) for term in terms)
    categorical_by_term = tuple(
        tuple(atom for atom in atoms if _is_categorical_atom(atom, evaluator))
        for atoms in atoms_by_term
    )
    plans: list[frozenset[str]] = [frozenset() for _ in terms]
    covered: set[tuple[frozenset[str], frozenset[str]]] = set()
    if intercept:
        covered.add((frozenset(), frozenset()))

    for index in sorted(range(len(terms)), key=lambda item: (len(atoms_by_term[item]), item)):
        atoms = atoms_by_term[index]
        categorical_atoms = categorical_by_term[index]
        categorical_set = frozenset(categorical_atoms)
        numeric_atoms = frozenset(atom for atom in atoms if atom not in categorical_set)
        components = tuple(
            frozenset(choice)
            for width in range(len(categorical_atoms) + 1)
            for choice in combinations(categorical_atoms, width)
        )
        missing = [
            component for component in components if (numeric_atoms, component) not in covered
        ]
        required = set(categorical_atoms)
        if missing:
            for component in missing:
                required.intersection_update(component)
        plans[index] = frozenset(required)
        covered.update(
            (numeric_atoms, component) for component in components if required.issubset(component)
        )
    return tuple(plans)


def _encode_atom(
    expression: str,
    evaluator: _SafeEvaluator,
    *,
    treatment_contrasts: bool,
) -> tuple[FloatArray, tuple[str, ...], Mapping[str, tuple[Any, ...]]]:
    evaluated = evaluator.evaluate(expression)
    values = np.asarray(evaluated.values)
    categorical = (
        evaluated.force_categorical
        or values.dtype.kind in {"O", "U", "S", "b"}
        or bool(_declared_levels(evaluator.data, expression))
    )
    if not categorical:
        numeric = np.asarray(values, dtype=float)
        if numeric.ndim == 1:
            numeric = numeric[:, None]
        if numeric.ndim != 2 or numeric.shape[0] != evaluator.data.n_rows:
            raise FormulaError(f"Expression {expression!r} has an invalid result shape.")
        names = (
            evaluated.names
            if len(evaluated.names) == numeric.shape[1]
            else tuple(f"{expression}[{position + 1}]" for position in range(numeric.shape[1]))
        )
        return numeric, names, {}
    if values.ndim != 1:
        raise FormulaError("Categorical expressions must produce one-dimensional values.")
    raw_name = _raw_factor_name(expression)
    levels = _source_levels(expression, raw_name, evaluator, values)
    if not levels:
        raise FormulaError(f"Categorical predictor {expression!r} has no observed levels.")
    encoded_levels = levels[1:] if treatment_contrasts else levels
    if not encoded_levels:
        matrix = np.empty((values.size, 0), dtype=float)
    else:
        matrix = np.column_stack([values == level for level in encoded_levels]).astype(float)
    names = tuple(f"{expression}[{level!s}]" for level in encoded_levels)
    return matrix, names, {raw_name: levels}


def _term_matrix(
    term: str,
    evaluator: _SafeEvaluator,
    *,
    treatment_atoms: frozenset[str],
) -> tuple[FloatArray, tuple[str, ...], Mapping[str, tuple[Any, ...]]]:
    atoms = _split_operator(term, ":")
    encoded = [
        _encode_atom(
            atom,
            evaluator,
            treatment_contrasts=atom in treatment_atoms,
        )
        for atom in atoms
    ]
    if len(encoded) == 1:
        return encoded[0]
    matrices = [item[0] for item in encoded]
    names = [item[1] for item in encoded]
    output_columns: list[FloatArray] = []
    output_names: list[str] = []
    levels: dict[str, tuple[Any, ...]] = {}
    for _, _, item_levels in encoded:
        levels.update(item_levels)
    if any(matrix.shape[1] == 0 for matrix in matrices):
        return np.empty((evaluator.data.n_rows, 0)), (), levels
    for indices in product(*(range(matrix.shape[1]) for matrix in matrices)):
        column = np.ones(evaluator.data.n_rows)
        labels: list[str] = []
        for matrix, term_names, index in zip(matrices, names, indices, strict=True):
            column *= matrix[:, index]
            labels.append(term_names[index])
        output_columns.append(column)
        output_names.append(":".join(labels))
    return np.column_stack(output_columns), tuple(output_names), levels


def _group_labels(expression: str, data: ColumnarData) -> NDArray[Any]:
    names = _split_operator(expression, ":")
    if any(name not in data for name in names):
        absent = [name for name in names if name not in data]
        raise FormulaError(
            f"Grouping columns are absent: {', '.join(absent)}.",
            code="FORMULA-SYMBOL-001",
        )
    if len(names) == 1:
        return np.asarray(data[names[0]])
    labels = np.empty(data.n_rows, dtype=object)
    for position in range(data.n_rows):
        labels[position] = tuple(
            value.item() if isinstance(value, np.generic) else value
            for value in (data[name][position] for name in names)
        )
    return labels


@dataclass(frozen=True, slots=True)
class RandomDesignBlock:
    """Compiled row-level random design and deterministic group factorization."""

    matrix: FloatArray
    group_labels: tuple[Any, ...]
    group_codes: IntArray
    group_levels: tuple[Any, ...]
    term_names: tuple[str, ...]
    correlated: bool
    name: str
    expanded: bool = False

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        codes = np.asarray(self.group_codes, dtype=np.int64)
        if matrix.ndim != 2 or codes.ndim != 1 or matrix.shape[0] != codes.size:
            raise FormulaError("A random design block has inconsistent dimensions.")
        matrix = np.array(matrix, copy=True)
        codes = np.array(codes, copy=True)
        matrix.setflags(write=False)
        codes.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "group_codes", codes)
        object.__setattr__(self, "group_labels", tuple(self.group_labels))
        object.__setattr__(self, "group_levels", tuple(self.group_levels))
        object.__setattr__(self, "term_names", tuple(self.term_names))

    @property
    def Z(self) -> FloatArray:
        return self.matrix

    @property
    def groups(self) -> tuple[Any, ...]:
        return self.group_labels

    @property
    def group(self) -> tuple[Any, ...]:
        return self.group_labels

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix,
            "group_labels": self.group_labels,
            "groups": self.group_labels,
            "group_codes": self.group_codes,
            "group_levels": self.group_levels,
            "term_names": self.term_names,
            "correlated": self.correlated,
            "name": self.name,
            "expanded": self.expanded,
        }


@dataclass(frozen=True, slots=True)
class FormulaExplanation:
    """Dry-run dimensions, equations, coding, and rank diagnostics."""

    formula: str
    response: str
    n_rows: int
    fixed_shape: tuple[int, int]
    fixed_names: tuple[str, ...]
    fixed_rank: int
    aliased_fixed: tuple[str, ...]
    random_blocks: tuple[Mapping[str, Any], ...]
    excluded_rows: int
    factor_levels: Mapping[str, tuple[Any, ...]]
    contrast_coding: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "response": self.response,
            "n_rows": self.n_rows,
            "fixed_shape": list(self.fixed_shape),
            "fixed_names": list(self.fixed_names),
            "fixed_rank": self.fixed_rank,
            "aliased_fixed": list(self.aliased_fixed),
            "random_blocks": [dict(item) for item in self.random_blocks],
            "excluded_rows": self.excluded_rows,
            "factor_levels": {name: list(levels) for name, levels in self.factor_levels.items()},
            "contrast_coding": dict(self.contrast_coding),
        }

    def __str__(self) -> str:
        blocks = "\n".join(
            f"  {item['name']}: Z{tuple(item['shape'])}, groups={item['groups']}, "
            f"covariance={item['covariance']}"
            for item in self.random_blocks
        )
        return (
            f"Formula: {self.formula}\n"
            f"Response: {self.response} ({self.n_rows} analysis rows)\n"
            f"Fixed design: X{self.fixed_shape}, rank={self.fixed_rank}, "
            f"columns={list(self.fixed_names)}\n"
            f"Random designs:\n{blocks or '  none'}\n"
            f"Excluded source rows: {self.excluded_rows}"
        )


@dataclass(frozen=True, slots=True)
class DesignMatrices:
    """Backend-neutral compiled response, fixed, and random designs."""

    response: FloatArray
    fixed: FloatArray
    fixed_names: tuple[str, ...]
    random_blocks: tuple[RandomDesignBlock, ...]
    row_ids: tuple[str, ...]
    audit: DataAudit
    spec: FormulaSpec
    factor_levels: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    contrast_coding: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        response = np.asarray(self.response, dtype=float).reshape(-1)
        fixed = np.asarray(self.fixed, dtype=float)
        if fixed.ndim != 2 or fixed.shape[0] != response.size:
            raise FormulaError("Fixed and response design dimensions do not agree.")
        if fixed.shape[1] != len(self.fixed_names):
            raise FormulaError("Fixed column names do not match the design matrix.")
        response = np.array(response, copy=True)
        fixed = np.array(fixed, copy=True)
        response.setflags(write=False)
        fixed.setflags(write=False)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "fixed", fixed)
        object.__setattr__(self, "fixed_names", tuple(self.fixed_names))
        object.__setattr__(self, "random_blocks", tuple(self.random_blocks))
        object.__setattr__(self, "row_ids", tuple(self.row_ids))
        object.__setattr__(
            self,
            "factor_levels",
            MappingProxyType({name: tuple(levels) for name, levels in self.factor_levels.items()}),
        )
        object.__setattr__(self, "contrast_coding", MappingProxyType(dict(self.contrast_coding)))

    @property
    def y(self) -> FloatArray:
        return self.response

    @property
    def X(self) -> FloatArray:
        return self.fixed

    def to_backend_data(self) -> dict[str, Any]:
        """Return the shared payload consumed by numerical engines."""

        return {
            "response": self.response,
            "fixed": self.fixed,
            "fixed_names": self.fixed_names,
            "random_blocks": tuple(block.to_dict() for block in self.random_blocks),
            "row_ids": self.row_ids,
            "audit": self.audit,
        }

    def explanation(self) -> FormulaExplanation:
        """Build structured dry-run diagnostics without fitting."""

        rank = int(np.linalg.matrix_rank(self.fixed))
        aliased: tuple[str, ...] = ()
        if rank < self.fixed.shape[1]:
            # Deterministic incremental rank identifies columns adding no new span.
            selected: list[int] = []
            rejected: list[str] = []
            current_rank = 0
            for index, name in enumerate(self.fixed_names):
                candidate = selected + [index]
                candidate_rank = int(np.linalg.matrix_rank(self.fixed[:, candidate]))
                if candidate_rank > current_rank:
                    selected.append(index)
                    current_rank = candidate_rank
                else:
                    rejected.append(name)
            aliased = tuple(rejected)
        blocks = tuple(
            {
                "name": block.name,
                "shape": block.matrix.shape,
                "groups": len(block.group_levels),
                "terms": list(block.term_names),
                "covariance": "unstructured" if block.correlated else "diagonal",
            }
            for block in self.random_blocks
        )
        return FormulaExplanation(
            formula=self.spec.source or "",
            response=self.spec.response,
            n_rows=self.response.size,
            fixed_shape=self.fixed.shape,
            fixed_names=self.fixed_names,
            fixed_rank=rank,
            aliased_fixed=aliased,
            random_blocks=blocks,
            excluded_rows=self.audit.excluded_rows,
            factor_levels=self.factor_levels,
            contrast_coding=self.contrast_coding,
        )

    def explain(self) -> str:
        """Return a human-readable dry-run report."""

        return str(self.explanation())


def _variables(spec: FormulaSpec) -> tuple[str, ...]:
    names: set[str] = set()
    for term in spec.fixed_terms:
        for atom in _split_operator(term, ":"):
            names.update(_expression_names(atom))
    for block in spec.random_terms:
        for term in block.terms:
            for atom in _split_operator(term, ":"):
                names.update(_expression_names(atom))
        names.update(_split_operator(block.group, ":"))
    names.discard(spec.response)
    return tuple(sorted(names))


def compile_formula(
    formula: str | FormulaSpec,
    data: Any,
    *,
    missing: str = "drop",
) -> DesignMatrices:
    """Compile a formula and data into deterministic numeric matrices."""

    spec = parse_formula(formula) if isinstance(formula, str) else formula
    audited: AuditedData = audit_data(
        data,
        response=spec.response,
        covariates=_variables(spec),
        missing=missing,
    )
    table = audited.data
    try:
        response = np.asarray(table[spec.response], dtype=float)
    except (TypeError, ValueError) as exc:
        raise FormulaError("The response must be numeric for matrix compilation.") from exc
    evaluator = _SafeEvaluator(table, level_source=audited.source)
    fixed_matrices: list[FloatArray] = []
    fixed_names: list[str] = []
    factor_levels: dict[str, tuple[Any, ...]] = dict(audited.audit.factor_levels)
    contrast_coding: dict[str, str] = {}
    if spec.intercept:
        fixed_matrices.append(np.ones((table.n_rows, 1)))
        fixed_names.append("Intercept")
    fixed_contrast_plan = _contrast_plan(
        spec.fixed_terms,
        evaluator,
        intercept=spec.intercept,
    )
    for term, treatment_atoms in zip(
        spec.fixed_terms,
        fixed_contrast_plan,
        strict=True,
    ):
        matrix, term_column_names, term_factor_levels = _term_matrix(
            term,
            evaluator,
            treatment_atoms=treatment_atoms,
        )
        if matrix.shape[1]:
            fixed_matrices.append(matrix)
            fixed_names.extend(term_column_names)
        factor_levels.update(term_factor_levels)
        _record_contrast_coding(
            contrast_coding,
            term,
            term_factor_levels,
            treatment_atoms,
        )
    fixed = (
        np.column_stack(fixed_matrices)
        if fixed_matrices
        else np.empty((table.n_rows, 0), dtype=float)
    )
    blocks: list[RandomDesignBlock] = []
    for position, block in enumerate(spec.random_terms):
        matrices: list[FloatArray] = []
        random_column_names: list[str] = []
        if block.intercept:
            matrices.append(np.ones((table.n_rows, 1)))
            random_column_names.append("Intercept")
        random_contrast_plan = _contrast_plan(
            block.terms,
            evaluator,
            intercept=block.intercept,
        )
        for term, treatment_atoms in zip(
            block.terms,
            random_contrast_plan,
            strict=True,
        ):
            matrix, term_names, term_factor_levels = _term_matrix(
                term,
                evaluator,
                treatment_atoms=treatment_atoms,
            )
            if matrix.shape[1]:
                matrices.append(matrix)
                random_column_names.extend(term_names)
            factor_levels.update(term_factor_levels)
        row_design = (
            np.column_stack(matrices) if matrices else np.empty((table.n_rows, 0), dtype=float)
        )
        labels = _group_labels(block.group, table)
        group_levels = _stable_levels(labels)
        level_lookup = {
            (type(value).__name__, repr(value)): index for index, value in enumerate(group_levels)
        }
        codes = np.asarray(
            [
                level_lookup[
                    (
                        type(value.item() if isinstance(value, np.generic) else value).__name__,
                        repr(value.item() if isinstance(value, np.generic) else value),
                    )
                ]
                for value in labels
            ],
            dtype=np.int64,
        )
        blocks.append(
            RandomDesignBlock(
                matrix=row_design,
                group_labels=tuple(labels.tolist()),
                group_codes=codes,
                group_levels=group_levels,
                term_names=tuple(random_column_names),
                correlated=block.correlated,
                name=block.group or f"random{position + 1}",
            )
        )
    # The compilation additions make factor and contrast decisions explicit.
    audit = DataAudit(
        input_rows=audited.audit.input_rows,
        analysis_rows=audited.audit.analysis_rows,
        records=audited.audit.records,
        factor_levels=factor_levels,
        factor_ordered=audited.audit.factor_ordered,
        contrast_coding=contrast_coding,
        transformations=tuple(
            {"kind": "safe-formula-expression", "expression": term}
            for term in spec.fixed_terms
            if any(character in term for character in "()+-/*")
        ),
        source_fingerprint=audited.audit.source_fingerprint,
        analysis_fingerprint=audited.audit.analysis_fingerprint,
    )
    return DesignMatrices(
        response=response,
        fixed=fixed,
        fixed_names=tuple(fixed_names),
        random_blocks=tuple(blocks),
        row_ids=table.row_ids,
        audit=audit,
        spec=spec,
        factor_levels=factor_levels,
        contrast_coding=contrast_coding,
    )


def model_matrix(
    formula: str | FormulaSpec,
    data: Any,
    *,
    missing: str = "drop",
) -> tuple[FloatArray, tuple[str, ...]]:
    """Return only the fixed-effect design and names."""

    compiled = compile_formula(formula, data, missing=missing)
    return compiled.fixed, compiled.fixed_names


def explain_formula(
    formula: str | FormulaSpec,
    data: Any | None = None,
    *,
    missing: str = "drop",
) -> str:
    """Explain parsed semantics, and matrix dimensions when data are supplied."""

    spec = parse_formula(formula) if isinstance(formula, str) else formula
    return (
        spec.explain() if data is None else compile_formula(spec, data, missing=missing).explain()
    )


dry_run = compile_formula


__all__ = [
    "DesignMatrices",
    "FormulaExplanation",
    "FormulaSpec",
    "RandomDesignBlock",
    "RandomTerm",
    "compile_formula",
    "dry_run",
    "explain_formula",
    "model_matrix",
    "parse_formula",
]
