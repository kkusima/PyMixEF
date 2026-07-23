"""Dense Gaussian linear mixed-model reference engine.

This engine evaluates the exact marginal Gaussian likelihood and restricted
likelihood.  It supports multiple independent random-effect blocks, correlated
or diagonal within-group covariance, known observation weights, and an explicit
structured residual covariance template.  It deliberately forms dense
covariance matrices and is therefore a correctness/reference engine rather than
a sparse high-scale production engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, optimize, special

from .base import (
    BackendInputError,
    BackendNumericalError,
    CompiledData,
    cho_solve,
    convergence_mapping,
    covariance_slices,
    factorize,
    field,
    finite_gradient,
    logdet_from_cholesky,
    make_payload,
    optimizer_covariance,
    prepare_data,
    random_covariance,
    safe_cholesky,
    select_optimizer_result,
)


def _residual_template(compiled: CompiledData) -> NDArray[np.float64]:
    """Resolve and validate a residual covariance/correlation template."""

    raw = compiled.residual_covariance
    n = compiled.n_obs
    if raw is None:
        template = np.eye(n)
    elif isinstance(raw, (np.ndarray, list, tuple)):
        template = np.asarray(raw, dtype=float)
    else:
        matrix_member = field(raw, "matrix", "covariance_matrix", default=None)
        if matrix_member is None:
            raise BackendInputError(
                "residual covariance must be an array or expose matrix()/matrix"
            )
        if callable(matrix_member):
            times = field(
                compiled.source,
                "times",
                "time",
                "visit_times",
                default=np.arange(n, dtype=float),
            )
            attempts = (
                lambda: matrix_member(size=n, times=times),
                lambda: matrix_member(n, times),
                lambda: matrix_member(times),
                lambda: matrix_member(n),
                lambda: matrix_member(),
            )
            last_error: Exception | None = None
            for attempt in attempts:
                try:
                    template = np.asarray(attempt(), dtype=float)
                    break
                except (TypeError, ValueError) as exc:
                    last_error = exc
            else:
                raise BackendInputError(
                    "could not evaluate residual covariance matrix",
                    details={"last_error": repr(last_error)},
                )
        else:
            template = np.asarray(matrix_member, dtype=float)

    if template.shape != (n, n):
        raise BackendInputError(
            f"residual covariance has shape {template.shape}; expected {(n, n)}"
        )
    if np.any(~np.isfinite(template)):
        raise BackendInputError("residual covariance contains non-finite values")
    template = (template + template.T) / 2
    try:
        safe_cholesky(template, jitter_scale=1e-12, max_tries=2)
    except linalg.LinAlgError as exc:
        raise BackendInputError("residual covariance must be positive definite") from exc

    # Analytic/frequency weights scale residual standard deviations.
    scaling = 1.0 / np.sqrt(compiled.weights)
    return scaling[:, None] * template * scaling[None, :]


@dataclass(slots=True)
class _StructuredResidual:
    """Adapter for covariance.CovarianceStructure-like objects."""

    source: Any
    compiled: CompiledData
    row_blocks: tuple[NDArray[np.int64], ...]
    index: NDArray[np.float64]
    parameter_count: int
    raw_names: tuple[str, ...]

    @classmethod
    def create(cls, source: Any, compiled: CompiledData) -> _StructuredResidual:
        grouping = field(
            compiled.source,
            "residual_groups",
            "residual_group",
            "subjects",
            "subject",
            default=None,
        )
        if grouping is None and getattr(source, "group", None):
            grouping = field(compiled.source, "groups", "group", default=None)
        if grouping is None:
            row_blocks = (np.arange(compiled.n_obs, dtype=np.int64),)
        else:
            codes, levels = factorize(grouping)
            if codes.size != compiled.n_obs:
                raise BackendInputError(
                    "structured residual grouping must have one value per observation"
                )
            row_blocks = tuple(np.flatnonzero(codes == code) for code in range(len(levels)))

        raw_index = field(
            compiled.source,
            "residual_index",
            "visit_times",
            "times",
            "time",
            "visits",
            "visit",
            default=np.arange(compiled.n_obs),
        )
        try:
            index = np.asarray(raw_index, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            index = factorize(raw_index)[0].astype(float)
        if index.size != compiled.n_obs:
            raise BackendInputError("structured residual index must have one value per observation")

        counts: list[int] = []
        names: tuple[str, ...] | None = None
        for rows in row_blocks:
            try:
                count = int(source.parameter_count(size=len(rows)))
            except TypeError:
                count = int(source.parameter_count(len(rows)))
            counts.append(count)
            try:
                block_names = tuple(str(item) for item in source.parameter_names(size=len(rows)))
            except TypeError:
                block_names = tuple(str(item) for item in source.parameter_names(len(rows)))
            if names is None:
                names = block_names
            elif block_names != names:
                raise BackendInputError(
                    "structured residual groups imply different parameterizations; "
                    "use equal visit dimensions or the MMRM backend"
                )
        if len(set(counts)) != 1:
            raise BackendInputError(
                "structured residual groups have incompatible covariance dimensions"
            )
        return cls(
            source=source,
            compiled=compiled,
            row_blocks=row_blocks,
            index=index,
            parameter_count=counts[0],
            raw_names=names or (),
        )

    def initial(self, scale: float) -> NDArray[np.float64]:
        result = np.zeros(self.parameter_count)
        for position, name in enumerate(self.raw_names):
            normalized = name.lower()
            if (
                normalized.startswith("log_sd")
                or normalized.startswith("log_innovation_sd")
                or normalized.startswith("log_chol")
            ):
                result[position] = np.log(max(scale, 1e-4))
        return result

    def matrix(self, theta: ArrayLike) -> NDArray[np.float64]:
        theta = np.asarray(theta, dtype=float)
        result = np.zeros((self.compiled.n_obs, self.compiled.n_obs))
        covariance_method = getattr(self.source, "covariance", None)
        matrix_method = getattr(self.source, "matrix", None)
        method = covariance_method if callable(covariance_method) else matrix_method
        if not callable(method):
            raise BackendInputError("structured residual object does not expose covariance()")
        for rows in self.row_blocks:
            block_index = self.index[rows]
            attempts = (
                lambda rows=rows, block_index=block_index: method(
                    theta, size=len(rows), index=block_index
                ),
                lambda rows=rows, block_index=block_index: method(
                    theta, size=len(rows), times=block_index
                ),
                lambda rows=rows: method(theta, size=len(rows)),
                lambda: method(theta),
            )
            last_error: Exception | None = None
            for attempt in attempts:
                try:
                    block = np.asarray(attempt(), dtype=float)
                    break
                except (TypeError, ValueError) as exc:
                    last_error = exc
            else:
                raise BackendInputError(
                    "could not evaluate structured residual covariance",
                    details={"last_error": repr(last_error)},
                )
            if block.shape != (len(rows), len(rows)):
                raise BackendInputError(
                    "structured residual covariance returned the wrong block dimension"
                )
            result[np.ix_(rows, rows)] = block
        scaling = 1.0 / np.sqrt(self.compiled.weights)
        return scaling[:, None] * result * scaling[None, :]

    def natural(self, theta: ArrayLike) -> dict[str, float]:
        theta = np.asarray(theta, dtype=float)
        structure = str(getattr(self.source, "name", "")).lower()
        result: dict[str, float] = {}
        for name, value in zip(self.raw_names, theta, strict=True):
            normalized = name.lower()
            if normalized == "logit_rho":
                result["residual_spatial_correlation"] = float(special.expit(value))
            elif "correlation_unconstrained" in normalized:
                if structure == "compound-symmetry":
                    dimension = len(self.row_blocks[0])
                    lower = -1 / (dimension - 1) if dimension > 1 else 0.0
                    correlation = lower + (1 - lower) / (1 + np.exp(-value))
                else:
                    correlation = np.tanh(value)
                result["residual_correlation"] = float(correlation)
            elif "partial_correlation" in normalized:
                result[f"residual_{normalized}"] = float(np.tanh(value))
            elif normalized.startswith("log_"):
                natural_name = normalized.removeprefix("log_")
                result[f"residual_{natural_name}"] = float(np.exp(value))
            else:
                result[f"residual_{normalized}"] = float(value)
        return result


@dataclass(slots=True)
class _Evaluation:
    log_likelihood: float
    beta: NDArray[np.float64]
    beta_covariance: NDArray[np.float64]
    residual: NDArray[np.float64]
    V: NDArray[np.float64]
    V_cholesky: NDArray[np.float64]
    G: NDArray[np.float64]
    rank: int
    logdet_x_information: float
    jitter: float


class GaussianLMMBackend:
    """Exact dense/reference Gaussian LMM engine."""

    name = "lmm"

    def fit(
        self,
        data: Any,
        *,
        method: str | None = None,
        reml: bool | None = None,
        maxiter: int = 1_000,
        tolerance: float = 1e-8,
        compute_hessian: bool = True,
        **options: Any,
    ) -> dict[str, Any]:
        compiled = prepare_data(data)
        if method is None:
            method = str(field(compiled.source, "method", "estimator", default="REML"))
        if reml is not None:
            method = "REML" if reml else "ML"
        method = method.upper()
        if method not in {"ML", "REML"}:
            raise BackendInputError("Gaussian LMM method must be 'ML' or 'REML'")

        n, p = compiled.X.shape
        if n == 0:
            raise BackendInputError("cannot fit an empty response")
        if method == "REML" and n <= np.linalg.matrix_rank(compiled.X):
            raise BackendInputError("REML requires more observations than fixed-effect rank")

        parameterizations = covariance_slices(compiled.random_blocks)
        random_theta_count = sum(item.size for item, _ in parameterizations)
        response_scale = max(float(np.std(compiled.y, ddof=1)) if n > 1 else 1.0, 1e-3)
        theta_parts = [item.initial(response_scale * 0.35) for item, _ in parameterizations]

        raw_residual = compiled.residual_covariance
        is_structured = bool(
            raw_residual is not None
            and not isinstance(raw_residual, (np.ndarray, list, tuple))
            and callable(getattr(raw_residual, "parameter_count", None))
            and (
                callable(getattr(raw_residual, "covariance", None))
                or callable(getattr(raw_residual, "matrix", None))
            )
        )
        structured_residual = (
            _StructuredResidual.create(raw_residual, compiled) if is_structured else None
        )
        residual_template = None if is_structured else _residual_template(compiled)
        if structured_residual is not None:
            if compiled.residual_covariance_fixed and structured_residual.parameter_count:
                raise BackendInputError(
                    "fixed structured residual covariance requires a zero-parameter "
                    "KnownCovariance object or an explicit matrix"
                )
            residual_theta_count = structured_residual.parameter_count
            if residual_theta_count:
                theta_parts.append(structured_residual.initial(response_scale * 0.8))
            residual_mode = "structured"
        elif not compiled.residual_covariance_fixed:
            residual_theta_count = 1
            theta_parts.append(np.array([np.log(response_scale * 0.8)]))
            residual_mode = "template-scale"
        else:
            residual_theta_count = 0
            residual_mode = "fixed"
        residual_section = slice(random_theta_count, random_theta_count + residual_theta_count)
        theta0 = np.concatenate(theta_parts) if theta_parts else np.zeros(0)

        def residual_matrix(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            if residual_mode == "structured":
                assert structured_residual is not None
                return structured_residual.matrix(theta[residual_section])
            assert residual_template is not None
            if residual_mode == "template-scale":
                return np.exp(2 * theta[residual_section.start]) * residual_template
            return residual_template

        def evaluate(theta: ArrayLike) -> _Evaluation:
            theta = np.asarray(theta, dtype=float)
            G = random_covariance(parameterizations, theta[:random_theta_count])
            R = residual_matrix(theta)
            Z = compiled.random_design
            V = R if not Z.shape[1] else R + Z @ G @ Z.T
            cholesky, jitter = safe_cholesky(V)

            whitened_y = linalg.solve_triangular(
                cholesky, compiled.y, lower=True, check_finite=False
            )
            whitened_X = linalg.solve_triangular(
                cholesky, compiled.X, lower=True, check_finite=False
            )
            beta, _, rank, singular_values = np.linalg.lstsq(whitened_X, whitened_y, rcond=None)
            residual = compiled.y - compiled.X @ beta
            solved_residual = cho_solve(cholesky, residual)
            quadratic = float(residual @ solved_residual)
            logdet_v = logdet_from_cholesky(cholesky)

            information = whitened_X.T @ whitened_X
            beta_covariance = np.linalg.pinv(information, rcond=1e-10)
            if rank:
                positive = singular_values[:rank] ** 2
                logdet_information = float(np.log(positive).sum())
            else:
                logdet_information = 0.0

            if method == "ML":
                log_likelihood = -0.5 * (n * np.log(2 * np.pi) + logdet_v + quadratic)
            else:
                log_likelihood = -0.5 * (
                    (n - rank) * np.log(2 * np.pi) + logdet_v + logdet_information + quadratic
                )
            return _Evaluation(
                log_likelihood=float(log_likelihood),
                beta=beta,
                beta_covariance=beta_covariance,
                residual=residual,
                V=V,
                V_cholesky=cholesky,
                G=G,
                rank=int(rank),
                logdet_x_information=logdet_information,
                jitter=jitter,
            )

        def objective(theta: ArrayLike) -> float:
            try:
                value = -evaluate(theta).log_likelihood
                return value if np.isfinite(value) else 1e100
            except (ValueError, FloatingPointError, linalg.LinAlgError):
                return 1e100

        if theta0.size:
            bounds = [(-12.0, 12.0)] * theta0.size
            result = optimize.minimize(
                objective,
                theta0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": int(maxiter), "ftol": tolerance, "gtol": tolerance},
            )
            # Powell is a deterministic rescue for rough covariance surfaces,
            # followed by L-BFGS-B so the reported gradient matches the optimum.
            initial_gradient = np.asarray(getattr(result, "jac", np.array([np.inf])))
            if (
                not result.success
                or not np.isfinite(result.fun)
                or np.max(np.abs(initial_gradient)) > max(np.sqrt(tolerance), 1e-5)
            ):
                rescue = optimize.minimize(
                    objective,
                    np.clip(result.x if np.all(np.isfinite(result.x)) else theta0, -12, 12),
                    method="Powell",
                    bounds=bounds,
                    options={"maxiter": int(maxiter), "xtol": tolerance, "ftol": tolerance},
                )
                refined = optimize.minimize(
                    objective,
                    rescue.x,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": int(maxiter), "ftol": tolerance, "gtol": tolerance},
                )
                result = select_optimizer_result(
                    refined,
                    rescue,
                    objective_tolerance=1e-8,
                )
            theta_hat = np.asarray(result.x, dtype=float)
            success = bool(result.success and np.isfinite(result.fun))
            optimizer_message = str(result.message)
            iterations = int(getattr(result, "nit", 0))
            evaluations = int(getattr(result, "nfev", 0))
            raw_gradient = getattr(result, "jac", None)
            gradient = (
                finite_gradient(objective, theta_hat)
                if raw_gradient is None
                else np.asarray(raw_gradient, dtype=float)
            )
            if np.any(~np.isfinite(gradient)):
                gradient = finite_gradient(objective, theta_hat)
        else:
            theta_hat = theta0
            success = True
            optimizer_message = "No covariance parameters required optimization"
            iterations = evaluations = 0
            gradient = np.zeros(0)

        try:
            final = evaluate(theta_hat)
        except (ValueError, linalg.LinAlgError) as exc:
            raise BackendNumericalError(
                "optimized LMM covariance could not be factorized",
                details={"optimizer_message": optimizer_message},
            ) from exc

        # Identify aliased fixed effects with pivoted QR.
        _, _, pivots = linalg.qr(compiled.X, mode="economic", pivoting=True)
        aliased_indices = sorted(int(index) for index in pivots[final.rank :])
        aliased_names = [compiled.fixed_names[index] for index in aliased_indices]

        theta_covariance, hessian_positive, curvature_source = optimizer_covariance(
            objective,
            theta_hat,
            result if theta_hat.size else None,
            compute_hessian=compute_hessian,
            finite_difference_limit=25,
        )

        natural_parameters: dict[str, float] = {
            name: float(value) for name, value in zip(compiled.fixed_names, final.beta, strict=True)
        }
        unconstrained_parameters: dict[str, float] = dict(natural_parameters)
        theta_names: list[str] = []
        natural_covariance_names: list[str] = []
        natural_covariance_values: list[float] = []
        covariance_eigenvalues: dict[str, list[float]] = {}
        covariance_ratios: list[float] = []
        for parameterization, section in parameterizations:
            block_values = parameterization.names_and_values(theta_hat[section])
            natural_parameters.update(block_values)
            natural_covariance_names.extend(block_values)
            natural_covariance_values.extend(block_values.values())
            raw_names = parameterization.unconstrained_names()
            theta_names.extend(raw_names)
            unconstrained_parameters.update(
                {
                    name: float(value)
                    for name, value in zip(raw_names, theta_hat[section], strict=True)
                }
            )
            block_covariance = parameterization.matrix(theta_hat[section])
            eigenvalues = np.linalg.eigvalsh(block_covariance)
            covariance_eigenvalues[parameterization.block.name] = eigenvalues.tolist()
            covariance_ratios.append(
                float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > 0 else 0.0
            )
        if residual_mode == "template-scale":
            residual_sd = float(np.exp(theta_hat[residual_section.start]))
            natural_parameters["residual_sd"] = residual_sd
            natural_covariance_names.append("residual_sd")
            natural_covariance_values.append(residual_sd)
            theta_names.append("log_residual_sd")
            unconstrained_parameters["log_residual_sd"] = float(theta_hat[residual_section.start])
        elif residual_mode == "structured" and structured_residual is not None:
            residual_values = structured_residual.natural(theta_hat[residual_section])
            natural_parameters.update(residual_values)
            natural_covariance_names.extend(residual_values)
            natural_covariance_values.extend(residual_values.values())
            structured_names = [f"residual:{name}" for name in structured_residual.raw_names]
            theta_names.extend(structured_names)
            unconstrained_parameters.update(
                {
                    name: float(value)
                    for name, value in zip(
                        structured_names, theta_hat[residual_section], strict=True
                    )
                }
            )

        # Delta-method transform from covariance theta to sd/correlation scale.
        if theta_hat.size:

            def natural_theta_values(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                values: list[float] = []
                for parameterization, section in parameterizations:
                    values.extend(parameterization.names_and_values(theta[section]).values())
                if residual_mode == "template-scale":
                    values.append(float(np.exp(theta[residual_section.start])))
                elif residual_mode == "structured" and structured_residual is not None:
                    values.extend(structured_residual.natural(theta[residual_section]).values())
                return np.asarray(values)

            jacobian = np.empty((len(natural_covariance_values), theta_hat.size))
            for column in range(theta_hat.size):
                step = 1e-5 * max(1.0, abs(theta_hat[column]))
                delta = np.zeros(theta_hat.size)
                delta[column] = step
                jacobian[:, column] = (
                    natural_theta_values(theta_hat + delta)
                    - natural_theta_values(theta_hat - delta)
                ) / (2 * step)
            transformed_theta_covariance = jacobian @ theta_covariance @ jacobian.T
        else:
            transformed_theta_covariance = np.zeros((0, 0))
        parameter_covariance = linalg.block_diag(
            final.beta_covariance, transformed_theta_covariance
        )

        Z = compiled.random_design
        if Z.shape[1]:
            marginal_residual_solution = cho_solve(final.V_cholesky, final.residual)
            random_effects = final.G @ Z.T @ marginal_residual_solution
            conditional_covariance = final.G - final.G @ Z.T @ cho_solve(
                final.V_cholesky, Z @ final.G
            )
            conditional_covariance = (conditional_covariance + conditional_covariance.T) / 2
            conditional_sd = np.sqrt(np.maximum(np.diag(conditional_covariance), 0.0))
            conditional_fitted = compiled.X @ final.beta + Z @ random_effects
            random_labels = [
                label for block in compiled.random_blocks for label in block.coefficient_labels()
            ]
        else:
            random_effects = np.zeros(0)
            conditional_covariance = np.zeros((0, 0))
            conditional_sd = np.zeros(0)
            conditional_fitted = compiled.X @ final.beta
            random_labels = []

        fitted = conditional_fitted
        residuals = compiled.y - fitted
        standardized = residuals / np.sqrt(np.maximum(np.diag(final.V), np.finfo(float).tiny))
        singular = bool(covariance_ratios and min(covariance_ratios) < 1e-6)
        boundary_names = [
            name
            for name, value in natural_parameters.items()
            if name.startswith("sd(") and value < response_scale * 1e-5
        ]
        warnings: list[dict[str, Any]] = []
        if aliased_names:
            warnings.append(
                {
                    "code": "LMM-RANK-001",
                    "severity": "warning",
                    "message": "fixed-effect design is rank deficient",
                    "details": {"aliased_coefficients": aliased_names},
                }
            )
        if singular:
            warnings.append(
                {
                    "code": "COV-SINGULAR-001",
                    "severity": "warning",
                    "message": "a random-effect covariance block is singular or near-singular",
                    "details": {"eigenvalue_ratios": covariance_ratios},
                }
            )
        if final.jitter:
            warnings.append(
                {
                    "code": "LMM-JITTER-001",
                    "severity": "info",
                    "message": "tiny diagonal jitter was required for covariance factorization",
                    "details": {"jitter": final.jitter},
                }
            )

        convergence = convergence_mapping(
            success=success,
            message=optimizer_message,
            iterations=iterations,
            function_evaluations=evaluations,
            gradient=gradient,
            hessian_positive_definite=hessian_positive,
            singular=singular,
            boundary_parameters=boundary_names,
            warnings=warnings,
            extra={
                "fixed_effect_rank": final.rank,
                "fixed_effect_columns": p,
                "aliased_coefficients": aliased_names,
                "curvature_source": curvature_source,
            },
        )

        parameter_names = list(compiled.fixed_names) + natural_covariance_names
        return make_payload(
            parameters=natural_parameters,
            unconstrained_parameters=unconstrained_parameters,
            parameter_covariance=parameter_covariance,
            fitted_values=fitted,
            residuals=residuals,
            random_effects=random_effects,
            objective=float(-final.log_likelihood),
            log_likelihood=float(final.log_likelihood),
            method=method,
            engine=self.name,
            convergence=convergence,
            diagnostic_data={
                "observations": {
                    "observed": compiled.y.tolist(),
                    "fitted_conditional": fitted.tolist(),
                    "residual": residuals.tolist(),
                    "standardized_residual": standardized.tolist(),
                },
                "random_effects": {
                    "name": random_labels,
                    "conditional_mode": random_effects.tolist(),
                    "conditional_sd": conditional_sd.tolist(),
                },
            },
            extra={
                "parameter_names": parameter_names,
                "unconstrained_parameter_names": list(compiled.fixed_names) + theta_names,
                "population_fitted_values": compiled.X @ final.beta,
                "fixed_design": compiled.X,
                "fixed_effect_covariance": final.beta_covariance,
                "fixed_effect_names": list(compiled.fixed_names),
                "marginal_observation_covariance": final.V,
                "random_effect_observation_covariance": (final.V - residual_matrix(theta_hat)),
                "objective_convention": (
                    "negative exact normalized Gaussian restricted log likelihood"
                    if method == "REML"
                    else "negative exact normalized Gaussian marginal log likelihood"
                ),
                "reference_engine": True,
                "dense_covariance": True,
                "fixed_effect_rank": final.rank,
                "aliased_coefficients": aliased_names,
                "random_covariance_eigenvalues": covariance_eigenvalues,
                "random_covariance_eigenvalue_ratios": covariance_ratios,
                "conditional_random_effect_covariance": conditional_covariance,
                "residual_covariance": residual_matrix(theta_hat),
                "residual_covariance_mode": residual_mode,
                "profiled_fixed_effect_covariance": final.beta_covariance,
                "covariance_parameter_covariance": theta_covariance,
                "likelihood_includes_data_constants": True,
            },
        )


LMMBackend = GaussianLMMBackend
DenseLMMBackend = GaussianLMMBackend


def fit_lmm(data: Any, **options: Any) -> dict[str, Any]:
    """Fit a Gaussian LMM with the dense reference backend."""

    return GaussianLMMBackend().fit(data, **options)


__all__ = ["DenseLMMBackend", "GaussianLMMBackend", "LMMBackend", "fit_lmm"]
