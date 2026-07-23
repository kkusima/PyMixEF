"""Dense Laplace-approximation generalized linear mixed-model backend.

Implemented likelihoods are Bernoulli, binomial-logit, Poisson-log, and
negative-binomial-2-log.  The engine jointly optimizes fixed effects and random
covariance parameters while solving the conditional random-effect mode with an
exact Newton Hessian at every outer evaluation.  Constants in the conditional
probability masses are retained.

Extended families, zero inflation, censoring, AGHQ, and noncanonical links are
rejected explicitly; the reference engine never substitutes PQL or another
scientifically different estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, optimize, special

from .. import families
from .base import (
    BackendInputError,
    BackendNumericalError,
    BackendUnsupportedError,
    cho_solve,
    convergence_mapping,
    covariance_slices,
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


def _family_kind(family: Any) -> str:
    if isinstance(family, families.Bernoulli):
        return "bernoulli"
    if isinstance(family, families.Binomial):
        return "binomial"
    if isinstance(family, families.Poisson):
        return "poisson"
    if isinstance(family, families.NegativeBinomial2) and not isinstance(
        family, families.NegativeBinomial1
    ):
        return "negative-binomial-2"
    if isinstance(family, str):
        name = family
    elif isinstance(family, dict):
        name = str(family.get("name", family.get("family", "")))
    else:
        name = str(getattr(family, "name", family.__class__.__name__))
    normalized = name.lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "binary": "bernoulli",
        "bernoulli": "bernoulli",
        "binomial": "binomial",
        "poisson": "poisson",
        "nb2": "negative-binomial-2",
        "negativebinomial2": "negative-binomial-2",
        "negative-binomial2": "negative-binomial-2",
        "negative-binomial-2": "negative-binomial-2",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise BackendUnsupportedError(
            f"Laplace GLMM does not support family {name!r}",
            details={
                "supported_families": [
                    "Bernoulli(logit)",
                    "Binomial(logit)",
                    "Poisson(log)",
                    "NegativeBinomial2(log)",
                ]
            },
        ) from exc


def _validate_link(family: Any, kind: str) -> None:
    link = getattr(getattr(family, "link", None), "name", None)
    if link is None:
        return
    expected = "logit" if kind in {"bernoulli", "binomial"} else "log"
    if link != expected:
        raise BackendUnsupportedError(
            f"Laplace GLMM supports {kind} only with its canonical {expected} link",
            details={"requested_link": link, "supported_link": expected},
        )


@dataclass(slots=True)
class _ConditionalMode:
    b: NDArray[np.float64]
    hessian: NDArray[np.float64]
    log_probability: NDArray[np.float64]
    score: NDArray[np.float64]
    curvature: NDArray[np.float64]
    converged: bool
    iterations: int
    gradient_norm: float


@dataclass(slots=True)
class _LaplaceEvaluation:
    log_likelihood: float
    beta: NDArray[np.float64]
    G: NDArray[np.float64]
    G_cholesky: NDArray[np.float64]
    mode: _ConditionalMode
    eta: NDArray[np.float64]
    conditional_mean: NDArray[np.float64]
    dispersion: float | None


class LaplaceGLMMBackend:
    """Dense grouped-random-effect Laplace GLMM reference backend."""

    name = "laplace"

    def fit(
        self,
        data: Any,
        *,
        family: Any = None,
        maxiter: int = 500,
        tolerance: float = 1e-7,
        inner_maxiter: int = 100,
        inner_tolerance: float = 1e-9,
        compute_hessian: bool = True,
        **options: Any,
    ) -> dict[str, Any]:
        compiled = prepare_data(data, require_random=True)
        if compiled.residual_covariance is not None:
            raise BackendUnsupportedError(
                "Laplace GLMM does not implement Gaussian residual covariance structures"
            )
        if family is None:
            family = field(compiled.source, "family", default=families.Bernoulli())
        kind = _family_kind(family)
        _validate_link(family, kind)

        y = compiled.y
        if np.any(y != np.floor(y)):
            raise BackendInputError(f"{kind} response must contain integer counts")
        if np.any(y < 0):
            raise BackendInputError(f"{kind} response cannot be negative")
        if kind == "bernoulli" and np.any(y > 1):
            raise BackendInputError("Bernoulli response must contain only zero and one")
        trials = compiled.trials
        if kind == "binomial":
            if trials is None and isinstance(family, families.Binomial):
                raw_trials = family.trials
                if raw_trials is not None:
                    trials = np.broadcast_to(np.asarray(raw_trials, dtype=float), y.shape)
            if trials is None:
                raise BackendInputError("binomial Laplace GLMM requires a trials vector")
            if np.any(trials != np.floor(trials)) or np.any(trials < y):
                raise BackendInputError(
                    "binomial trials must be integer and at least the successes"
                )

        parameterizations = covariance_slices(compiled.random_blocks)
        random_theta_count = sum(item.size for item, _ in parameterizations)
        Z = compiled.random_design
        q = Z.shape[1]
        p = compiled.n_fixed

        fixed_dispersion: float | None = None
        estimate_dispersion = False
        if kind == "negative-binomial-2":
            candidate = field(compiled.source, "dispersion", "nb_dispersion", default=None)
            if candidate is None and isinstance(family, families.NegativeBinomial2):
                candidate = family.fixed_dispersion
            if candidate is None:
                estimate_dispersion = True
            else:
                fixed_dispersion = float(candidate)
                if not np.isfinite(fixed_dispersion) or fixed_dispersion <= 0:
                    raise BackendInputError("NB2 dispersion must be strictly positive")

        def conditional_terms(
            eta: NDArray[np.float64], dispersion: float | None
        ) -> tuple[
            NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
        ]:
            """Log mass, eta-score, negative eta-Hessian, and conditional mean."""

            if kind in {"bernoulli", "binomial"}:
                probability = special.expit(eta)
                n_trials = np.ones_like(y) if kind == "bernoulli" else trials
                assert n_trials is not None
                log_probability = (
                    special.gammaln(n_trials + 1)
                    - special.gammaln(y + 1)
                    - special.gammaln(n_trials - y + 1)
                    + special.xlogy(y, probability)
                    + special.xlog1py(n_trials - y, -probability)
                )
                score = y - n_trials * probability
                curvature = n_trials * probability * (1 - probability)
                conditional_mean = n_trials * probability
            elif kind == "poisson":
                conditional_mean = np.exp(np.clip(eta, -35, 35))
                log_probability = (
                    special.xlogy(y, conditional_mean) - conditional_mean - special.gammaln(y + 1)
                )
                score = y - conditional_mean
                curvature = conditional_mean
            else:
                assert dispersion is not None
                conditional_mean = np.exp(np.clip(eta, -35, 35))
                size = dispersion
                log_probability = (
                    special.gammaln(y + size)
                    - special.gammaln(size)
                    - special.gammaln(y + 1)
                    + size * (np.log(size) - np.log(size + conditional_mean))
                    + y * (np.log(conditional_mean) - np.log(size + conditional_mean))
                )
                score = size * (y - conditional_mean) / (size + conditional_mean)
                curvature = size * conditional_mean * (size + y) / (size + conditional_mean) ** 2
            return log_probability, score, curvature, conditional_mean

        # A fixed-only GLM provides a substantially better deterministic start.
        def fixed_objective(beta: ArrayLike, dispersion: float | None) -> float:
            eta = compiled.X @ np.asarray(beta) + compiled.offset
            log_probability, _, _, _ = conditional_terms(eta, dispersion)
            return float(-np.dot(compiled.weights, log_probability))

        if kind in {"bernoulli", "binomial"}:
            overall = np.clip(
                y.sum() / (y.size if kind == "bernoulli" else np.sum(trials)), 1e-4, 1 - 1e-4
            )
            beta_seed = np.zeros(p)
            if p and np.allclose(compiled.X[:, 0], 1):
                beta_seed[0] = special.logit(overall)
        else:
            beta_seed = np.zeros(p)
            if p and np.allclose(compiled.X[:, 0], 1):
                beta_seed[0] = np.log(max(np.average(y, weights=compiled.weights), 1e-3))
        dispersion_seed = fixed_dispersion if fixed_dispersion is not None else 2.0
        fixed_start = optimize.minimize(
            fixed_objective,
            beta_seed,
            args=(dispersion_seed,),
            method="BFGS",
            options={"maxiter": 200, "gtol": 1e-7},
        )
        beta0 = np.asarray(fixed_start.x if np.all(np.isfinite(fixed_start.x)) else beta_seed)

        random_start = np.concatenate([item.initial(0.4) for item, _ in parameterizations])
        outer_parts = [beta0, random_start]
        if estimate_dispersion:
            outer_parts.append(np.array([np.log(dispersion_seed)]))
        outer0 = np.concatenate(outer_parts)

        def unpack_outer(
            outer: ArrayLike,
        ) -> tuple[NDArray[np.float64], NDArray[np.float64], float | None]:
            outer = np.asarray(outer, dtype=float)
            beta = outer[:p]
            random_theta = outer[p : p + random_theta_count]
            if kind == "negative-binomial-2":
                dispersion = float(np.exp(outer[-1])) if estimate_dispersion else fixed_dispersion
            else:
                dispersion = None
            return beta, random_theta, dispersion

        def solve_mode(
            beta: NDArray[np.float64],
            G: NDArray[np.float64],
            G_cholesky: NDArray[np.float64],
            dispersion: float | None,
        ) -> _ConditionalMode:
            G_inverse = cho_solve(G_cholesky, np.eye(q))
            b = np.zeros(q)

            def values(current: NDArray[np.float64]):
                eta = compiled.X @ beta + Z @ current + compiled.offset
                log_probability, score, curvature, _ = conditional_terms(eta, dispersion)
                negative_joint = float(
                    -np.dot(compiled.weights, log_probability) + 0.5 * current @ G_inverse @ current
                )
                gradient = -Z.T @ (compiled.weights * score) + G_inverse @ current
                hessian = Z.T @ ((compiled.weights * curvature)[:, None] * Z) + G_inverse
                return negative_joint, gradient, hessian, log_probability, score, curvature

            converged = False
            iterations_used = 0
            for iteration in range(1, inner_maxiter + 1):
                iterations_used = iteration
                value, gradient, hessian, log_probability, score, curvature = values(b)
                gradient_norm = float(np.max(np.abs(gradient)))
                if gradient_norm <= inner_tolerance:
                    converged = True
                    break
                try:
                    chol_h, _ = safe_cholesky(hessian)
                    step = cho_solve(chol_h, gradient)
                except linalg.LinAlgError:
                    step = np.linalg.pinv(hessian) @ gradient
                directional = float(gradient @ step)
                step_scale = 1.0
                accepted = False
                for _ in range(30):
                    candidate = b - step_scale * step
                    candidate_value = values(candidate)[0]
                    if (
                        np.isfinite(candidate_value)
                        and candidate_value <= value - 1e-4 * step_scale * directional
                    ):
                        b = candidate
                        accepted = True
                        break
                    step_scale *= 0.5
                if not accepted:
                    break
                if np.max(np.abs(step_scale * step)) <= inner_tolerance * (1 + np.max(np.abs(b))):
                    converged = True
                    break
            value, gradient, hessian, log_probability, score, curvature = values(b)
            return _ConditionalMode(
                b=b,
                hessian=hessian,
                log_probability=log_probability,
                score=score,
                curvature=curvature,
                converged=converged or np.max(np.abs(gradient)) <= np.sqrt(inner_tolerance),
                iterations=iterations_used,
                gradient_norm=float(np.max(np.abs(gradient))),
            )

        def evaluate(outer: ArrayLike) -> _LaplaceEvaluation:
            beta, random_theta, dispersion = unpack_outer(outer)
            G = random_covariance(parameterizations, random_theta)
            G_cholesky, _ = safe_cholesky(G)
            mode = solve_mode(beta, G, G_cholesky, dispersion)
            if not mode.converged and mode.gradient_norm > 1e-4:
                raise linalg.LinAlgError("conditional mode did not converge")
            eta = compiled.X @ beta + Z @ mode.b + compiled.offset
            _, _, _, conditional_mean = conditional_terms(eta, dispersion)
            hessian_cholesky, _ = safe_cholesky(mode.hessian)
            # log ∫ p(y|b)p(b)db under a second-order expansion at b_hat.
            log_likelihood = (
                float(np.dot(compiled.weights, mode.log_probability))
                - 0.5 * float(mode.b @ cho_solve(G_cholesky, mode.b))
                - 0.5 * logdet_from_cholesky(G_cholesky)
                - 0.5 * logdet_from_cholesky(hessian_cholesky)
            )
            return _LaplaceEvaluation(
                log_likelihood=log_likelihood,
                beta=beta,
                G=G,
                G_cholesky=G_cholesky,
                mode=mode,
                eta=eta,
                conditional_mean=conditional_mean,
                dispersion=dispersion,
            )

        def objective(outer: ArrayLike) -> float:
            try:
                value = -evaluate(outer).log_likelihood
                return value if np.isfinite(value) else 1e100
            except (ValueError, FloatingPointError, OverflowError, linalg.LinAlgError):
                return 1e100

        bounds = [(-30.0, 30.0)] * p + [(-10.0, 10.0)] * random_theta_count
        if estimate_dispersion:
            bounds.append((-10.0, 12.0))
        result = optimize.minimize(
            objective,
            outer0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": tolerance, "gtol": tolerance},
        )
        initial_gradient = np.asarray(getattr(result, "jac", np.array([np.inf])))
        if (
            not result.success
            or not np.isfinite(result.fun)
            or np.max(np.abs(initial_gradient)) > max(np.sqrt(tolerance), 5e-4)
        ):
            rescue = optimize.minimize(
                objective,
                np.clip(result.x if np.all(np.isfinite(result.x)) else outer0, -10, 10),
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
                objective_tolerance=1e-7,
            )

        outer_hat = np.asarray(result.x, dtype=float)
        try:
            final = evaluate(outer_hat)
        except (ValueError, linalg.LinAlgError) as exc:
            raise BackendNumericalError(
                "Laplace GLMM failed to obtain a valid final conditional mode",
                details={"optimizer_message": str(result.message)},
            ) from exc

        outer_covariance, hessian_positive, curvature_source = optimizer_covariance(
            objective,
            outer_hat,
            result,
            compute_hessian=compute_hessian,
            finite_difference_limit=14,
        )

        beta, random_theta_hat, dispersion_hat = unpack_outer(outer_hat)
        natural_parameters: dict[str, float] = {
            name: float(value) for name, value in zip(compiled.fixed_names, beta, strict=True)
        }
        unconstrained_parameters: dict[str, float] = dict(natural_parameters)
        natural_covariance_names: list[str] = []
        raw_covariance_names: list[str] = []
        covariance_ratios: list[float] = []
        for parameterization, section in parameterizations:
            values = parameterization.names_and_values(random_theta_hat[section])
            natural_parameters.update(values)
            natural_covariance_names.extend(values)
            raw_names = parameterization.unconstrained_names()
            raw_covariance_names.extend(raw_names)
            unconstrained_parameters.update(
                {
                    name: float(value)
                    for name, value in zip(raw_names, random_theta_hat[section], strict=True)
                }
            )
            eigenvalues = np.linalg.eigvalsh(parameterization.matrix(random_theta_hat[section]))
            covariance_ratios.append(
                float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > 0 else 0.0
            )
        if kind == "negative-binomial-2":
            assert dispersion_hat is not None
            natural_parameters["dispersion"] = float(dispersion_hat)
            if estimate_dispersion:
                unconstrained_parameters["log_dispersion"] = float(outer_hat[-1])

        # Delta method from outer optimizer scale to the reported natural scale.
        def natural_vector(outer: NDArray[np.float64]) -> NDArray[np.float64]:
            beta_value, theta_value, dispersion_value = unpack_outer(outer)
            values = list(beta_value)
            for parameterization, section in parameterizations:
                values.extend(parameterization.names_and_values(theta_value[section]).values())
            if kind == "negative-binomial-2":
                assert dispersion_value is not None
                values.append(float(dispersion_value))
            return np.asarray(values)

        natural_hat = natural_vector(outer_hat)
        jacobian = np.empty((natural_hat.size, outer_hat.size))
        for column in range(outer_hat.size):
            step = 1e-5 * max(1.0, abs(outer_hat[column]))
            delta = np.zeros(outer_hat.size)
            delta[column] = step
            jacobian[:, column] = (
                natural_vector(outer_hat + delta) - natural_vector(outer_hat - delta)
            ) / (2 * step)
        parameter_covariance = jacobian @ outer_covariance @ jacobian.T

        conditional_covariance = np.linalg.pinv(final.mode.hessian, rcond=1e-10)
        conditional_sd = np.sqrt(np.maximum(np.diag(conditional_covariance), 0.0))
        random_labels = [
            label for block in compiled.random_blocks for label in block.coefficient_labels()
        ]
        residuals = y - final.conditional_mean
        boundary_names = [
            name
            for name, value in natural_parameters.items()
            if name.startswith("sd(") and value < 1e-5
        ]
        singular = bool((covariance_ratios and min(covariance_ratios) < 1e-6) or boundary_names)
        separated = bool(
            kind in {"bernoulli", "binomial"}
            and (
                np.max(np.abs(beta)) > 8
                or np.any((special.expit(final.eta) < 1e-7) | (special.expit(final.eta) > 1 - 1e-7))
            )
        )
        warnings: list[dict[str, Any]] = []
        if singular:
            warnings.append(
                {
                    "code": "COV-SINGULAR-001",
                    "severity": "warning",
                    "message": "random-effect covariance is singular or near-singular",
                    "details": {"eigenvalue_ratios": covariance_ratios},
                }
            )
        if separated:
            warnings.append(
                {
                    "code": "GLMM-SEPARATION-001",
                    "severity": "warning",
                    "message": "binary fit shows separation or near-separation indicators",
                    "details": {"maximum_absolute_fixed_effect": float(np.max(np.abs(beta)))},
                }
            )
        if not final.mode.converged:
            warnings.append(
                {
                    "code": "GLMM-MODE-001",
                    "severity": "warning",
                    "message": "conditional mode met only the relaxed gradient tolerance",
                    "details": {"gradient_inf_norm": final.mode.gradient_norm},
                }
            )

        result_gradient = getattr(result, "jac", None)
        if result_gradient is None or np.any(~np.isfinite(result_gradient)):
            result_gradient = finite_gradient(objective, outer_hat)
        fixed_effect_rank = int(np.linalg.matrix_rank(compiled.X))
        convergence = convergence_mapping(
            success=bool(result.success and np.isfinite(result.fun)),
            message=str(result.message),
            iterations=int(getattr(result, "nit", 0)),
            function_evaluations=int(getattr(result, "nfev", 0)),
            gradient=result_gradient,
            hessian_positive_definite=hessian_positive,
            singular=singular,
            boundary_parameters=boundary_names,
            warnings=warnings,
            extra={
                "conditional_mode_converged": final.mode.converged,
                "conditional_mode_iterations": final.mode.iterations,
                "conditional_mode_gradient_inf_norm": final.mode.gradient_norm,
                "separation_flag": separated,
                "fixed_effect_rank": fixed_effect_rank,
                "curvature_source": curvature_source,
            },
        )
        parameter_names = list(compiled.fixed_names) + natural_covariance_names
        if kind == "negative-binomial-2":
            parameter_names.append("dispersion")
        raw_names = list(compiled.fixed_names) + raw_covariance_names
        if estimate_dispersion:
            raw_names.append("log_dispersion")

        return make_payload(
            parameters=natural_parameters,
            unconstrained_parameters=unconstrained_parameters,
            parameter_covariance=parameter_covariance,
            fitted_values=final.conditional_mean,
            residuals=residuals,
            random_effects=final.mode.b,
            objective=float(-final.log_likelihood),
            log_likelihood=float(final.log_likelihood),
            method="Laplace",
            engine=self.name,
            convergence=convergence,
            diagnostic_data={
                "observations": {
                    "observed": y.tolist(),
                    "linear_predictor_conditional": final.eta.tolist(),
                    "fitted_conditional": final.conditional_mean.tolist(),
                    "response_residual": residuals.tolist(),
                },
                "random_effects": {
                    "name": random_labels,
                    "conditional_mode": final.mode.b.tolist(),
                    "conditional_sd_laplace": conditional_sd.tolist(),
                },
            },
            extra={
                "parameter_names": parameter_names,
                "unconstrained_parameter_names": raw_names,
                "family": kind,
                "approximation": "first-order Laplace at the joint conditional mode",
                "quadrature_order": 1,
                "objective_convention": "negative normalized Laplace-approximated marginal log likelihood",
                "likelihood_includes_data_constants": True,
                "conditional_random_effect_covariance_laplace": conditional_covariance,
                "outer_parameter_covariance": outer_covariance,
                "random_covariance_eigenvalue_ratios": covariance_ratios,
                "reference_engine": True,
                "dense_covariance": True,
                "fixed_effect_rank": fixed_effect_rank,
                "unsupported_extensions": [
                    "AGHQ",
                    "zero inflation",
                    "hurdle likelihood",
                    "ordinal/multinomial",
                    "truncation/censoring",
                    "noncanonical links",
                ],
            },
        )


GLMMBackend = LaplaceGLMMBackend


def fit_glmm(data: Any, **options: Any) -> dict[str, Any]:
    """Fit a supported GLMM by dense first-order Laplace approximation."""

    return LaplaceGLMMBackend().fit(data, **options)


__all__ = ["GLMMBackend", "LaplaceGLMMBackend", "fit_glmm"]
