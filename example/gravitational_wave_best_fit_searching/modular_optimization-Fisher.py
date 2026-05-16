import os
import json
import time
import argparse
from typing import Tuple, Optional

import numpy as np
from scipy.optimize import minimize

# FEW / waveform & noise
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens
from stableemrifisher.utils import generate_PSD, inner_product
from stableemrifisher.fisher import StableEMRIFisher

import utils


# -----------------------------
# PARIS global context (picklable functions require module scope)
# -----------------------------
_PARIS_REF_CENTER = None          # type: Optional[np.ndarray]
_PARIS_SPREAD_SCALE = None        # type: Optional[float]
_PARIS_OBJECTIVE = None           # type: Optional[callable]
_PARIS_TARGET_KIND = None         # type: Optional[str]  # 'optimal_snr', 'optimal_snr_phase_max', 'phase_match'
_PARIS_EARLY_STOP_HIT = False

# Fisher-parallelotope affine prior (primary for this script)
_PARIS_AFFINE_CENTER = None       # type: Optional[np.ndarray]
_PARIS_AFFINE_Q = None            # type: Optional[np.ndarray]
_PARIS_AFFINE_B = None            # type: Optional[np.ndarray]
_PARIS_DIM = None                 # type: Optional[int]


def _clip_physical_params(theta: np.ndarray, pa_template: str) -> np.ndarray:
    """Clip mapped physical parameters to minimal physical ranges.

    Indices: 0:m1, 1:m2, 2:a, 3:p0, 4:e0, [5:chi2], [last:Phi_phi0]
    """
    x = np.asarray(theta, dtype=float).copy()
    if x.ndim == 1:
        if x.shape[0] >= 1: x[0] = np.clip(x[0], 1e4, 1e7)
        if x.shape[0] >= 2: x[1] = np.clip(x[1], 1e0, 1e4)
        if x.shape[0] >= 3: x[2] = np.clip(x[2], -0.999, 0.999)
        if x.shape[0] >= 4: x[3] = np.clip(x[3], 6.0, 100.0) # p0
        if x.shape[0] >= 5: x[4] = np.clip(x[4], 1e-4, 0.9)   # e0
        
        # Hard physical boundary for SuperKludge
        if x.shape[0] >= 5:
            if x[3] < 6.0 + 2.0 * x[4]:
                x[3] = 6.0 + 2.0 * x[4] + 0.1
        
        if pa_template == '1PA':
            if x.shape[0] >= 6: x[5] = np.clip(x[5], -0.99, 0.99)
            if x.shape[0] >= 7: x[6] = x[6] % (2 * np.pi)
        else: # 0PA
            if x.shape[0] >= 6: x[5] = x[5] % (2 * np.pi)
        return x
    else:
        x[:, 0] = np.clip(x[:, 0], 1e4, 1e7)
        x[:, 1] = np.clip(x[:, 1], 1e0, 1e4)
        if x.shape[1] >= 3: x[:, 2] = np.clip(x[:, 2], -0.999, 0.999)
        if x.shape[1] >= 4: x[:, 3] = np.clip(x[:, 3], 6.0, 100.0)
        if x.shape[1] >= 5: x[:, 4] = np.clip(x[:, 4], 1e-4, 0.9)
        
        # Hard physical boundary check for batch
        for i in range(x.shape[0]):
            if x[i, 3] < 6.0 + 2.0 * x[i, 4]:
                x[i, 3] = 6.0 + 2.0 * x[i, 4] + 0.1
                
        if pa_template == '1PA':
            if x.shape[1] >= 6: x[:, 5] = np.clip(x[:, 5], -0.99, 0.99)
            if x.shape[1] >= 7: x[:, 6] = x[:, 6] % (2 * np.pi)
        else:
            if x.shape[1] >= 6: x[:, 5] = x[:, 5] % (2 * np.pi)
        return x


def paris_prior_transform(u):
    """Prior transform using Fisher-parallelotope when configured."""
    u = np.asarray(u, dtype=float)
    if _PARIS_AFFINE_CENTER is not None and _PARIS_AFFINE_Q is not None and _PARIS_AFFINE_B is not None:
        center = _PARIS_AFFINE_CENTER
        Q = _PARIS_AFFINE_Q
        b = _PARIS_AFFINE_B
        dim = _PARIS_DIM

        def map_one(u1):
            t = 2.0 * np.asarray(u1)[:dim] - 1.0
            return center + Q @ (b * t)

        if u.ndim == 1:
            theta = map_one(u)
            pa_tmpl = '1PA' if dim >= 7 else '0PA'
            return _clip_physical_params(theta, pa_tmpl)
        else:
            out = np.zeros((u.shape[0], dim), dtype=float)
            for i in range(u.shape[0]):
                out[i] = map_one(u[i])
            pa_tmpl = '1PA' if dim >= 7 else '0PA'
            return _clip_physical_params(out, pa_tmpl)

    # Legacy multiplicative band (fallback)
    ref = _PARIS_REF_CENTER
    s = _PARIS_SPREAD_SCALE
    u = np.asarray(u)
    return ref * (1 - s + u * 2 * s)


def paris_inverse_prior_transform(params):
    """Inverse of paris_prior_transform.

    For affine Fisher mapping: t = diag(1/b) Q^T (theta - center), u = 0.5*(t+1).
    Falls back to multiplicative inverse if affine not configured.
    """
    theta = np.asarray(params, dtype=float)
    if _PARIS_AFFINE_CENTER is not None and _PARIS_AFFINE_Q is not None and _PARIS_AFFINE_B is not None:
        center = _PARIS_AFFINE_CENTER
        Q = _PARIS_AFFINE_Q
        b = _PARIS_AFFINE_B
        inv_b = 1.0 / b

        def inv_one(th):
            d = np.asarray(th) - center
            t = (Q.T @ d) * inv_b
            return 0.5 * (t + 1.0)

        if theta.ndim == 1:
            return inv_one(theta)
        out = np.zeros_like(theta, dtype=float)
        for i in range(theta.shape[0]):
            out[i] = inv_one(theta[i])
        return out

    ref = _PARIS_REF_CENTER
    s = _PARIS_SPREAD_SCALE
    return (theta / ref - (1 - s)) / (2 * s)


def paris_log_density(params):
    """Top-level log-density (actually score) wrapper with early-stop.

    - Receives physical parameters (after prior_transform) per parismc contract.
    - Returns a scalar/array of scores (larger is better). After early-stop trigger,
      subsequent evaluations return -inf to end sampling quickly.
    """
    global _PARIS_EARLY_STOP_HIT

    params = np.asarray(params)

    def eval_one(x):
        global _PARIS_EARLY_STOP_HIT
        if _PARIS_EARLY_STOP_HIT:
            return float('-inf')
        try:    
            val = float(_PARIS_OBJECTIVE(x))
        except Exception:
            return float('-inf')
        # Early-stop policy per user spec
        if _PARIS_TARGET_KIND in ('optimal_snr', 'optimal_snr_phase_max'):
            if val >= 19.0:
                _PARIS_EARLY_STOP_HIT = True
                try:
                    print(f"[EARLY-STOP] SNR {val:.6f} >= 19; future calls => -inf")
                except Exception:
                    pass
        elif _PARIS_TARGET_KIND == 'phase_match':
            # score = -phase_diff; trigger when phase_diff < 1.5 => score > -1.5
            if val > -1.5:
                _PARIS_EARLY_STOP_HIT = True
                try:
                    print(f"[EARLY-STOP] phase-diff {-val:.6f} < 1.5; future calls => -inf")
                except Exception:
                    pass
        return val

    if params.ndim == 1:
        return eval_one(params)
    out = np.zeros(params.shape[0], dtype=float)
    for i in range(params.shape[0]):
        out[i] = eval_one(params[i])
    return out


def compute_fisher_parallelotope(ctx: dict,
                                 theta0: np.ndarray,
                                 pa_template: str,
                                 use_gpu: bool = True, prior_sigma_range: float = 100) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Compute Fisher-based parallelotope (±10σ at target SNR) around theta0.

    Returns (Q, b, meta):
      - Q: eigenvectors of covariance (d x d)
      - b: side half-lengths along Q, b_i = 10*sqrt(eigval_i)
      - meta: {'diag_sigma', 'eigvals', 'snr_model', 'scale_applied', 'reg_added'}
    """
    # Prepare waveform generator
    if 'waveform_response' in ctx and ctx['waveform_response'] is not None:
        waveform_response = ctx['waveform_response']
    else:
        waveform_response = build_waveform_response(T=ctx['T'], dt=ctx['dt'], use_gpu=use_gpu)

    channels = [A1TDISens, E1TDISens, T1TDISens]
    noise_kwargs = [{"sens_fn": ch} for ch in channels]

    # Parameter names and flags (use 5D derivatives as in old_method; chi2 fixed via add_param_args when 1PA)
    param_names = ['m1', 'm2', 'a', 'p0', 'e0']
    use_1pa = (pa_template == '1PA')

    # Build SEF configuration
    add_param_args = {
        'chi2': float(ctx['chi2']),
        '1PA': bool(use_1pa),
        'evolve_primary': False,
        '2PA': False,
    }
    sef_kwargs = {
        'EMRI_waveform_gen': waveform_response,
        'param_names': param_names,
        'der_order': 4,
        'Ndelta': 12,
        'use_gpu': bool(use_gpu),
        'noise_model': get_sensitivity,
        'channels': channels,
        'noise_kwargs': noise_kwargs,
        'add_param_args': add_param_args,
        'deltas': None,
    }
    # Fisher parameter vector (only 5 core dims positionally)
    fisher_params = list(np.asarray(theta0[:5], dtype=float)) + [
        float(ctx['Y0']), float(ctx['dist']), float(ctx['qS']), float(ctx['phiS']),
        float(ctx['qK']), float(ctx['phiK']), float(ctx['Phi_phi0']), float(ctx['Phi_theta0']), float(ctx['Phi_r0'])
    ]

    # Initialize SEF and compute Fisher
    try:
        # Follow old_method.py style: pass T, dt as keyword args
        sef = StableEMRIFisher(*fisher_params, T=float(ctx['T']), dt=float(ctx['dt']), **sef_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"StableEMRIFisher init failed with kwargs T,dt: {e} | "
            f"len(fisher_params)={len(fisher_params)}, param_names={param_names}"
        )
    try:
        # Initialize internal PSD/noise and window via SEF API
        sef.SNRcalc_SEF()
        if not hasattr(sef, 'deltas') or sef.deltas is None:
            sef.Fisher_Stability()
        F = np.asarray(sef.FisherCalc(), dtype=float)
    except Exception as e:
        raise RuntimeError(f"Fisher computation failed: {e}")

    # Compute model SNR at theta0 (template order)
    if pa_template == '0PA':
        m1, m2, a, p0, e0 = theta0[:5]
        chi2 = float(ctx['chi2'])
        wave_params = [m1, m2, a, p0, e0, ctx['Y0'],
                       ctx['dist'], ctx['qS'], ctx['phiS'], ctx['qK'], ctx['phiK'],
                       ctx['Phi_phi0'], ctx['Phi_theta0'], ctx['Phi_r0'],
                       chi2, False, False, False]
        emri_flags = {"T": ctx['T'], "dt": ctx['dt'], '1PA': False, 'evolve_primary': False, '2PA': False}
    else:
        m1, m2, a, p0, e0, chi2 = theta0[:6]
        wave_params = [m1, m2, a, p0, e0, ctx['Y0'],
                       ctx['dist'], ctx['qS'], ctx['phiS'], ctx['qK'], ctx['phiK'],
                       ctx['Phi_phi0'], ctx['Phi_theta0'], ctx['Phi_r0'],
                       chi2, True, False, False]
        emri_flags = {"T": ctx['T'], "dt": ctx['dt'], '1PA': True, 'evolve_primary': False, '2PA': False}

    waveform_tmpl = waveform_response(*wave_params, **emri_flags)
    PSD_funcs = generate_PSD(waveform=waveform_tmpl, dt=float(ctx['dt']), noise_PSD=get_sensitivity,
                             channels=channels, noise_kwargs=noise_kwargs, use_gpu=bool(use_gpu))
    snr_model = float(np.sqrt(inner_product(waveform_tmpl, waveform_tmpl, PSD_funcs, float(ctx['dt']), use_gpu=bool(use_gpu))))

    # Scale Fisher to target SNR
    scale = (utils.TARGET_SNR / max(snr_model, 1e-30)) ** 2
    F_scaled = F * scale

    # Regularize and invert to covariance
    dim = F_scaled.shape[0]
    reg = 1e-12 * np.trace(F_scaled) / max(dim, 1)
    F_scaled = F_scaled + reg * np.eye(dim)
    try:
        cov = np.linalg.inv(F_scaled)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(F_scaled)
    cov = 0.5 * (cov + cov.T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-30)
    sigma_diag = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    b = prior_sigma_range * np.sqrt(evals)

    meta = {
        'diag_sigma': sigma_diag.tolist(),
        'eigvals': evals.tolist(),
        'snr_model': float(snr_model),
        'scale_applied': float(scale),
        'reg_added': float(reg),
    }
    if pa_template == '1PA':
        # Extend to 6D with independent chi2 axis
        Q_ext = np.eye(6)
        Q_ext[:5, :5] = evecs
        b_chi2 = 0.99
        b_ext = np.concatenate([b, [b_chi2]])
        meta.update({'chi2_axis': 'independent', 'b_chi2': float(b_chi2)})
        return Q_ext, b_ext, meta
    return evecs, b, meta


def covariance_from_fisher_parallelotope(Q: np.ndarray, b: np.ndarray, pa_template: str, prior_sigma_range: float = 100) -> np.ndarray:
    """Construct covariance matrix from Fisher-parallelotope outputs (Q, b).

    Given that b = 10 * sqrt(eigvals), we recover eigvals = (b/10)^2 and return Q diag(eigvals) Q^T.
    """
    b = np.asarray(b, dtype=float)
    eigvals = (b / prior_sigma_range) ** 2
    cov = Q @ (np.diag(eigvals)) @ Q.T
    cov = 0.5 * (cov + cov.T)
    return cov


def build_waveform_response(T: float, dt: float, use_gpu: bool = False) -> ResponseWrapper:
    """Create a LISA ResponseWrapper consistent with existing modules."""
    from few.waveform import GenerateEMRIWaveform
    from few.waveform.waveform import SuperKludgeWaveform

    sum_kwargs = dict(pad_output=True, odd_len=True)
    waveform_model = GenerateEMRIWaveform(SuperKludgeWaveform, sum_kwargs=sum_kwargs, return_list=False)

    t0 = 10000.0
    tdi_gen = "2nd generation"
    order = 20
    index_lambda = 8  # phiS
    index_beta = 7    # qS

    response = ResponseWrapper(
        waveform_gen=waveform_model,
        Tobs=T,
        t0=t0,
        dt=dt,
        index_lambda=index_lambda,
        index_beta=index_beta,
        flip_hx=True,
        use_gpu=use_gpu,
        is_ecliptic_latitude=False,
        remove_garbage="zero",
        orbits=EqualArmlengthOrbits(use_gpu=use_gpu),
        order=order,
        tdi=tdi_gen,
        tdi_chan="AET",
    )
    print("[INFO] Finished loading modules and building ResponseWrapper")
    return response


def prepare_2pa_fiducial(signal_row: np.ndarray, use_gpu: bool = False):
    """
    Build fiducial 2PA waveform, PSD, and FFT from a signal parameter row.

    signal_row columns:
      [m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, dt, T, chi2]
    """
    (
        m1, m2, a, p0, e0, Y0,
        dist, qS, phiS, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0,
        dt, T, chi2
    ) = signal_row

    waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu)

    evolve_1PA = True
    evolve_primary = False
    evolve_2PA = True

    wave_params = [
        m1, m2, a, p0, e0, Y0,
        dist, qS, phiS, qK, phiK,
        Phi_phi0, Phi_theta0, Phi_r0,
        chi2, evolve_1PA, evolve_primary, evolve_2PA,
    ]
    emri_kwargs = {"T": T, "dt": dt, '1PA': evolve_1PA, 'evolve_primary': evolve_primary, '2PA': evolve_2PA}

    waveform_2pa = waveform_response(*wave_params, **emri_kwargs)

    channels = [A1TDISens, E1TDISens, T1TDISens]
    noise_kwargs = [{"sens_fn": ch} for ch in channels]
    PSD_funcs = generate_PSD(
        waveform=waveform_2pa,
        dt=dt,
        noise_PSD=get_sensitivity,
        channels=channels,
        noise_kwargs=noise_kwargs,
        use_gpu=use_gpu,
    )

    # Verify SNR level (grid builder normalized dist to target PA2 SNR already)
    snr = float(np.sqrt(inner_product(waveform_2pa, waveform_2pa, PSD_funcs, dt, use_gpu=use_gpu)))

    waveform_2pa_fft = utils.compute_fft_with_windowing(waveform_2pa, dt, use_gpu=use_gpu)
    N_fiducial = len(waveform_2pa[0])
    print("[INFO] Finished preparing 2PA waveform (GPU)")

    return {
        'm1': m1, 'm2': m2, 'a': a, 'p0': p0, 'e0': e0, 'Y0': Y0,
        'dist': dist, 'qS': qS, 'phiS': phiS, 'qK': qK, 'phiK': phiK,
        'Phi_phi0': Phi_phi0, 'Phi_theta0': Phi_theta0, 'Phi_r0': Phi_r0,
        'dt': dt, 'T': T, 'chi2': chi2,
        'waveform_response': waveform_response,
        'PSD_funcs': PSD_funcs,
        'waveform_2pa_fft': waveform_2pa_fft,
        'N_fiducial': N_fiducial,
        'snr': snr,
    }


def objective_factory(pa_template: str,
                      target_func: str,
                      ctx: dict,
                      maximize_phase: bool = False,
                      maximize_time: bool = False,
                      use_gpu_for_snr: bool = True,
                      global_best_no_max: dict = None):
    """
    Build a score(theta) where larger is better.
    theta now includes Phi_phi0 as the last element.
    """
    if global_best_no_max is None:
        global_best_no_max = {"snr": -1e60, "theta": None}
    
    if target_func in ('optimal_snr', 'optimal_snr_phase_max'):
        fixed = {
            'waveform_response': ctx['waveform_response'],
            'PSD': ctx['PSD_funcs'],
            'dt': ctx['dt'],
            'T': ctx['T'],
            'N_fiducial': ctx['N_fiducial'],
            'waveform_2pa_fft': ctx['waveform_2pa_fft'],
            'xp': np,
            'use_gpu': bool(use_gpu_for_snr),
            'maximize_phase': bool(maximize_phase),
            'maximize_time': bool(maximize_time),
        }
    else:
        fixed = {}

    best_score_internal = [-1e60]
    iter_count = [0]

    def score_optimal_snr(theta: np.ndarray) -> float:
        iter_count[0] += 1
        # Explicit clipping and NaN protection
        theta = _clip_physical_params(theta, pa_template)
        
        # SuperKludge physical boundary: p0 >= 6 + 2*e0
        if theta[3] < 6.0 + 2.0 * theta[4]:
            return -1e20
            
        phi_phi0 = theta[-1] if not fixed.get('maximize_phase') else ctx['Phi_phi0']

        try:
            if pa_template == '0PA':
                m1, m2, a, p0, e0 = theta[:5]
                chi2 = ctx['chi2']
                ov = utils.calculate_optimal_snr_0pa_vs_2pa(
                    m1, m2, a, p0, e0, ctx['Y0'], 
                    ctx['dist'], ctx['qS'], ctx['phiS'], ctx['qK'], ctx['phiK'],
                    phi_phi0, ctx['Phi_theta0'], ctx['Phi_r0'],
                    chi2, **fixed,
                )
            else:  # '1PA'
                m1, m2, a, p0, e0, chi2 = theta[:6]
                ov = utils.calculate_optimal_snr_1pa_vs_2pa(
                    m1, m2, a, p0, e0, ctx['Y0'], 
                    ctx['dist'], ctx['qS'], ctx['phiS'], ctx['qK'], ctx['phiK'],
                    phi_phi0, ctx['Phi_theta0'], ctx['Phi_r0'],
                    chi2, **fixed,
                )
            
            snr_val = float(ov * utils.TARGET_SNR)
            if np.isnan(snr_val) or np.isinf(snr_val):
                return -1e20
        except Exception:
            return -1e20

        if snr_val > best_score_internal[0]:
            best_score_internal[0] = snr_val
            # Quick check for No-Max mismatch update
            if not fixed.get('maximize_time') and not fixed.get('maximize_phase'):
                if snr_val > global_best_no_max["snr"]:
                    global_best_no_max["snr"] = snr_val
                    global_best_no_max["theta"] = theta.copy()
            
            if iter_count[0] % 50 == 0:
                print(f"[ITER: {iter_count[0]} | BEST SNR: {snr_val:.6f}]")
                
        return snr_val

    if target_func in ('optimal_snr', 'optimal_snr_phase_max'):
        return score_optimal_snr
    else:
        return score_phase_match


def nelder_mead_optimize(theta0: np.ndarray, objective, maxiter: int = 500, xatol: float = 1e-8, fatol: float = 1e-10):
    res = minimize(
        objective,
        theta0,
        method='Nelder-Mead',
        options={'maxiter': maxiter, 'maxfev': maxiter + 100, 'xatol': xatol, 'fatol': fatol},
    )
    return res


def run_paris(ndim: int,
              prior_center: np.ndarray,
              score_func,
              spread_scale: float,
              savepath: str,
              seed_cloud: int = 1000,
              seed_jitter: float = 1e-4,
              target_kind: str = None,
              lhs_save_dir: Optional[str] = None,
              affine_Q: Optional[np.ndarray] = None,
              affine_b: Optional[np.ndarray] = None,
              num_iterations: int = 1000,
              extra_physical_seeds: Optional[np.ndarray] = None):
    """Run PARIS sampler maximizing a score."""
    import parismc

    os.makedirs(savepath, exist_ok=True)

    global _PARIS_REF_CENTER, _PARIS_SPREAD_SCALE, _PARIS_OBJECTIVE, _PARIS_TARGET_KIND, _PARIS_EARLY_STOP_HIT
    global _PARIS_AFFINE_CENTER, _PARIS_AFFINE_Q, _PARIS_AFFINE_B, _PARIS_DIM
    _PARIS_REF_CENTER = np.asarray(prior_center, dtype=float).copy()
    _PARIS_SPREAD_SCALE = float(spread_scale)
    _PARIS_OBJECTIVE = score_func
    _PARIS_TARGET_KIND = target_kind
    _PARIS_EARLY_STOP_HIT = False
    _PARIS_DIM = ndim

    if affine_Q is not None and affine_b is not None:
        _PARIS_AFFINE_CENTER = np.asarray(prior_center, dtype=float).copy()
        _PARIS_AFFINE_Q = np.asarray(affine_Q, dtype=float).copy()
        _PARIS_AFFINE_B = np.asarray(affine_b, dtype=float).copy()
    else:
        _PARIS_AFFINE_CENTER = None
        _PARIS_AFFINE_Q = None
        _PARIS_AFFINE_B = None

    n_seed = 30 # Increased from 25
    sigma = 1e-2 
    init_cov_list = [sigma**2 * np.eye(ndim) for _ in range(n_seed)]
    config = parismc.SamplerConfig(
        merge_type='single',
        alpha=1200, # Increased from 1000
        trail_size=int(1e3),
        boundary_limiting=True,
        use_beta=True,
        integral_num=int(1e5),
        gamma=30, # Decreased from 40 for more frequent adaptation
        exclude_scale_z=np.inf,
        use_pool=False, 
        n_pool=36,
    )

    sampler = parismc.Sampler(
        ndim=ndim,
        n_seed=n_seed,
        log_density_func=paris_log_density,
        init_cov_list=init_cov_list,
        prior_transform=paris_prior_transform,
        config=config,
    )

    point_blocks = []
    log_blocks = []

    if extra_physical_seeds is not None:
        print(f"[PARIS] Adding {len(extra_physical_seeds)} physical seeds")
        u_seeds = paris_inverse_prior_transform(extra_physical_seeds)
        # Handle cases where inverse transform might go slightly out of [0, 1] due to clipping/rounding
        u_seeds = np.clip(u_seeds, 1e-6, 1.0 - 1e-6)
        val_seeds = paris_log_density(paris_prior_transform(u_seeds))
        point_blocks.append(u_seeds.reshape(-1, ndim))
        log_blocks.append(np.asarray(val_seeds, dtype=float).reshape(-1))

    n_samples = max(0, int(seed_cloud))
    if n_samples > 0:
        from smt.sampling_methods import LHS
        xlimits = np.column_stack([np.zeros(ndim), np.ones(ndim)])
        sampling = LHS(xlimits=xlimits)
        lhs_points = sampling(n_samples)
        lhs_vals = paris_log_density(paris_prior_transform(lhs_points))
        point_blocks.append(lhs_points)
        log_blocks.append(np.asarray(lhs_vals, dtype=float).reshape(-1))

    external_lhs_points = np.vstack(point_blocks)
    external_lhs_log_densities = np.concatenate(log_blocks)

    sampler.run_sampling(
        num_iterations=num_iterations,
        savepath=savepath,
        print_iter=100,
        external_lhs_points=external_lhs_points,
        external_lhs_log_densities=external_lhs_log_densities,
    )

    return sampler, paris_prior_transform, external_lhs_points


def main():
    start_time_all = time.time()
    parser = argparse.ArgumentParser(description="Modular optimization (0PA/1PA vs 2PA)")
    
    parser.add_argument('--optimizer', choices=['nelder-mead', 'PARIS'], default='PARIS')
    parser.add_argument('--target-func', choices=['optimal_snr', 'optimal_snr_phase_max'], default='optimal_snr_phase_max')
    parser.add_argument('--startingpoints', type=str, default='signal_parameter_array_IMRI.npy', help='Optional .npy path for starting points')
    
    parser.add_argument('--grid-index', type=int, default=None)
    parser.add_argument('--force-fresh', action='store_true')
    parser.add_argument('--seed-cloud', type=int, default=1200)

    args = parser.parse_args()

    scenario = utils.scenario
    pa_template = utils.pa_template

    print(f"[INFO] Running optimization for Scenario: {scenario}, Template: {pa_template}")

    signal_param_array = utils.load_signal_param_array()
    startpoint_array = utils.load_startingpoint_param_array(filename=args.startingpoints, allow_missing=True)
    num_pts = signal_param_array.shape[0]

    params_filename = f"optimized_params_{scenario}_{pa_template}.npy"
    mismatch_filename = f"optimized_mismatch_{scenario}_{pa_template}.npy"

    if not args.force_fresh and os.path.exists(params_filename) and os.path.exists(mismatch_filename):
        print(f"[INIT] Loading existing results from {params_filename}")
        all_optimized_params = np.load(params_filename)
        all_optimized_mismatch = np.load(mismatch_filename)
    else:
        all_optimized_params = startpoint_array.copy() if startpoint_array is not None else signal_param_array.copy()
        all_optimized_mismatch = np.ones(num_pts)

    # Easiest-First Targeting (Prioritize signals closest to 0.1 threshold)
    if args.grid_index is not None:
        indices = [args.grid_index]
    else:
        target_idx = -1
        min_miss = 2.0  # Mismatch is bounded by 1.0 (or 2.0 for some definitions)
        for i in range(num_pts):
            if all_optimized_mismatch[i] > 0.1:
                if all_optimized_mismatch[i] < min_miss:
                    min_miss = all_optimized_mismatch[i]
                    target_idx = i
        if target_idx == -1:
            print("[FINISH] All signals satisfy mismatch < 0.1.")
            return
        indices = [target_idx]

    sampler_base = "paris_samplers"

    for idx in indices:
        sig_row = signal_param_array[idx]
        print(f'\n--- Processing Index {idx} (Current Mismatch: {all_optimized_mismatch[idx]:.4f}) ---')
        ctx = prepare_2pa_fiducial(sig_row, use_gpu=True)

        current_best_row = all_optimized_params[idx]
        if pa_template == '0PA':
            # m1, m2, a, p0, e0, Phi_phi0
            theta0 = np.array([current_best_row[0], current_best_row[1], current_best_row[2], current_best_row[3], current_best_row[4], current_best_row[11]], dtype=float)
            theta_true = np.array([sig_row[0], sig_row[1], sig_row[2], sig_row[3], sig_row[4], sig_row[11]], dtype=float)
            ndim = 6
        else:
            # m1, m2, a, p0, e0, chi2, Phi_phi0
            theta0 = np.array([current_best_row[0], current_best_row[1], current_best_row[2], current_best_row[3], current_best_row[4], current_best_row[16], current_best_row[11]], dtype=float)
            theta_true = np.array([sig_row[0], sig_row[1], sig_row[2], sig_row[3], sig_row[4], sig_row[16], sig_row[11]], dtype=float)
            ndim = 7

        # Shared state for tracking best no-max result across all layers
        global_best_no_max = {"snr": -1e60, "theta": None}
        obj_both_max = objective_factory(pa_template=pa_template, target_func=args.target_func, ctx=ctx, maximize_phase=True, maximize_time=True, global_best_no_max=global_best_no_max)
        obj_phase_max = objective_factory(pa_template=pa_template, target_func=args.target_func, ctx=ctx, maximize_phase=True, maximize_time=False, global_best_no_max=global_best_no_max)
        obj_no_max = objective_factory(pa_template=pa_template, target_func=args.target_func, ctx=ctx, maximize_phase=False, maximize_time=False, global_best_no_max=global_best_no_max)

        # --- LAYER 1: Physical Discovery ---
        print("\n[LAYER 1] Finding physical mountain (Multi-Constraint Discovery)...")
        l1_phys_results = []
        # A. Both-Max (Wide Basin)
        discovery_starts_both = [theta_true, theta0, (theta_true + theta0)/2]
        for start_p in discovery_starts_both:
            res = nelder_mead_optimize(start_p, lambda x: -float(obj_both_max(x)), maxiter=250)
            l1_phys_results.append(res.x)
        
        # B. Phase-Max (Immediate t=0 alignment for local truth)
        res_p0 = nelder_mead_optimize(theta_true, lambda x: -float(obj_phase_max(x)), maxiter=300)
        l1_phys_results.append(res_p0.x)

        rng = np.random.default_rng(42)
        is_near = all_optimized_mismatch[idx] < 0.15
        n_l1_random = 6 if all_optimized_mismatch[idx] > 0.4 else 4
        for _ in range(n_l1_random):
            # If near-miss, jitter theta0 (current best), otherwise jitter theta_true
            p_ref = theta0 if (is_near and rng.random() > 0.3) else theta_true
            jitter_scale = 1e-4 if is_near else 1e-3
            p = _clip_physical_params(p_ref * (1.0 + rng.uniform(-jitter_scale, jitter_scale, size=ndim)), pa_template)
            # Mixed random starts
            obj = obj_both_max if rng.random() > 0.5 else obj_phase_max
            res = nelder_mead_optimize(p, lambda x: -float(obj(x)), maxiter=200)
            l1_phys_results.append(res.x)

        # --- LAYER 1.1: Transfer to t=0 ---
        print("\n[LAYER 1.1] Transferring physical mountains to t=0 (Phase-Max Polish)...")
        l1_t0_results = []
        for p in l1_phys_results:
            # Shift towards t=0 peak while allowing phase to be flexible
            res = nelder_mead_optimize(p, lambda x: -float(obj_phase_max(x)), maxiter=200)
            l1_t0_results.append(res.x)
        
        # --- LAYER 1.2: Phase Alignment (No-Max) ---
        print("\n[LAYER 1.2] Aligning physical results to t=0 (Phase-only optimization)...")
        l1_aligned_results = []
        for p_phys in l1_t0_results:
            # 1D Phase Search
            phases = np.linspace(0, 2*np.pi, 100)
            best_phi = 0.0
            best_s = -1e60
            for phi in phases:
                p_test = p_phys.copy()
                p_test[-1] = phi
                s = float(obj_no_max(p_test))
                if s > best_s:
                    best_s = s
                    best_phi = phi
            
            # Local polish on phase only
            def phi_obj(phi_val):
                p = p_phys.copy()
                p[-1] = phi_val[0] % (2*np.pi)
                return -float(obj_no_max(p))
            
            res_phi = minimize(phi_obj, x0=[best_phi], bounds=[(0, 2*np.pi)], method='L-BFGS-B')
            p_aligned = p_phys.copy()
            p_aligned[-1] = res_phi.x[0]
            l1_aligned_results.append(p_aligned)
            
            # Full-parameter No-Max polish for ALL seeds to pull them into the peak
            res_quick = nelder_mead_optimize(p_aligned, lambda x: -float(obj_no_max(x)), maxiter=150)
            l1_aligned_results.append(res_quick.x)

        # --- LAYER 2: PARIS (Phase-Maximized Exploration) ---
        print(f"\n[LAYER 2] PARIS Global Search (Phase-Max, ndim={ndim})...")
        l1_aligned_results = np.array(l1_aligned_results)
        # Filter for unique results
        unique_l1 = np.unique(np.round(l1_aligned_results, decimals=8), axis=0)
        mean_l2 = np.mean(unique_l1, axis=0)
        
        if len(unique_l1) > 1:
            cov_l2 = np.cov(unique_l1, rowvar=False) + np.diag((theta_true * 1e-8)**2)
        else:
            cov_l2 = np.diag((theta_true * 2e-4)**2)
            
        evals, evecs = np.linalg.eigh(cov_l2)
        evals = np.maximum(evals, 1e-32)
        
        # Dynamic magnification: 8-sigma for hard, 4-sigma for near-miss
        mag = 8.0 if all_optimized_mismatch[idx] > 0.15 else 4.0
        b_l2 = mag * np.sqrt(evals)

        savepath_l2 = os.path.join(sampler_base, f"idx_{idx}_l2")
        
        def paris_save_callback(sampler_obj, iter_i):
            """Periodic check to see if the best intrinsic parameters from PARIS 
               yield a better NO-MAX mismatch after phase locking."""
            if iter_i % 250 == 0:
                sampler_obj.save_state()
                # Find current best point in PARIS history (by Phase-Max SNR)
                element_num = sampler_obj.element_num_list[0]
                if element_num == 0: return
                vals = sampler_obj.searched_log_densities_list[0][:element_num]
                best_idx = np.argmax(vals)
                best_u = sampler_obj.searched_points_list[0][best_idx]
                best_theta = paris_prior_transform(best_u)
                
                # Perform a quick 50-point phase scan to lock phase for evaluation
                phases = np.linspace(0, 2*np.pi, 50)
                best_phi = 0.0
                best_s = -1e60
                for phi in phases:
                    p_test = best_theta.copy()
                    p_test[-1] = phi
                    s = float(obj_no_max(p_test))
                    if s > best_s:
                        best_s = s
                        best_phi = phi
                
                mm_curr = 1.0 - (best_s / utils.TARGET_SNR)
                if mm_curr < all_optimized_mismatch[idx]:
                    print(f"[SAVE-CHECK] Saving progress at iter {iter_i}. Best No-Max SNR: {best_s:.4f}, Mismatch: {mm_curr:.6f}")
                    all_optimized_mismatch[idx] = mm_curr
                    new_row = sig_row.copy()
                    bt = best_theta.copy(); bt[-1] = best_phi
                    if pa_template == '0PA':
                        new_row[:5] = bt[:5]; new_row[11] = bt[5]
                    else:
                        new_row[:5] = bt[:5]; new_row[16] = bt[5]; new_row[11] = bt[6]
                    all_optimized_params[idx] = new_row
                    np.save(params_filename, all_optimized_params)
                    np.save(mismatch_filename, all_optimized_mismatch)

        import parismc
        # Robust adaptation for smoother Phase-Max surface: alpha=1200, gamma=30
        config_l2 = parismc.SamplerConfig(
            merge_type='single', alpha=1200, trail_size=1000, 
            boundary_limiting=True, use_beta=True, gamma=30, use_pool=False
        )
        
        global _PARIS_REF_CENTER, _PARIS_SPREAD_SCALE, _PARIS_OBJECTIVE, _PARIS_TARGET_KIND, _PARIS_EARLY_STOP_HIT
        global _PARIS_AFFINE_CENTER, _PARIS_AFFINE_Q, _PARIS_AFFINE_B, _PARIS_DIM
        _PARIS_REF_CENTER = mean_l2.copy(); _PARIS_SPREAD_SCALE = 1.0; _PARIS_OBJECTIVE = obj_phase_max
        _PARIS_TARGET_KIND = 'optimal_snr_phase_max'; _PARIS_EARLY_STOP_HIT = False; _PARIS_DIM = ndim
        _PARIS_AFFINE_CENTER = mean_l2.copy(); _PARIS_AFFINE_Q = evecs; _PARIS_AFFINE_B = b_l2

        n_seed_l2 = 32 # Increased for better exploration
        sigma_l2 = 1e-2
        init_cov_l2 = [sigma_l2**2 * np.eye(ndim) for _ in range(n_seed_l2)]
        
        sampler_l2 = parismc.Sampler(
            ndim=ndim, n_seed=n_seed_l2, log_density_func=paris_log_density,
            init_cov_list=init_cov_l2, prior_transform=paris_prior_transform, config=config_l2
        )
        
        u_seeds = paris_inverse_prior_transform(unique_l1)
        u_seeds = np.clip(u_seeds, 1e-6, 1.0 - 1e-6)
        val_seeds = paris_log_density(paris_prior_transform(u_seeds))
        
        from smt.sampling_methods import LHS
        xlimits = np.column_stack([np.zeros(ndim), np.ones(ndim)])
        sampling = LHS(xlimits=xlimits)
        lhs_points = sampling(1000)
        lhs_vals = paris_log_density(paris_prior_transform(lhs_points))
        
        ext_pts = np.vstack([u_seeds, lhs_points])
        ext_vals = np.concatenate([val_seeds, lhs_vals])

        sampler_l2.run_sampling(
            num_iterations=2000, savepath=savepath_l2, print_iter=100,
            external_lhs_points=ext_pts, external_lhs_log_densities=ext_vals,
            callback=paris_save_callback
        )
        
        # --- LAYER 3: Final No-Max Swarm Polish ---
        print("\n[LAYER 3] Final No-Max Swarm Polish...")
        
        # 1. Collect candidate pool from PARIS history
        candidates = []
        element_num = sampler_l2.element_num_list[0]
        pts = sampler_l2.searched_points_list[0][:element_num]
        vals = sampler_l2.searched_log_densities_list[0][:element_num]
        
        # Sort by SNR and filter for unique physical parameters (excluding phase)
        top_indices = np.argsort(vals)[-100:][::-1] 
        added_pts = []
        for tidx in top_indices:
            p_phys = paris_prior_transform(pts[tidx])
            # Check uniqueness based on intrinsic parameters only
            intrinsic = p_phys[:-1]
            is_unique = True
            for ap in added_pts:
                if np.allclose(intrinsic, ap[:-1], rtol=1e-5):
                    is_unique = False; break
            if is_unique:
                candidates.append(p_phys)
                added_pts.append(p_phys)
            if len(candidates) >= 8: break
            
        # Also include the global best track if it's unique
        if global_best_no_max["theta"] is not None:
            g_best = global_best_no_max["theta"]
            is_unique = True
            for ap in added_pts:
                if np.allclose(g_best, ap, rtol=1e-4):
                    is_unique = False; break
            if is_unique:
                candidates.append(g_best)
        
        print(f"[SWARM] Polishing {len(candidates)} candidates with phase-locking...")
        swarm_results = []
        for i, c_start in enumerate(candidates):
            # Dense 1D Phase-lock scan (200 pts)
            phases = np.linspace(0, 2*np.pi, 200)
            best_phi = 0.0
            best_s = -1e60
            for phi in phases:
                p_test = c_start.copy()
                p_test[-1] = phi
                s = float(obj_no_max(p_test))
                if s > best_s:
                    best_s = s
                    best_phi = phi
            
            c_aligned = c_start.copy()
            c_aligned[-1] = best_phi
            
            # Short local polish (300 iters)
            res_short = nelder_mead_optimize(c_aligned, lambda x: -float(obj_no_max(x)), maxiter=300)
            swarm_results.append((float(obj_no_max(res_short.x)), res_short.x))
            print(f"  Candidate {i+1}/{len(candidates)}: Local Best SNR {swarm_results[-1][0]:.4f}")
            
        # Pick best from swarm for final deep polish
        swarm_results.sort(key=lambda x: x[0], reverse=True)
        best_theta_swarm = swarm_results[0][1]
        
        print(f"[FINAL] Deep Polish from swarm champion (SNR: {swarm_results[0][0]:.4f})...")
        res_final = nelder_mead_optimize(best_theta_swarm, lambda x: -float(obj_no_max(x)), maxiter=1000, xatol=1e-10, fatol=1e-11)
        best_theta = res_final.x
        
        final_snr = float(obj_no_max(best_theta))
        new_mismatch = 1.0 - (final_snr / utils.TARGET_SNR)
        print(f"[RESULT] idx={idx} Final Mismatch: {new_mismatch:.6e}")

        if new_mismatch < all_optimized_mismatch[idx]:
            all_optimized_mismatch[idx] = new_mismatch
            new_row = sig_row.copy()
            if pa_template == '0PA':
                new_row[:5] = best_theta[:5]; new_row[11] = best_theta[5]
            else:
                new_row[:5] = best_theta[:5]; new_row[16] = best_theta[5]; new_row[11] = best_theta[6]
            all_optimized_params[idx] = new_row
            np.save(params_filename, all_optimized_params)
            np.save(mismatch_filename, all_optimized_mismatch)
        
        print(f"[INFO] Finished Index {idx}. New Global Mismatch: {all_optimized_mismatch[idx]:.6f}")
        break

    print("\n[FINISH] Optimization session complete.")
    end_time_all = time.time()
    print(f"\n[TIME] Total execution time: {end_time_all - start_time_all:.2f} seconds")



if __name__ == '__main__':
    main()
