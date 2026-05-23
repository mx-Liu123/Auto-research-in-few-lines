import os
import json
import time
import argparse
import threading
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
# PARIS global context
# -----------------------------
_PARIS_REF_CENTER = None          # type: Optional[np.ndarray]
_PARIS_SPREAD_SCALE = None        # type: Optional[float]
_PARIS_OBJECTIVE = None           # type: Optional[callable]
_PARIS_TARGET_KIND = None         # type: Optional[str]
_PARIS_EARLY_STOP_HIT = False
_GLOBAL_BEST_LOCK = threading.Lock()

_PARIS_AFFINE_CENTER = None       # type: Optional[np.ndarray]
_PARIS_AFFINE_Q = None            # type: Optional[np.ndarray]
_PARIS_AFFINE_B = None            # type: Optional[np.ndarray]
_PARIS_DIM = None                 # type: Optional[int]
_PARIS_FULL_DIM = None            # type: Optional[int]


def _clip_physical_params(theta: np.ndarray, pa_template: str) -> np.ndarray:
    """Clip mapped physical parameters to minimal physical ranges.
    Indices: 0:m1, 1:m2, 2:a, 3:p0, 4:e0, ... [last:Phi_phi0]
    """
    x = np.asarray(theta, dtype=float).copy()
    if x.ndim == 1:
        if x.shape[0] >= 1: x[0] = np.clip(x[0], 1e4, 1e7)
        if x.shape[0] >= 2: x[1] = np.clip(x[1], 1e-3, 1e4)
        if x.shape[0] >= 3: x[2] = np.clip(x[2], -0.999, 0.999)
        if x.shape[0] >= 5: x[4] = np.clip(x[4], 1e-4, 0.9)
        
        # Retrograde-aware separatrix buffer: p_sep increases as spin becomes more negative
        if x.shape[0] >= 5:
            spin = x[2] if x.shape[0] >= 3 else 0.0
            e0 = x[4]
            p_sep = 6.0 + 2.0 * e0 + max(0.0, -3.8 * spin)
            if x.shape[0] >= 4:
                x[3] = np.clip(x[3], p_sep + 0.1, 100.0)
        
        if pa_template == '1PA':
            if x.shape[0] >= 6: x[5] = np.clip(x[5], -0.99, 0.99)
            if x.shape[0] >= 7: x[6] = x[6] % (2 * np.pi)
            if x.shape[0] >= 8: x[7] = x[7] % (2 * np.pi)
            if x.shape[0] >= 9: x[8] = x[8] % (2 * np.pi)
        else: # 0PA
            if x.shape[0] >= 6: x[5] = x[5] % (2 * np.pi)
            if x.shape[0] >= 7: x[6] = x[6] % (2 * np.pi)
            if x.shape[0] >= 8: x[7] = x[7] % (2 * np.pi)
        return x
    else:
        x[:, 0] = np.clip(x[:, 0], 1e4, 1e7)
        x[:, 1] = np.clip(x[:, 1], 1e-3, 1e4)
        if x.shape[1] >= 3: x[:, 2] = np.clip(x[:, 2], -0.999, 0.999)
        if x.shape[1] >= 5: x[:, 4] = np.clip(x[:, 4], 1e-4, 0.9)
        
        for i in range(x.shape[0]):
            spin = x[i, 2] if x.shape[1] >= 3 else 0.0
            e0 = x[i, 4] if x.shape[1] >= 5 else 0.0
            p_sep = 6.0 + 2.0 * e0 + max(0.0, -3.8 * spin)
            if x.shape[1] >= 4:
                x[i, 3] = np.clip(x[i, 3], p_sep + 0.1, 100.0)
                
        if pa_template == '1PA':
            if x.shape[1] >= 6: x[:, 5] = np.clip(x[:, 5], -0.99, 0.99)
            if x.shape[1] >= 7: x[:, 6] = x[:, 6] % (2 * np.pi)
            if x.shape[1] >= 8: x[:, 7] = x[:, 7] % (2 * np.pi)
            if x.shape[1] >= 9: x[:, 8] = x[:, 8] % (2 * np.pi)
        else:
            if x.shape[1] >= 6: x[:, 5] = x[:, 5] % (2 * np.pi)
            if x.shape[1] >= 7: x[:, 6] = x[:, 6] % (2 * np.pi)
            if x.shape[1] >= 8: x[:, 7] = x[:, 7] % (2 * np.pi)
        return x


def paris_prior_transform(u):
    u = np.asarray(u, dtype=float)
    if _PARIS_AFFINE_CENTER is not None and _PARIS_AFFINE_Q is not None and _PARIS_AFFINE_B is not None:
        center = _PARIS_AFFINE_CENTER; Q = _PARIS_AFFINE_Q; b = _PARIS_AFFINE_B; dim = _PARIS_DIM
        def map_one(u1):
            t = 2.0 * np.asarray(u1)[:dim] - 1.0
            return center + Q @ (b * t)
        if u.ndim == 1:
            theta = map_one(u)
            pa_tmpl = '1PA' if dim >= 8 else '0PA'
            return _clip_physical_params(theta, pa_tmpl)
        else:
            out = np.zeros((u.shape[0], dim), dtype=float)
            for i in range(u.shape[0]): out[i] = map_one(u[i])
            pa_tmpl = '1PA' if dim >= 8 else '0PA'
            return _clip_physical_params(out, pa_tmpl)
    ref = _PARIS_REF_CENTER; s = _PARIS_SPREAD_SCALE
    return ref * (1 - s + u * 2 * s)


def paris_inverse_prior_transform(params):
    theta = np.asarray(params, dtype=float)
    if _PARIS_AFFINE_CENTER is not None and _PARIS_AFFINE_Q is not None and _PARIS_AFFINE_B is not None:
        center = _PARIS_AFFINE_CENTER; Q = _PARIS_AFFINE_Q; b = _PARIS_AFFINE_B; inv_b = 1.0 / b
        def inv_one(th):
            d = np.asarray(th) - center
            t = (Q.T @ d) * inv_b
            return 0.5 * (t + 1.0)
        if theta.ndim == 1: return inv_one(theta)
        out = np.zeros_like(theta, dtype=float)
        for i in range(theta.shape[0]): out[i] = inv_one(theta[i])
        return out
    ref = _PARIS_REF_CENTER; s = _PARIS_SPREAD_SCALE
    return (theta / ref - (1 - s)) / (2 * s)


def paris_log_density(params):
    """Log-density function for PARIS."""
    global _PARIS_EARLY_STOP_HIT
    params = np.asarray(params)
    def eval_one(x):
        global _PARIS_EARLY_STOP_HIT
        if _PARIS_EARLY_STOP_HIT: return float('-inf')
        try:
            if _PARIS_DIM is not None and _PARIS_FULL_DIM is not None and _PARIS_DIM < _PARIS_FULL_DIM:
                x_full = np.zeros(_PARIS_FULL_DIM)
                x_full[:_PARIS_DIM] = x
                x_full[-1] = 0.0
                val = float(_PARIS_OBJECTIVE(x_full))
            else:
                val = float(_PARIS_OBJECTIVE(x))
        except Exception: return float('-inf')
        if _PARIS_TARGET_KIND == 'optimal_snr':
            if val >= 19.95: _PARIS_EARLY_STOP_HIT = True
        return val
    if params.ndim == 1: return eval_one(params)
    out = np.zeros(params.shape[0], dtype=float)
    for i in range(params.shape[0]): out[i] = eval_one(params[i])
    return out


def build_waveform_response(T: float, dt: float, use_gpu: bool = False) -> ResponseWrapper:
    """Create a LISA ResponseWrapper consistent with the 2PA signal model."""
    from few.waveform import GenerateEMRIWaveform
    from few.waveform.waveform import SuperKludgeWaveform
    sum_kwargs = dict(pad_output=True, odd_len=True)
    waveform_model = GenerateEMRIWaveform(SuperKludgeWaveform, sum_kwargs=sum_kwargs, return_list=False)
    t0 = 10000.0; tdi_gen = "2nd generation"; order = 20; index_lambda = 8; index_beta = 7
    response = ResponseWrapper(
        waveform_gen=waveform_model, Tobs=T, t0=t0, dt=dt, index_lambda=index_lambda, index_beta=index_beta,
        flip_hx=True, use_gpu=use_gpu, is_ecliptic_latitude=False, remove_garbage="zero",
        orbits=EqualArmlengthOrbits(use_gpu=use_gpu), order=order, tdi=tdi_gen, tdi_chan="AET",
    )
    return response


def prepare_2pa_fiducial(signal_row: np.ndarray, use_gpu: bool = False):
    """Generate 2PA fiducial signal and pre-calculate PSD/FFT context."""
    (m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, dt, T, chi2) = signal_row
    waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu)
    wave_params = [m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, chi2, True, False, True]
    emri_kwargs = {"T": T, "dt": dt, '1PA': True, 'evolve_primary': False, '2PA': True}
    waveform_2pa = waveform_response(*wave_params, **emri_kwargs)
    channels = [A1TDISens, E1TDISens, T1TDISens]
    noise_kwargs = [{"sens_fn": ch} for ch in channels]
    PSD_funcs = generate_PSD(waveform=waveform_2pa, dt=dt, noise_PSD=get_sensitivity, channels=channels, noise_kwargs=noise_kwargs, use_gpu=use_gpu)
    snr = float(np.sqrt(inner_product(waveform_2pa, waveform_2pa, PSD_funcs, dt, use_gpu=use_gpu)))
    waveform_2pa_fft = utils.compute_fft_with_windowing(waveform_2pa, dt, use_gpu=use_gpu)
    N_fiducial = len(waveform_2pa[0])
    return {
        'm1': m1, 'm2': m2, 'a': a, 'p0': p0, 'e0': e0, 'Y0': Y0, 'dist': dist, 'qS': qS, 'phiS': phiS, 'qK': qK, 'phiK': phiK,
        'Phi_phi0': Phi_phi0, 'Phi_theta0': Phi_theta0, 'Phi_r0': Phi_r0, 'dt': dt, 'T': T, 'chi2': chi2,
        'waveform_response': waveform_response, 'PSD_funcs': PSD_funcs, 'waveform_2pa_fft': waveform_2pa_fft, 'N_fiducial': N_fiducial, 'snr': snr,
    }


def get_complex_overlap(h1_fft, h2_fft, PSD, dt, N, use_gpu=False):
    if use_gpu: import cupy as xp
    else: import numpy as xp
    PSD = xp.atleast_2d(xp.asarray(PSD))
    df = (N * dt) ** -1
    integrand = 4 * ((h1_fft.conj() * h2_fft) / PSD)
    complex_inner_prod = (integrand.sum(axis=0) * df).sum()
    if use_gpu: return complex_inner_prod.get()
    return complex_inner_prod


def real_overlap_lock(p_phys, ctx, pa_template, maxiter=100, maximize_time=False, include_mass_spin=False):
    """Surgically adjust parameters to maximize overlap. 
    If maximize_time=True, finds the best tc and then adjusts physical params.
    If include_mass_spin=True, also optimizes m1, m2, a, p0, e0, and phase offsets.
    """
    from scipy.optimize import minimize
    import cupy as xp
    psd_arr = xp.asarray(ctx['PSD_funcs']); inv_psd = 4.0 / psd_arr
    df = 1.0 / (ctx['N_fiducial'] * ctx['dt']); h2_fft = xp.asarray(ctx['waveform_2pa_fft'])
    h2_dot_h2 = ctx['snr']**2
    
    # Selection of indices to optimize
    if include_mass_spin:
        if pa_template == '0PA':
            indices_to_opt = [0, 1, 2, 3, 4, 5, 6] # m1, m2, a, p0, e0, phi_theta, phi_r
        else:
            indices_to_opt = [0, 1, 2, 3, 4, 5, 6, 7] # m1, m2, a, p0, e0, chi2, phi_theta, phi_r
    else:
        indices_to_opt = [3, 4] # Always p0, e0
        
    def local_f(x):
        p = p_phys.copy()
        for i, idx_in_p in enumerate(indices_to_opt):
            p[idx_in_p] = x[i]
        p = _clip_physical_params(p, pa_template)
        try:
            # Separability check to avoid wasted time on unphysical signals
            spin = p[2]; e0 = p[4]; p0 = p[3]
            if p0 < 6.0 + 2.0 * e0 + max(0.0, -3.9 * spin): return 0.0
            
            waveform_tmpl = ctx['waveform_response'](*build_wave_params(p, pa_template, ctx), **build_emri_kwargs(p, pa_template, ctx))
            h1_data = match_len(waveform_tmpl, ctx)
            h1_fft = utils.compute_fft_with_windowing(h1_data, ctx['dt'], use_gpu=True)
            h1_dot_h1 = float((inv_psd * xp.abs(h1_fft)**2).sum() * df)
            denom_sqrt = np.sqrt(max(h1_dot_h1 * h2_dot_h2, 1e-32))
            
            if maximize_time:
                integrand_sum = (inv_psd * (h1_fft.conj() * h2_fft)).sum(axis=0)
                padded = xp.zeros(ctx['N_fiducial'], dtype=xp.complex128)
                padded[1 : len(integrand_sum)+1] = integrand_sum
                z_t = xp.fft.ifft(padded) * (1.0 / ctx['dt'])
                snr_val = float(xp.max(xp.abs(z_t))) / denom_sqrt * utils.TARGET_SNR
            else:
                z = (inv_psd * (h1_fft.conj() * h2_fft)).sum() * df
                snr_val = float(xp.abs(z)) / denom_sqrt * utils.TARGET_SNR
            return -float(snr_val)
        except: return 0.0
    
    x0 = [p_phys[i] for i in indices_to_opt]
    res = minimize(local_f, x0, method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-4})
    
    p_final = p_phys.copy()
    for i, idx_in_p in enumerate(indices_to_opt):
        p_final[idx_in_p] = res.x[i]
    p_final = _clip_physical_params(p_final, pa_template)
    
    # Final phase lock at t=0
    waveform_tmpl = ctx['waveform_response'](*build_wave_params(p_final, pa_template, ctx), **build_emri_kwargs(p_final, pa_template, ctx))
    h1_fft = utils.compute_fft_with_windowing(match_len(waveform_tmpl, ctx), ctx['dt'], use_gpu=True)
    z = (inv_psd * (h1_fft.conj() * h2_fft)).sum() * df
    dphi = -float(xp.angle(z))
    p_final[-1] = (p_final[-1] + dphi) % (2*np.pi)
    
    h1_dot_h1 = float((inv_psd * xp.abs(h1_fft)**2).sum() * df)
    denom_sqrt = np.sqrt(max(h1_dot_h1 * h2_dot_h2, 1e-32))
    
    if maximize_time:
        integrand_sum = (inv_psd * (h1_fft.conj() * h2_fft)).sum(axis=0)
        padded = xp.zeros(ctx['N_fiducial'], dtype=xp.complex128)
        padded[1 : len(integrand_sum)+1] = integrand_sum
        z_t = xp.fft.ifft(padded) * (1.0 / ctx['dt'])
        final_snr = float(xp.max(xp.abs(z_t))) / denom_sqrt * utils.TARGET_SNR
    else:
        final_snr = float(xp.abs(z)) / denom_sqrt * utils.TARGET_SNR

    return p_final, final_snr

def phase_time_lock_2D(p_phys, ctx, pa_template, maxiter=50):
    """Surgically adjust p0, e0 and analytical phase Phi_phi0 to lock onto peak overlap."""
    from scipy.optimize import minimize
    def local_f(x):
        p = p_phys.copy(); p[3] = x[0]; p[4] = x[1]
        p = _clip_physical_params(p, pa_template)
        try:
            waveform_tmpl = ctx['waveform_response'](*build_wave_params(p, pa_template, ctx), **build_emri_kwargs(p, pa_template, ctx))
            h1_fft = utils.compute_fft_with_windowing(match_len(waveform_tmpl, ctx), ctx['dt'], use_gpu=True)
            z = get_complex_overlap(ctx['waveform_2pa_fft'], h1_fft, ctx['PSD_funcs'], ctx['dt'], ctx['N_fiducial'], use_gpu=True)
            h1_dot_h1 = utils.inner_product_from_fft(h1_fft, h1_fft, ctx['PSD_funcs'], ctx['dt'], ctx['N_fiducial'], use_gpu=True, maximize_phase=False, maximize_time=False)
            h2_dot_h2 = ctx['snr']**2; snr_val = abs(z) / np.sqrt(max(h1_dot_h1 * h2_dot_h2, 1e-32)) * utils.TARGET_SNR
            return -float(snr_val)
        except: return 0.0
    x0 = [p_phys[3], p_phys[4]]; res = minimize(local_f, x0, method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-4})
    p_final = p_phys.copy(); p_final[3] = res.x[0]; p_final[4] = res.x[1]; p_final = _clip_physical_params(p_final, pa_template)
    waveform_tmpl = ctx['waveform_response'](*build_wave_params(p_final, pa_template, ctx), **build_emri_kwargs(p_final, pa_template, ctx))
    h1_fft = utils.compute_fft_with_windowing(match_len(waveform_tmpl, ctx), ctx['dt'], use_gpu=True)
    z = get_complex_overlap(ctx['waveform_2pa_fft'], h1_fft, ctx['PSD_funcs'], ctx['dt'], ctx['N_fiducial'], use_gpu=True)
    h1_dot_h1 = utils.inner_product_from_fft(h1_fft, h1_fft, ctx['PSD_funcs'], ctx['dt'], ctx['N_fiducial'], use_gpu=True, maximize_phase=False, maximize_time=False)
    h2_dot_h2 = ctx['snr']**2; snr_final = abs(z) / np.sqrt(max(h1_dot_h1 * h2_dot_h2, 1e-32)) * utils.TARGET_SNR
    dphi = -np.angle(z); p_final[-1] = (p_final[-1] + dphi) % (2*np.pi)
    return p_final, snr_final


def build_wave_params(theta, pa_template, ctx):
    theta = _clip_physical_params(theta, pa_template)
    if pa_template == '0PA':
        m1, m2, a, p0, e0 = theta[:5]
        phi_theta0, phi_r0 = (theta[5], theta[6]) if len(theta) >= 8 else (ctx['Phi_theta0'], ctx['Phi_r0'])
        chi2 = ctx['chi2']; evolve_1PA = False
    else:
        m1, m2, a, p0, e0, chi2 = theta[:6]
        phi_theta0, phi_r0 = (theta[6], theta[7]) if len(theta) >= 9 else (ctx['Phi_theta0'], ctx['Phi_r0'])
        evolve_1PA = True
    phi_phi0 = theta[-1]
    return [m1, m2, a, p0, e0, ctx['Y0'], ctx['dist'], ctx['qS'], ctx['phiS'], ctx['qK'], ctx['phiK'], phi_phi0, phi_theta0, phi_r0, chi2, evolve_1PA, False, False]


def build_emri_kwargs(theta, pa_template, ctx):
    return {"T": ctx['T'], "dt": ctx['dt'], '1PA': (pa_template == '1PA'), 'evolve_primary': False, '2PA': False}


def match_len(waveform_tmpl, ctx):
    if len(waveform_tmpl[0]) < ctx['N_fiducial']:
        import cupy as cp; pad_len = ctx['N_fiducial'] - len(waveform_tmpl[0])
        return [cp.pad(cp.asarray(ch), (0, pad_len)) for ch in waveform_tmpl]
    elif len(waveform_tmpl[0]) > ctx['N_fiducial']: return [ch[:ctx['N_fiducial']] for ch in waveform_tmpl]
    return waveform_tmpl


def objective_factory(pa_template: str, target_func: str, ctx: dict, maximize_phase: bool = False, maximize_time: bool = False, use_gpu_for_snr: bool = True, global_best_no_max: dict = None, anneal_params: tuple = None):
    """High-performance objective factory using pre-calculated GPU constants and thread-safe updates."""
    if global_best_no_max is None: global_best_no_max = {"snr": -1e60, "theta": None}
    import cupy as xp; psd_arr = xp.asarray(ctx['PSD_funcs']); inv_psd = 4.0 / psd_arr
    df = 1.0 / (ctx['N_fiducial'] * ctx['dt']); h2_fft = xp.asarray(ctx['waveform_2pa_fft'])
    h2_dot_h2 = ctx['snr']**2; best_score_internal = [-1e60]; iter_count = [0]
    
    def score_optimal_snr(theta: np.ndarray) -> float:
        iter_count[0] += 1; theta = _clip_physical_params(theta, pa_template)
        try:
            # Hard safety check for separatrix before calling waveform generator
            spin = theta[2]; e0 = theta[4]; p0 = theta[3]
            p_sep_hard = 6.0 + 2.0 * e0 + max(0.0, -3.9 * spin)
            if p0 < p_sep_hard: return -1e20
            
            waveform_tmpl = ctx['waveform_response'](*build_wave_params(theta, pa_template, ctx), **build_emri_kwargs(theta, pa_template, ctx))
            h1_data = match_len(waveform_tmpl, ctx)
            if np.any(np.isnan(h1_data[0])): return -1e20
            
            h1_fft = utils.compute_fft_with_windowing(h1_data, ctx['dt'], use_gpu=True)
            h1_dot_h1 = float((inv_psd * xp.abs(h1_fft)**2).sum() * df)
            denom_sqrt = np.sqrt(max(h1_dot_h1 * h2_dot_h2, 1e-32))
            
            if maximize_time:
                integrand_sum = (inv_psd * (h1_fft.conj() * h2_fft)).sum(axis=0)
                padded = xp.zeros(ctx['N_fiducial'], dtype=xp.complex128); padded[1 : len(integrand_sum)+1] = integrand_sum
                z_t = xp.fft.ifft(padded) * (1.0 / ctx['dt']); z_abs = xp.abs(z_t); idx_opt = int(xp.argmax(z_abs))
                snr_maximized = float(z_abs[idx_opt]) / denom_sqrt * utils.TARGET_SNR
                z_zero = z_t[0]; snr_t0 = float(xp.abs(z_zero)) / denom_sqrt * utils.TARGET_SNR

                if snr_t0 > global_best_no_max["snr"]:
                    with _GLOBAL_BEST_LOCK:
                        if snr_t0 > global_best_no_max["snr"]:
                            dphi_z = -float(xp.angle(z_zero)); theta_l = theta.copy(); theta_l[-1] = (theta[-1] + dphi_z) % (2*np.pi)
                            global_best_no_max["snr"] = snr_t0; global_best_no_max["theta"] = theta_l
                            
                # Calculate time shift for penalty
                N = ctx['N_fiducial']
                abs_idx_shift = min(idx_opt, N - idx_opt)
                abs_t_shift = abs_idx_shift * ctx['dt']
            else:
                z = (inv_psd * (h1_fft.conj() * h2_fft)).sum() * df; snr_t0 = float(xp.abs(z)) / denom_sqrt * utils.TARGET_SNR
                snr_maximized = snr_t0 if maximize_phase else float(z.real) / denom_sqrt * utils.TARGET_SNR
                abs_t_shift = 0.0
                if snr_t0 > global_best_no_max["snr"]:
                    with _GLOBAL_BEST_LOCK:
                        if snr_t0 > global_best_no_max["snr"]:
                            dphi = -float(xp.angle(z)); theta_l = theta.copy(); theta_l[-1] = (theta[-1] + dphi) % (2*np.pi)
                            global_best_no_max["snr"] = snr_t0; global_best_no_max["theta"] = theta_l

            if anneal_params is not None:
                a_s, a_e, penalty_weight = anneal_params
                if iter_count[0] > a_s:
                    w = min(penalty_weight, (iter_count[0] - a_s) / (a_e - a_s) * penalty_weight) if a_e > a_s else penalty_weight
                    # Apply a time-shift penalty instead of blending snr_t0
                    snr_val = snr_maximized - w * (abs_t_shift / 3600.0)**2 # Penalty based on hours squared
                else: 
                    snr_val = snr_maximized
            else: 
                snr_val = snr_maximized
                
            if np.isnan(snr_val) or np.isinf(snr_val): return -1e20
        except Exception: return -1e20
        if snr_val > best_score_internal[0]: best_score_internal[0] = snr_val
        if iter_count[0] % 200 == 0: 
            t_shift_str = f" | T-SHIFT: {abs_t_shift:.1f}s" if maximize_time else ""
            print(f"[ITER: {iter_count[0]} | CURR SNR: {snr_val:.4f} | BEST: {best_score_internal[0]:.4f} | GLOBAL NO-MAX: {global_best_no_max['snr']:.6f}{t_shift_str}]", flush=True)
        return float(snr_val)
    return score_optimal_snr


def nelder_mead_optimize(theta0: np.ndarray, objective, maxiter: int = 500, xatol: float = 1e-8, fatol: float = 1e-10):
    return minimize(objective, theta0, method='Nelder-Mead', options={'maxiter': maxiter, 'maxfev': maxiter + 400, 'xatol': xatol, 'fatol': fatol})


def main():
    start_time_all = time.time(); parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', default='PARIS')
    parser.add_argument('--grid-index', type=int, default=None)
    parser.add_argument('--startingpoints', type=str, default=None, help='Optional .npy path for starting points')
    args = parser.parse_args()
    
    scenario = utils.scenario; pa_template = utils.pa_template
    signal_param_array = utils.load_signal_param_array()

    params_filename = f"optimized_params_{scenario}_{pa_template}.npy"; mismatch_filename = f"optimized_mismatch_{scenario}_{pa_template}.npy"
    
    if args.startingpoints:
        print(f"Loading search starting points from explicit path: {args.startingpoints}")
        all_optimized_params = np.load(args.startingpoints)
        all_optimized_mismatch = np.ones(all_optimized_params.shape[0])
    elif os.path.exists(params_filename) and os.path.exists(mismatch_filename):
        all_optimized_params = np.load(params_filename); all_optimized_mismatch = np.load(mismatch_filename)
    else:
        all_optimized_params = signal_param_array.copy(); all_optimized_mismatch = np.ones(signal_param_array.shape[0])
    
    if args.grid_index is not None: indices = [args.grid_index]
    else:
        target_idx = -1; max_miss = -1.0
        # Target anything > 0.01 to ensure high quality across the board
        for i in range(signal_param_array.shape[0]):
            if all_optimized_mismatch[i] > 0.01 and all_optimized_mismatch[i] > max_miss: max_miss = all_optimized_mismatch[i]; target_idx = i
        if target_idx == -1: return
        indices = [target_idx]
        
    sampler_base = "paris_samplers"
    import parismc 
    
    for idx in indices:
        sig_row = signal_param_array[idx]; ctx = prepare_2pa_fiducial(sig_row, use_gpu=True); current_best_row = all_optimized_params[idx]
        if pa_template == '0PA':
            theta0 = np.array([current_best_row[0], current_best_row[1], current_best_row[2], current_best_row[3], current_best_row[4], current_best_row[12], current_best_row[13], current_best_row[11]], dtype=float)
            theta_true = np.array([sig_row[0], sig_row[1], sig_row[2], sig_row[3], sig_row[4], sig_row[12], sig_row[13], sig_row[11]], dtype=float)
            ndim = 8
        else:
            theta0 = np.array([current_best_row[0], current_best_row[1], current_best_row[2], current_best_row[3], current_best_row[4], current_best_row[16], current_best_row[12], current_best_row[13], current_best_row[11]], dtype=float)
            theta_true = np.array([sig_row[0], sig_row[1], sig_row[2], sig_row[3], sig_row[4], sig_row[16], sig_row[12], sig_row[13], sig_row[11]], dtype=float)
            ndim = 9
            
        global_best_no_max = {"snr": -1e60, "theta": None}
        obj_no_max = objective_factory(pa_template, 'optimal_snr', ctx, False, False, True, global_best_no_max)
        s0 = float(obj_no_max(theta0)); global_best_no_max["snr"] = s0; global_best_no_max["theta"] = theta0.copy()
        
        print(f"\n[OPTIMIZING INDEX {idx}] Target Mismatch: {all_optimized_mismatch[idx]:.4f}")
        
        print("\n[LAYER 1] Two-Stage Discovery (750s limit)...")
        seeds_to_try = []
        for base_p in [theta_true, theta0]:
            seeds_to_try.append(base_p.copy())
            for m1_f in [0.9, 1.0, 1.1]:
                for p0_shift in [-3.0, 3.0]:
                    p = base_p.copy(); p[0] *= m1_f; p[3] += p0_shift; seeds_to_try.append(p)
            if base_p[2] < -0.1: 
                for a_v in [-0.95, -0.6, -0.2]:
                    for m1_f in [0.85, 1.15]:
                        for p0_s in [-4.0, 4.0]:
                            p_a = base_p.copy(); p_a[0] *= m1_f; p_a[2] = a_v; p_a[3] += p0_s; seeds_to_try.append(p_a)
        
        print(f"  [STAGE 1.A] Fast screening {len(seeds_to_try)} seeds (60 iter)...")
        screening_results = []; obj_scr = objective_factory(pa_template, 'optimal_snr', ctx, True, True, True, global_best_no_max)
        for p in seeds_to_try:
            if time.time() - start_time_all > 400: break
            res = nelder_mead_optimize(p, lambda x: -float(obj_scr(x)), 60, 1e-2)
            screening_results.append({'theta': res.x, 'snr': -res.fun})
        
        screening_results.sort(key=lambda x: x['snr'], reverse=True)
        top_seeds = [r['theta'] for r in screening_results[:15]]
        
        print(f"  [STAGE 1.B] Deep refining top {len(top_seeds)} seeds (200 iter)...")
        l1_candidates = []
        for p in top_seeds:
            if time.time() - start_time_all > 750: break
            res = nelder_mead_optimize(p, lambda x: -float(obj_scr(x)), 200, 1e-4)
            l1_candidates.append({'theta': res.x, 'snr': -res.fun})
            
        l1_candidates.sort(key=lambda x: x['snr'], reverse=True); unique_l1 = []
        for cand in l1_candidates:
            if not any(np.allclose(cand['theta'], u, rtol=1e-3) for u in unique_l1): unique_l1.append(cand['theta'])
            if len(unique_l1) >= 12: break
            
        print(f"\n[LAYER 2] PARIS Search with {len(unique_l1)} seeds...")
        l1_arr = np.array(unique_l1); center_l2 = l1_arr[0, :ndim]
        if center_l2[2] < -0.6: # Extreme retrograde
            paddings = np.array([450000.0, 3500.0, 0.85, 25.0, 0.35, 1.8, 1.8, np.pi]) if ndim==8 else np.array([450000.0, 3500.0, 0.85, 25.0, 0.35, 0.35, 1.8, 1.8, np.pi])
        elif center_l2[2] < -0.2:
            paddings = np.array([300000.0, 2500.0, 0.7, 20.0, 0.28, 1.4, 1.4, np.pi]) if ndim==8 else np.array([300000.0, 2500.0, 0.7, 20.0, 0.28, 0.28, 1.4, 1.4, np.pi])
        else:
            paddings = np.array([180000.0, 1500.0, 0.5, 15.0, 0.2, 1.0, 1.0, np.pi]) if ndim==8 else np.array([180000.0, 1500.0, 0.5, 15.0, 0.2, 0.2, 1.0, 1.0, np.pi])
        
        cov_l2 = np.diag(paddings**2)
        if len(l1_arr) > 1:
            w = np.linspace(1.0, 0.3, len(l1_arr)); mean_w = np.average(l1_arr[:, :ndim], axis=0, weights=w)
            d = l1_arr[:, :ndim] - mean_w; cov_l2 = (d.T @ (w[:, None] * d)) / np.sum(w) + np.diag((0.15 * paddings)**2)
        evals, evecs = np.linalg.eigh(cov_l2); b_l2 = 12.0 * np.sqrt(np.maximum(evals, 1e-32))
        
        def paris_cb(s_obj, it):
            if it % 200 == 0:
                best_t_cand = global_best_no_max["theta"]
                if best_t_cand is not None:
                    p_fi, nsnr = real_overlap_lock(best_t_cand, ctx, pa_template, 80, maximize_time=False, include_mass_spin=True)
                    mm = 1.0 - (nsnr / utils.TARGET_SNR)
                    if mm < all_optimized_mismatch[idx]:
                        all_optimized_mismatch[idx] = mm; nr = sig_row.copy()
                        if pa_template == '0PA': nr[:5] = p_fi[:5]; nr[12], nr[13], nr[11] = p_fi[5], p_fi[6], p_fi[7]
                        else: nr[:5] = p_fi[:5]; nr[16], nr[12], nr[13], nr[11] = p_fi[5], p_fi[6], p_fi[7], p_fi[8]
                        all_optimized_params[idx] = nr; np.save(params_filename, all_optimized_params); np.save(mismatch_filename, all_optimized_mismatch)
                        print(f"  [CHECKPOINT] New best real mismatch: {mm:.4f}")
                    
        obj_l2 = objective_factory(pa_template, 'optimal_snr', ctx, True, True, True, global_best_no_max, (400, 1800, 1.2))
        global _PARIS_REF_CENTER, _PARIS_SPREAD_SCALE, _PARIS_OBJECTIVE, _PARIS_TARGET_KIND, _PARIS_EARLY_STOP_HIT, _PARIS_AFFINE_CENTER, _PARIS_AFFINE_Q, _PARIS_AFFINE_B, _PARIS_DIM, _PARIS_FULL_DIM
        _PARIS_REF_CENTER = center_l2.copy(); _PARIS_SPREAD_SCALE = 1.0; _PARIS_OBJECTIVE = obj_l2; _PARIS_TARGET_KIND = 'optimal_snr'
        _PARIS_DIM, _PARIS_FULL_DIM = ndim, ndim; _PARIS_AFFINE_CENTER, _PARIS_AFFINE_Q, _PARIS_AFFINE_B = center_l2.copy(), evecs, b_l2
        config_l2 = parismc.SamplerConfig(alpha=800, trail_size=1200, use_beta=True, gamma=15, use_pool=False)
        sampler_l2 = parismc.Sampler(ndim, 8, paris_log_density, [0.01 * np.diag(np.diag(cov_l2)) for _ in range(8)], paris_prior_transform, config_l2)
        u_s = np.clip(paris_inverse_prior_transform(np.vstack([l1_arr[:, :ndim], global_best_no_max["theta"][:ndim]])), 0.01, 0.99)
        v_s = paris_log_density(paris_prior_transform(u_s))
        from smt.sampling_methods import LHS; lhs_pts = LHS(xlimits=np.column_stack([np.zeros(ndim), np.ones(ndim)]))(600); lhs_v = paris_log_density(paris_prior_transform(lhs_pts))
        sampler_l2.run_sampling(2000, os.path.join(sampler_base, f"idx_{idx}_l2"), 100, external_lhs_points=np.vstack([u_s, lhs_pts]), external_lhs_log_densities=np.concatenate([v_s, lhs_v]), callback=paris_cb)
        
        print("\n[LAYER 3] Multi-Stage Polish..."); best_no_max_p = global_best_no_max["theta"]
        pts = sampler_l2.searched_points_list[0][:sampler_l2.element_num_list[0]]; vls = sampler_l2.searched_log_densities_list[0][:sampler_l2.element_num_list[0]]
        top_idx = np.argsort(vls)[-300:][::-1]; cands = []; added = []
        # Priority 1: Best point found without maximization
        if best_no_max_p is not None: cands.append(('no_max', best_no_max_p)); added.append(best_no_max_p)
        # Priority 2: Top PARIS points
        for ti in top_idx:
            ph = paris_prior_transform(pts[ti])
            if not any(np.allclose(ph, a, rtol=1e-3) for a in added): cands.append(('time_max', ph)); added.append(ph)
            if len(cands) >= 10: break
            
        for kind, cp in cands:
            if time.time() - start_time_all > 3500: break
            
            if kind == 'no_max':
                # Special direct polish for already-good t=0 points
                print(f"  [POLISH] Direct t=0 refinement for global best no-max point (SNR: {global_best_no_max['snr']:.4f})")
                curr_x = cp
            else:
                # Standard walk for time-maximized mountains
                obj_tm = objective_factory(pa_template, 'optimal_snr', ctx, True, True, True, global_best_no_max)
                res_tm = nelder_mead_optimize(cp, lambda x: -float(obj_tm(x)), 300, 1e-6)
                curr_x = res_tm.x
                # More gradual penalty walk to robustly slide the peak to t=0
                for w_target in [0.02, 0.1, 0.5, 2.5, 12.0, 60.0, 300.0, 1500.0, 6000.0]:
                    obj_walk = objective_factory(pa_template, 'optimal_snr', ctx, True, True, True, global_best_no_max, (0, 1, w_target))
                    res_walk = nelder_mead_optimize(curr_x, lambda x: -float(obj_walk(x)), 250, 1e-7)
                    curr_x = res_walk.x
            
            obj_nm = objective_factory(pa_template, 'optimal_snr', ctx, False, False, True, global_best_no_max)
            res_nm = nelder_mead_optimize(curr_x, lambda x: -float(obj_nm(x)), 800, 1e-10)
            
            # Final high-precision surgical phase and physical parameter lock
            p_fi, nsnr = real_overlap_lock(res_nm.x, ctx, pa_template, 250, maximize_time=False, include_mass_spin=True)
            
            curr_miss = 1.0 - (nsnr / utils.TARGET_SNR)
            if curr_miss < all_optimized_mismatch[idx]:
                all_optimized_mismatch[idx] = curr_miss; nr = sig_row.copy()
                if pa_template == '0PA': nr[:5] = p_fi[:5]; nr[12], nr[13], nr[11] = p_fi[5], p_fi[6], p_fi[7]
                else: nr[:5] = p_fi[:5]; nr[16], nr[12], nr[13], nr[11] = p_fi[5], p_fi[6], p_fi[7], p_fi[8]
                all_optimized_params[idx] = nr; np.save(params_filename, all_optimized_params); np.save(mismatch_filename, all_optimized_mismatch)
                print(f"  [FINAL POLISH] New best mismatch: {all_optimized_mismatch[idx]:.6f}")
        break

if __name__ == '__main__': main()
