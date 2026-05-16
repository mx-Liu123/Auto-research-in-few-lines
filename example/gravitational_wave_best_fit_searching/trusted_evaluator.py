import os
import sys
import json
import numpy as np

# Add current directory to path to import utils
sys.path.append(os.getcwd())
import utils

# FEW / waveform & noise
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits

def build_waveform_response(T: float, dt: float, use_gpu: bool = False) -> ResponseWrapper:
    """Create a LISA ResponseWrapper consistent with modular_optimization-Fisher.py."""
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
    return response

def evaluate_all(pa_template='0PA'):
    # 1. Load ground truth
    true_params_all = np.load("signal_parameter_array_IMRI.npy")
    num_pts = true_params_all.shape[0]
    
    # 2. Load optimized parameters
    opt_file = f"optimized_params_IMRI_{pa_template}.npy"
    if not os.path.exists(opt_file):
        print(f"Error: {opt_file} not found")
        return num_pts, [1.0] * num_pts
    
    opt_params_all = np.load(opt_file)
    
    unfinished_indices = []
    all_mismatches = []

    # Use a cached response wrapper to avoid rebuilding if T/dt are constant
    cached_response = None
    cached_T_dt = (None, None)

    for idx in range(num_pts):
        theta_true = true_params_all[idx]
        theta_opt = opt_params_all[idx]
        
        (
            m1, m2, a, p0, e0, Y0,
            dist, qS, phiS, qK, phiK,
            Phi_phi0, Phi_theta0, Phi_r0,
            dt, T, chi2
        ) = theta_true
        
        if cached_T_dt != (T, dt):
            cached_response = build_waveform_response(T=T, dt=dt, use_gpu=True)
            cached_T_dt = (T, dt)
        
        waveform_response = cached_response
        
        # PSD setup
        from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens
        from stableemrifisher.utils import generate_PSD
        channels = [A1TDISens, E1TDISens, T1TDISens]
        noise_kwargs = [{"sens_fn": ch} for ch in channels]
        
        # Generate 2PA ground truth
        wave_params_true = [
            m1, m2, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, 
            chi2, True, False, True  # evolve_1PA, evolve_primary, evolve_2PA
        ]
        emri_kwargs_2pa = {"T": T, "dt": dt, '1PA': True, 'evolve_primary': False, '2PA': True}
        waveform_2pa = waveform_response(*wave_params_true, **emri_kwargs_2pa)
        
        PSD_funcs = generate_PSD(
            waveform=waveform_2pa,
            dt=dt,
            noise_PSD=get_sensitivity,
            channels=channels,
            noise_kwargs=noise_kwargs,
            use_gpu=True,
        )
        
        N_fiducial = len(waveform_2pa[0])
        waveform_2pa_fft = utils.compute_fft_with_windowing(waveform_2pa, dt, use_gpu=True)
        
        # Use all parameters directly from the optimized array
        (
            m1_o, m2_o, a_o, p0_o, e0_o, Y0_o,
            dist_o, qS_o, phiS_o, qK_o, phiK_o,
            Phi_phi0_o, Phi_theta0_o, Phi_r0_o,
            dt_o, T_o, chi2_o
        ) = theta_opt

        # Calculate overlap WITHOUT maximization using the optimized row
        overlap = utils.calculate_optimal_snr_0pa_vs_2pa(
            m1_o, m2_o, a_o, p0_o, e0_o, Y0_o,
            dist_o, qS_o, phiS_o, qK_o, phiK_o,
            Phi_phi0_o, Phi_theta0_o, Phi_r0_o,
            chi2_o,
            waveform_response=waveform_response,
            PSD=PSD_funcs,
            dt=dt, T=T,
            N_fiducial=N_fiducial,
            waveform_2pa_fft=waveform_2pa_fft,
            xp=np,
            use_gpu=True,
            maximize_phase=False,
            maximize_time=False
        )
        
        mismatch = 1.0 - float(overlap)
        all_mismatches.append(mismatch)
        if mismatch > 0.1:
            unfinished_indices.append(idx)
            
    return len(unfinished_indices), all_mismatches

if __name__ == "__main__":
    try:
        unfinished_count, mismatches = evaluate_all()
        mismatch_array = np.array(mismatches)
        # 核心指标：未完成点数 (整数) + mismatch 平方均值 (小数)
        # 只要有一个点变好且没有点变差，这个指标就一定会变小
        refined_metric = float(unfinished_count) + np.mean(mismatch_array**2)
        
        result = {
            "unfinished_count": unfinished_count,
            "mismatches": mismatches,
            "status": "success",
            "metric": refined_metric 
        }
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
