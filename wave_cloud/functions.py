"""
functions.py
------------
All Wave Functions — registered and ready for serverless invocation.

Each function mirrors a Google Cloud Function:
  - Stateless
  - Accepts WaveContext (JSON payload)
  - Returns dict (JSON-serializable)

Available functions
-------------------
fn_lyapunov     : Maximal Lyapunov exponent of heart-EM system
fn_bifurcation  : Bifurcation diagram data
fn_scaling_law  : g_crit(w0) scaling law
fn_synapse      : THz-polarization synaptic weight update
fn_polariton    : hBN PhP dispersion curve
fn_stdp         : STDP kernel values
fn_hbn_epsilon  : hBN dielectric tensor at given frequencies
"""

import numpy as np
from .registry import wave_function
from .context  import WaveContext


# ════════════════════════════════════════════════════════════════════
# Wave Functions — wavephysai-heart
# ════════════════════════════════════════════════════════════════════

@wave_function("lyapunov")
def fn_lyapunov(ctx: WaveContext) -> dict:
    """
    Compute maximal Lyapunov exponent λ_max.

    Payload keys
    ------------
    w0  : float  — heart frequency [rad/s]  (default 1.0)
    g1  : float  — EM-heart coupling        (default 0.8)
    T   : float  — integration time [s]     (default 200)
    wE  : float  — EM frequency             (default 2.0)
    wp  : float  — vascular frequency       (default 1.5)
    mu  : float  — Van der Pol coefficient  (default 0.8)
    alpha: float — Duffing coefficient      (default 0.05)
    g2  : float  — heart-vascular coupling  (default 0.2)

    Returns
    -------
    lambda_max : float
    is_chaotic : bool
    params     : dict of used parameters
    """
    from scipy.integrate import solve_ivp

    w0    = ctx.get("w0",    1.0,  float)
    g1    = ctx.get("g1",    0.8,  float)
    T     = ctx.get("T",     200.0, float)
    wE    = ctx.get("wE",    2.0,  float)
    wp    = ctx.get("wp",    1.5,  float)
    mu    = ctx.get("mu",    0.8,  float)
    alpha = ctx.get("alpha", 0.05, float)
    g2    = ctx.get("g2",    0.2,  float)

    def rhs(t, y):
        E, dE, x, dx, p, dp = y
        return [dE,
                -wE**2*E - alpha*E**3 + g1*x,
                dx,
                -w0**2*x - mu*(x**2-1)*dx + g1*E + g2*p,
                dp,
                -wp**2*p + g2*x]

    y0  = [0.2, 0.0, 0.5, 0.0, 0.1, 0.0]
    y0p = [0.2 + 1e-6, 0.0, 0.5, 0.0, 0.1, 0.0]
    t_eval = np.arange(0, T, 0.05)

    sol  = solve_ivp(rhs, (0, T), y0,  t_eval=t_eval, rtol=1e-5, atol=1e-7)
    solp = solve_ivp(rhs, (0, T), y0p, t_eval=t_eval, rtol=1e-5, atol=1e-7)

    diff  = np.sqrt(np.sum((sol.y - solp.y)**2, axis=0))
    diff  = np.maximum(diff, 1e-300)
    skip  = len(t_eval) // 3
    lam   = float(np.mean(np.log(diff[skip:] / 1e-6)) / 0.05)

    return {
        "lambda_max": round(lam, 6),
        "is_chaotic": lam > 0,
        "params": {"w0": w0, "g1": g1, "T": T, "wE": wE, "wp": wp,
                   "mu": mu, "alpha": alpha, "g2": g2},
    }


@wave_function("bifurcation")
def fn_bifurcation(ctx: WaveContext) -> dict:
    """
    Compute bifurcation diagram data.

    Payload keys
    ------------
    w0       : float  — heart frequency (default 1.0)
    g1_min   : float  — coupling range start (default 0.05)
    g1_max   : float  — coupling range end   (default 1.5)
    n_points : int    — number of g1 values  (default 60)
    T        : float  — integration time [s] (default 250)

    Returns
    -------
    g1_values : list of g1
    x_peaks   : list of x peak values
    n_peaks   : total number of peaks collected
    """
    from scipy.integrate import solve_ivp
    from scipy.signal import find_peaks

    w0       = ctx.get("w0",       1.0,  float)
    g1_min   = ctx.get("g1_min",   0.05, float)
    g1_max   = ctx.get("g1_max",   1.5,  float)
    n_points = ctx.get("n_points", 60,   int)
    T        = ctx.get("T",        250.0, float)
    transient = 150.0

    def rhs(t, y, g1):
        E, dE, x, dx, p, dp = y
        wE=2.0; wp=1.5; mu=0.8; alpha=0.05; g2=0.2
        return [dE, -wE**2*E-alpha*E**3+g1*x, dx,
                -w0**2*x-mu*(x**2-1)*dx+g1*E+g2*p, dp, -wp**2*p+g2*x]

    g1_arr = np.linspace(g1_min, g1_max, n_points)
    g_out, x_out = [], []

    for g1 in g1_arr:
        t_eval = np.arange(transient, T, 0.05)
        sol = solve_ivp(rhs, (0, T), [0.2,0,0.5,0,0.1,0],
                        t_eval=t_eval, args=(g1,), rtol=1e-5, atol=1e-7)
        if not sol.success:
            continue
        peaks, _ = find_peaks(sol.y[2], height=0, distance=4)
        if len(peaks) == 0:
            continue
        vals = sol.y[2][peaks][-40:]
        g_out.extend([round(g1, 5)] * len(vals))
        x_out.extend([round(v, 6) for v in vals])

    return {
        "g1_values": g_out,
        "x_peaks":   x_out,
        "n_peaks":   len(x_out),
        "params":    {"w0": w0, "g1_min": g1_min,
                      "g1_max": g1_max, "n_points": n_points},
    }


@wave_function("scaling_law")
def fn_scaling_law(ctx: WaveContext) -> dict:
    """
    Compute g_crit(w0) scaling law.

    Payload keys
    ------------
    w0_min : float  (default 0.3)
    w0_max : float  (default 2.3)
    n_w0   : int    number of w0 values (default 8)
    T      : float  integration time per run (default 150)

    Returns
    -------
    w0_values  : list
    gcrit      : list
    fit_coeffs : [a, b, c]  for g_crit = a*w0^2 + b*w0 + c
    fit_label  : str
    """
    from scipy.integrate import solve_ivp

    w0_min = ctx.get("w0_min", 0.3,  float)
    w0_max = ctx.get("w0_max", 2.3,  float)
    n_w0   = ctx.get("n_w0",   8,    int)
    T      = ctx.get("T",      150.0, float)

    def get_lam(g1, w0):
        def rhs(t, y):
            E,dE,x,dx,p,dp=y
            return [dE,-4*E-0.05*E**3+g1*x,dx,
                    -w0**2*x-0.8*(x**2-1)*dx+g1*E+0.2*p,dp,-2.25*p+0.2*x]
        y0=[0.2,0,0.5,0,0.1,0]; y0p=[0.2+1e-6,0,0.5,0,0.1,0]
        t_e=np.arange(0,T,0.1)
        s=solve_ivp(rhs,(0,T),y0,t_eval=t_e,rtol=1e-4,atol=1e-6)
        sp=solve_ivp(rhs,(0,T),y0p,t_eval=t_e,rtol=1e-4,atol=1e-6)
        if s.success and sp.success:
            d=np.sqrt(np.sum((s.y-sp.y)**2,axis=0))
            d=np.maximum(d,1e-300); skip=len(t_e)//3
            return float(np.mean(np.log(d[skip:]/1e-6))/0.1)
        return 0.0

    w0_arr = np.linspace(w0_min, w0_max, n_w0)
    g_probe = np.linspace(0.05, 1.8, 20)
    gcrit = []
    for w0 in w0_arr:
        lams = [get_lam(g, w0) for g in g_probe]
        gc = 1.8
        for j in range(len(lams)-1):
            if lams[j] > 0 and lams[j+1] <= 0:
                gc = float((g_probe[j]+g_probe[j+1])/2)
                break
        gcrit.append(gc)

    coeffs = list(np.polyfit(w0_arr, gcrit, 2))
    a, b, c = coeffs
    label = f"g_crit = {a:.4f}·ω₀² + {b:.4f}·ω₀ + {c:.4f}"

    return {
        "w0_values":  [round(v, 4) for v in w0_arr],
        "gcrit":      [round(v, 4) for v in gcrit],
        "fit_coeffs": [round(v, 6) for v in coeffs],
        "fit_label":  label,
    }


# ════════════════════════════════════════════════════════════════════
# Wave Functions — graphene-hBN-synapse
# ════════════════════════════════════════════════════════════════════

@wave_function("synapse")
def fn_synapse(ctx: WaveContext) -> dict:
    """
    THz-polarization-controlled synaptic weight update.

    Payload keys
    ------------
    theta      : float  — THz polarization angle [°] (default 45)
    intensity  : float  — THz intensity [W/m²]       (default 5e5)
    n_pulses   : int    — number of pulses            (default 1)
    omega_cm   : float  — carrier frequency [cm⁻¹]   (default 790)
    E_F0       : float  — initial Fermi energy [eV]  (default 0.10)
    d_nm       : float  — hBN thickness [nm]         (default 10.0)

    Returns
    -------
    delta_EF_meV : Fermi level shift [meV]
    delta_G_uS   : conductance change [μS/sq]
    G_final      : final conductance [S/sq]
    plasticity   : "LTP" | "LTD" | "neutral"
    theta        : applied polarization [°]
    """
    theta     = ctx.get("theta",     45.0,  float)
    intensity = ctx.get("intensity", 5e5,   float)
    n_pulses  = ctx.get("n_pulses",  1,     int)
    omega_cm  = ctx.get("omega_cm",  790.0, float)
    E_F0      = ctx.get("E_F0",      0.10,  float)
    d_nm      = ctx.get("d_nm",      10.0,  float)

    # Polarization weights (Band I: LTP, Band II: LTD)
    th_rad = theta * np.pi / 180.0
    wI  = np.cos(th_rad)**2   # Band I (LTP)
    wII = np.sin(th_rad)**2   # Band II (LTD)

    # Fermi level shift per pulse (phenomenological)
    COUPLING = 5.0e-3  # eV·m²/W
    delta_EF_per_pulse = COUPLING * intensity * (wI - wII)
    delta_EF_total = delta_EF_per_pulse * n_pulses

    # Conductance change via Drude model
    E_Q   = 1.602e-19; HBAR = 1.055e-34
    kT    = 1.381e-23 * 300
    omega_rad = omega_cm * 2 * np.pi * 2.998e10
    tau   = 1e-13

    def sigma_intra(EF_eV):
        EF = EF_eV * E_Q
        ln = np.log(2 * np.cosh(EF / (2*kT)))
        return abs((E_Q**2 / (np.pi * HBAR**2)) * kT * ln
                   / complex(omega_rad, 1/tau))

    G0 = abs(sigma_intra(E_F0))
    G1 = abs(sigma_intra(max(E_F0 + delta_EF_total, 0.005)))
    delta_G = G1 - G0

    G_MIN, G_MAX = 1e-5, 1e-3
    G_final = float(np.clip(G0 + delta_G, G_MIN, G_MAX))

    if delta_EF_total > 1e-4:
        plasticity = "LTP"
    elif delta_EF_total < -1e-4:
        plasticity = "LTD"
    else:
        plasticity = "neutral"

    return {
        "delta_EF_meV": round(delta_EF_total * 1000, 4),
        "delta_G_uS":   round(delta_G * 1e6, 6),
        "G_final":      round(G_final, 9),
        "plasticity":   plasticity,
        "wI":           round(float(wI), 4),
        "wII":          round(float(wII), 4),
        "theta":        theta,
        "n_pulses":     n_pulses,
    }


@wave_function("polariton")
def fn_polariton(ctx: WaveContext) -> dict:
    """
    Compute hBN PhP dispersion q(ω) for a frequency range.

    Payload keys
    ------------
    band     : "I" or "II"   (default "I")
    d_nm     : float          hBN thickness [nm] (default 10)
    n_points : int            frequency points   (default 200)

    Returns
    -------
    omega_cm : list of frequencies [cm⁻¹]
    q_re     : list of Re[q] [μm⁻¹]
    q_im     : list of Im[q] [μm⁻¹]
    band     : "I" or "II"
    """
    band     = ctx.get("band",     "I",   str)
    d_nm     = ctx.get("d_nm",     10.0,  float)
    n_points = ctx.get("n_points", 200,   int)

    if band == "I":
        omega = np.linspace(762, 823, n_points)
        eps_inf_perp, omega_TO, omega_LO, gamma = 4.87, 760.0, 825.0, 5.0
        eps_inf_par,  oTO2,     oLO2,     gam2  = 2.95, 1370.0, 1610.0, 5.0
    else:
        omega = np.linspace(1380, 1600, n_points)
        eps_inf_perp, omega_TO, omega_LO, gamma = 4.87, 760.0, 825.0, 5.0
        eps_inf_par,  oTO2,     oLO2,     gam2  = 2.95, 1370.0, 1610.0, 5.0

    def eps_dl(w, ei, wTO, wLO, g):
        return ei * (1 + (wLO**2 - wTO**2)/(wTO**2 - w**2 - 1j*g*w))

    ep = eps_dl(omega, eps_inf_perp, omega_TO, omega_LO, gamma)
    ez = eps_dl(omega, eps_inf_par,  oTO2,      oLO2,     gam2)

    d   = d_nm * 1e-9
    ratio = np.sqrt(-ep / (ez + 1e-30j))
    phi   = np.arctan(1.0 / (ratio + 1e-30))
    q     = (np.pi + phi) / d   # n=1 mode

    return {
        "omega_cm": [round(v, 2) for v in omega.tolist()],
        "q_re":     [round(v * 1e-6, 4) for v in np.real(q).tolist()],
        "q_im":     [round(v * 1e-6, 4) for v in np.imag(q).tolist()],
        "band":     band,
        "d_nm":     d_nm,
        "unit":     "μm⁻¹",
    }


@wave_function("stdp")
def fn_stdp(ctx: WaveContext) -> dict:
    """
    Compute STDP kernel ΔW(Δt).

    Payload keys
    ------------
    dt_min   : float  [ms]  (default -80)
    dt_max   : float  [ms]  (default  80)
    n_points : int          (default 200)
    A_plus   : float        (default 0.01)
    A_minus  : float        (default 0.012)
    tau_plus : float  [ms]  (default 20)
    tau_minus: float  [ms]  (default 20)

    Returns
    -------
    delta_t : list [ms]
    dW      : list [normalized]
    """
    dt_min    = ctx.get("dt_min",    -80.0, float)
    dt_max    = ctx.get("dt_max",     80.0, float)
    n_points  = ctx.get("n_points",  200,   int)
    A_plus    = ctx.get("A_plus",    0.01,  float)
    A_minus   = ctx.get("A_minus",   0.012, float)
    tau_plus  = ctx.get("tau_plus",  20.0,  float)
    tau_minus = ctx.get("tau_minus", 20.0,  float)

    dt = np.linspace(dt_min, dt_max, n_points)
    dW = np.where(dt > 0,
                  A_plus  * np.exp(-dt / tau_plus),
                 -A_minus * np.exp( dt / tau_minus))

    return {
        "delta_t":  [round(v, 3) for v in dt.tolist()],
        "dW":       [round(v, 6) for v in dW.tolist()],
        "A_plus":   A_plus,
        "A_minus":  A_minus,
        "tau_plus": tau_plus,
        "tau_minus":tau_minus,
    }


@wave_function("hbn_epsilon")
def fn_hbn_epsilon(ctx: WaveContext) -> dict:
    """
    Compute hBN dielectric tensor at given frequencies.

    Payload keys
    ------------
    omega_min : float [cm⁻¹]  (default 600)
    omega_max : float [cm⁻¹]  (default 1700)
    n_points  : int            (default 300)

    Returns
    -------
    omega_cm  : list
    eps_perp_re, eps_perp_im : list (in-plane, Band I)
    eps_par_re,  eps_par_im  : list (out-of-plane, Band II)
    band_I    : [760, 825]   cm⁻¹
    band_II   : [1370, 1610] cm⁻¹
    """
    omega_min = ctx.get("omega_min", 600.0,  float)
    omega_max = ctx.get("omega_max", 1700.0, float)
    n_points  = ctx.get("n_points",  300,    int)

    omega = np.linspace(omega_min, omega_max, n_points)

    def eps_dl(w, ei, wTO, wLO, g):
        return ei * (1 + (wLO**2 - wTO**2)/(wTO**2 - w**2 - 1j*g*w))

    ep = eps_dl(omega, 4.87, 760.0,  825.0,  5.0)
    ez = eps_dl(omega, 2.95, 1370.0, 1610.0, 5.0)

    return {
        "omega_cm":    [round(v, 1) for v in omega.tolist()],
        "eps_perp_re": [round(v, 4) for v in np.real(ep).tolist()],
        "eps_perp_im": [round(v, 4) for v in np.imag(ep).tolist()],
        "eps_par_re":  [round(v, 4) for v in np.real(ez).tolist()],
        "eps_par_im":  [round(v, 4) for v in np.imag(ez).tolist()],
        "band_I":      [760, 825],
        "band_II":     [1370, 1610],
    }
