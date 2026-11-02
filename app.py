import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Oil Reservoir Pressure & Water Influx",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CUSTOM CSS (explicit colors: safe in light and dark themes)
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 32px; font-weight: 800; color: #174A6E;
            text-align: center; margin-bottom: 5px;
        }
        .subtitle {
            font-size: 16px; color: #64748b;
            text-align: center; margin-bottom: 25px;
        }
        .card {
            background: #f0f9ff; border: 2px solid #0ea5e9;
            border-radius: 14px; padding: 16px 12px;
            text-align: center; margin: 6px 0 12px 0;
        }
        .card .label {
            font-size: 15px; font-weight: 600; color: #075985;
        }
        .card .value {
            font-size: 26px; font-weight: 800; color: #0c4a6e;
            direction: ltr;
        }
        .card .unit { font-size: 13px; color: #475569; }
        .card.green {
            background: #f0fdf4; border-color: #38a169;
        }
        .card.green .label { color: #166534; }
        .card.green .value { color: #14532d; }
        .card.orange {
            background: #fff7ed; border-color: #f97316;
        }
        .card.orange .label { color: #9a3412; }
        .card.orange .value { color: #7c2d12; }
    </style>
    """,
    unsafe_allow_html=True,
)


def card(label, value, unit="", color=""):
    st.markdown(
        f"""
        <div class="card {color}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="unit">{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 3. HELPERS
# ============================================================

ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩٫", "0123456789."
)


def ei_negative(x):
    """Ei(-x) for x > 0 (series for x < 1, continued fraction otherwise)."""
    if not np.isfinite(x) or x <= 0:
        raise ValueError("x must be a finite positive number.")

    gamma = 0.5772156649015329
    eps = 1e-14
    max_iter = 500

    if x < 1.0:
        term = 1.0
        series_sum = 0.0
        for k in range(1, max_iter + 1):
            term *= -x / k
            addend = term / k
            series_sum += addend
            if abs(addend) < eps * max(1.0, abs(series_sum)):
                break
        return -gamma - math.log(x) - series_sum

    b = x + 1.0
    c = 1.0 / 1e-300
    d = 1.0 / b
    h_cf = d
    for i in range(1, max_iter + 1):
        a = -float(i * i)
        b += 2.0
        d = a * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + a / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = c * d
        h_cf *= delta
        if abs(delta - 1.0) < eps:
            break
    return -(h_cf * math.exp(-x))


def wed_ved(t_d):
    """Dimensionless water influx WeD(tD) for an infinite radial aquifer
    (Van Everdingen-Hurst), three-regime correlation."""
    if t_d <= 0:
        return 0.0
    if t_d < 0.01:
        return 2.0 * math.sqrt(t_d / math.pi)
    if t_d <= 200.0:
        s = math.sqrt(t_d)
        num_ = (1.2838 * s + 1.19328 * t_d
                + 0.269872 * t_d**1.5 + 0.00855294 * t_d**2)
        den_ = 1.0 + 0.616599 * s + 0.0413008 * t_d
        return num_ / den_
    # tD > 200 (published constants: +2.02566 and -4.29881)
    return (-4.29881 + 2.02566 * t_d) / math.log(t_d)


def parse_times(text):
    cleaned = text.translate(ARABIC_DIGITS).replace("،", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        raise ValueError("أدخل الأزمنة مفصولة بفواصل.")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise ValueError("الأزمنة يجب أن تكون أرقامًا مفصولة بفواصل.")
    arr = np.array(values, dtype=float)
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError("الأزمنة يجب أن تكون أرقامًا غير سالبة.")
    return np.unique(arr)  # sorted, no duplicates


def num(label, key, **kwargs):
    """Number input that starts empty."""
    return st.number_input(label, value=None, key=key, **kwargs)


def positive(value, name):
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")


# ============================================================
# 4. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Oil Reservoir Pressure & Water Influx</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Exponential Integral · Pot Aquifer · '
    "Schilthuis Steady-State · Van Everdingen-Hurst</div>",
    unsafe_allow_html=True,
)
st.info(
    "يحسب التطبيق الضغط عند نقطة مراقبة متأثرة بثلاثة آبار إنتاجية "
    "(مبدأ التراكب)، ثم الدخول المائي بثلاثة نماذج: بوت وشيلثويس "
    "وفان إفرامجينغن-هيرست (VED)."
)


# ============================================================
# 5. PERSONAL VARIABLE + TIMES
# ============================================================

st.header("1. المتغير الشخصي (i)")

i_value = num(
    "قيمة المتغير الشخصي i",
    "i_value",
    min_value=1,
    max_value=100000,
    step=1,
    help="يُستخدم لحساب q1 و raq تلقائيًا.",
)

q1 = raq = None
if i_value is not None:
    q1 = 100.0 + 10.0 * (i_value - 1)
    raq = 2500.0 + 100.0 * (i_value - 1)
    c1, c2 = st.columns(2)
    with c1:
        card("معدل إنتاج البئر الأول q1", f"{q1:,.2f}", "STB/day")
    with c2:
        card("نصف قطر الحافة المائية raq", f"{raq:,.2f}", "ft", "orange")

times_text = st.text_input(
    "الأزمنة (مفصولة بفواصل)",
    key="times_text",
    placeholder="0, 200, 400, 500, 800, 1000, 2000, 3000",
)
time_unit = st.radio(
    "وحدة الأزمنة المُدخلة",
    ["hours", "days"],
    horizontal=True,
    key="time_unit",
    help="تُحوَّل داخليًا: الساعات لمعادلة Ei، والأيام لتكامل شيلثويس.",
)


# ============================================================
# 6. PETROLEUM PARAMETERS
# ============================================================

st.header("2. الثوابت البترولية")

col1, col2, col3, col4 = st.columns(4)

with col1:
    Pi = num("الضغط البدائي Pi [psi]", "Pi", min_value=0.001)
    K = num("نفوذية المكمن K [mD]", "K", min_value=0.001)
    re_in = num(
        "نصف القطر الخارجي للمكمن re [ft] (فارغ = 2000)",
        "re",
        min_value=0.001,
    )

with col2:
    phi_percent = num(
        "المسامية phi [%]", "phi", min_value=0.001, max_value=100.0
    )
    ct = num(
        "الانضغاطية الكلية ct [psi⁻¹]",
        "ct",
        min_value=1e-12,
        format="%.3e",
    )
    K_aq_in = num(
        "نفوذية الطبقة المائية K_aq [mD] (فارغ = K)",
        "K_aq",
        min_value=0.001,
    )

with col3:
    B = num("عامل حجم النفط B [bbl/STB]", "B", min_value=0.001)
    h = num("ثخانة المكمن h [ft]", "h", min_value=0.001)
    h_aq = num("ثخانة الطبقة المائية h_aq [ft]", "h_aq", min_value=0.001)

with col4:
    mu = num("لزوجة النفط mu [cP]", "mu", min_value=0.001)
    mu_w_in = num(
        "لزوجة الماء mu_w [cP] (فارغ = mu)", "mu_w", min_value=0.001
    )


# ============================================================
# 7. MONITORING POINT AND WELLS
# ============================================================

st.header("3. إحداثيات الآبار ونقطة المراقبة")

st.subheader("نقطة المراقبة")
m1, m2, m3, m4 = st.columns(4)
with m1:
    point_name = st.text_input(
        "اسم نقطة المراقبة", key="pname", placeholder="Monitoring Point"
    )
with m2:
    point_symbol = st.text_input(
        "رمز النقطة", key="psym", max_chars=5, placeholder="M"
    )
with m3:
    Mx = num("إحداثي Mx [ft]", "Mx")
with m4:
    My = num("إحداثي My [ft]", "My")

st.subheader("بيانات الآبار الإنتاجية")

well_names = ["Well 1", "Well 2", "Well 3"]
well_x, well_y, well_q = [], [], []
for n, name in enumerate(well_names):
    a, b_, c_, d_ = st.columns(4)
    with a:
        st.markdown(f"**{name}**")
    with b_:
        well_x.append(num(f"X{n + 1} [ft]", f"wx{n}"))
    with c_:
        well_y.append(num(f"Y{n + 1} [ft]", f"wy{n}"))
    with d_:
        if n == 0:
            well_q.append(q1)
            card("q1 (تلقائي)", "—" if q1 is None else f"{q1:,.2f}", "STB/day")
        else:
            well_q.append(
                num(
                    f"q{n + 1} [STB/day]",
                    f"wq{n}",
                    min_value=0.0,
                )
            )


# ============================================================
# 8. CHECK INPUTS
# ============================================================

required = {
    "i": i_value, "Pi": Pi, "K": K, "phi": phi_percent, "ct": ct,
    "B": B, "h": h, "h_aq": h_aq, "mu": mu, "Mx": Mx, "My": My,
}
required.update({f"X{n+1}": v for n, v in enumerate(well_x)})
required.update({f"Y{n+1}": v for n, v in enumerate(well_y)})
required.update({f"q{n+1}": v for n, v in enumerate(well_q)})
missing = [k for k, v in required.items() if v is None]

if missing or not times_text.strip():
    st.warning(
        "أكمل جميع الحقول المطلوبة لعرض النتائج. "
        + (f"الحقول الناقصة: {', '.join(missing)}" if missing else "")
        + ("  — الأزمنة" if not times_text.strip() else "")
    )
    st.stop()


# ============================================================
# 9. CALCULATIONS
# ============================================================

try:
    for val, nm in [
        (Pi, "Pi"), (K, "K"), (ct, "ct"), (B, "B"), (h, "h"),
        (h_aq, "h_aq"), (mu, "mu"), (phi_percent, "phi"),
    ]:
        positive(val, nm)

    phi = phi_percent / 100.0
    re = 2000.0 if re_in is None else float(re_in)
    K_aq = K if K_aq_in is None else float(K_aq_in)
    mu_w = mu if mu_w_in is None else float(mu_w_in)
    positive(re, "re")
    positive(K_aq, "K_aq")
    positive(mu_w, "mu_w")

    if raq <= re:
        raise ValueError(
            f"raq ({raq:,.0f}) يجب أن يكون أكبر من re ({re:,.0f})."
        )
    if any(q < 0 for q in well_q):
        raise ValueError("معدلات الإنتاج السالبة غير مدعومة.")

    times_in = parse_times(times_text)
    t_hours = times_in * (24.0 if time_unit == "days" else 1.0)
    t_days = t_hours / 24.0

    # Aquifer constants (automatic).
    C_sch = (0.00708 * K_aq * h_aq) / (mu_w * math.log(raq / re))
    V_pore = math.pi * (raq**2 - re**2) * h_aq * phi  # ft3
    V_pore_bbl = V_pore / 5.615

    # Diffusivity and distances.
    y_diff = 0.0002637 * K / (phi * mu * ct)  # ft2/hour
    wx = np.array(well_x, dtype=float)
    wy = np.array(well_y, dtype=float)
    q = np.array(well_q, dtype=float)
    r = np.sqrt((Mx - wx) ** 2 + (My - wy) ** 2)
    if np.any(r <= 0):
        raise ValueError(
            "نقطة المراقبة تقع على إحداثيات أحد الآبار (r = 0)."
        )

    coef = 70.6 * q * mu * B / (K * h)  # psi per unit(-Ei)

    # Per-time pressure.
    n_t = len(t_hours)
    x_mat = np.full((n_t, 3), np.nan)
    dp_mat = np.zeros((n_t, 3))
    for k, th in enumerate(t_hours):
        if th <= 0:
            continue
        x_mat[k] = 948.0 * phi * mu * ct * r**2 / (K * th)
        ei = np.array([ei_negative(float(v)) for v in x_mat[k]])
        dp_mat[k] = coef * (-ei)

    dp_total = dp_mat.sum(axis=1)
    P_M = Pi - dp_total
    ri_t = np.sqrt(y_diff * t_hours)

    # Water influx.
    We_pot = -V_pore_bbl * ct * (Pi - P_M)  # bbl

    t_int = t_days.copy()
    dp_int = (Pi - P_M).copy()
    if t_int[0] > 0:  # start integration from t = 0 with dP = 0
        t_int = np.concatenate([[0.0], t_int])
        dp_int = np.concatenate([[0.0], dp_int])
    cum = np.concatenate(
        [[0.0], np.cumsum(-0.5 * (dp_int[1:] + dp_int[:-1]) * np.diff(t_int))]
    )
    cum = cum[1:] if len(cum) > n_t else cum
    We_sch = C_sch * cum  # bbl

    # Van Everdingen-Hurst (superposition, dimensionless WeD).
    B_ved = 1.119 * phi * h_aq * (re**2) * ct  # bbl/psi
    tD_coef = 0.00633 * K_aq / (phi * mu_w * ct * (re**2))  # 1/day
    t_D_total = tD_coef * t_days  # tD measured from t = 0

    t_sup, P_sup = t_days.copy(), P_M.copy()
    if t_sup[0] > 0:  # prepend t = 0 with P = Pi
        t_sup = np.concatenate([[0.0], t_sup])
        P_sup = np.concatenate([[Pi], P_sup])
    We_full = np.zeros(len(t_sup))
    for n_idx in range(1, len(t_sup)):
        acc = 0.0
        for j in range(1, n_idx + 1):
            dP_j = -P_sup[j - 1] + P_sup[j]
            tD_j = tD_coef * (t_sup[n_idx] - t_sup[j - 1])
            acc += dP_j * wed_ved(tD_j)
        We_full[n_idx] = B_ved * acc
    We_ved = We_full[1:] if len(We_full) > n_t else We_full  # bbl

    # --------------------------------------------------------
    # 10. GENERAL RESULTS
    # --------------------------------------------------------
    st.header("4. النتائج العامة")

    g1, g2, g3, g4, g5 = st.columns(5)
    with g1:
        card("معامل شيلثويس C_sch", f"{C_sch:,.4f}", "bbl/day/psi")
    with g2:
        card("الحجم الفعال V_pore", f"{V_pore:,.4g}", "ft³", "green")
    with g3:
        card("raq", f"{raq:,.2f}", "ft", "orange")
    with g4:
        card("ri عند أكبر زمن", f"{ri_t[-1]:,.2f}", "ft", "green")
    with g5:
        card("ثابت VED: B_ved", f"{B_ved:,.4g}", "bbl/psi")

    general = pd.DataFrame(
        {
            "Well": well_names,
            "q [STB/day]": q,
            "y [ft²/hr]": [y_diff] * 3,
            "ri [ft]": [ri_t[-1]] * 3,
            "raq [ft]": [raq] * 3,
            "r [ft]": r,
        }
    )
    st.dataframe(
        general.style.format(
            {
                "q [STB/day]": "{:,.2f}",
                "y [ft²/hr]": "{:,.4f}",
                "ri [ft]": "{:,.2f}",
                "raq [ft]": "{:,.2f}",
                "r [ft]": "{:,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"re = {re:,.0f} ft · K_aq = {K_aq:,.2f} mD · mu_w = {mu_w:,.2f} cP · "
        f"V_pore = {V_pore_bbl:,.4g} bbl. "
        "ri = √(y·t) وفق المعادلة المحددة في المسألة."
    )

    # --------------------------------------------------------
    # 11. TIME TABLE
    # --------------------------------------------------------
    st.header("5. الجدول الزمني المفصل")

    detailed = pd.DataFrame(
        {
            f"t [{time_unit}]": times_in,
            "t [hours]": t_hours,
            "ri [ft]": ri_t,
            "x1": x_mat[:, 0],
            "x2": x_mat[:, 1],
            "x3": x_mat[:, 2],
            "ΔP1 [psi]": dp_mat[:, 0],
            "ΔP2 [psi]": dp_mat[:, 1],
            "ΔP3 [psi]": dp_mat[:, 2],
            "ΔP total [psi]": dp_total,
            "P(M) [psi]": P_M,
            "tD (VED)": t_D_total,
            "We_pot [bbl]": We_pot,
            "We_sch [bbl]": We_sch,
            "We_ved [bbl]": We_ved,
        }
    )
    st.dataframe(
        detailed.style.format(
            {
                f"t [{time_unit}]": "{:,.2f}",
                "t [hours]": "{:,.2f}",
                "ri [ft]": "{:,.2f}",
                "x1": "{:.6g}",
                "x2": "{:.6g}",
                "x3": "{:.6g}",
                "ΔP1 [psi]": "{:,.4f}",
                "ΔP2 [psi]": "{:,.4f}",
                "ΔP3 [psi]": "{:,.4f}",
                "ΔP total [psi]": "{:,.4f}",
                "P(M) [psi]": "{:,.4f}",
                "We_pot [bbl]": "{:,.2f}",
                "We_sch [bbl]": "{:,.2f}",
                "tD (VED)": "{:.6g}",
                "We_ved [bbl]": "{:,.2f}",
            },
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.latex(r"P(M,t)=P_i-\sum_{j=1}^{3}\Delta P_j(t)")
    st.latex(r"W_{e,pot}=V_{pore}\,c_t\,(P_i-P)")
    st.latex(
        r"W_{e,sch}=C_{sch}\int_0^t (P_i-P)\,dt"
    )
    st.latex(
        r"W_{e,ved}(t_n)=B_{ved}\sum_{j=1}^{n}\Delta P_j\,"
        r"W_{eD}(t_{D,n}-t_{D,j-1}),\quad B_{ved}=1.119\,\phi\,h_{aq}\,r_e^2\,c_t"
    )

    if (P_M < 0).any():
        st.warning(
            "الضغط المحسوب أصبح سالبًا عند بعض الأزمنة؛ "
            "راجع المدخلات أو افتراضات النموذج."
        )

    last = len(times_in) - 1
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        card(
            f"P({point_symbol.strip() or 'M'}) عند t = {times_in[last]:,.2f}",
            f"{P_M[last]:,.4f}",
            "psi",
            "green",
        )
    with f2:
        card("We (بوت)", f"{We_pot[last]:,.2f}", "bbl")
    with f3:
        card("We (شيلثويس)", f"{We_sch[last]:,.2f}", "bbl", "orange")
    with f4:
        card("We (VED)", f"{We_ved[last]:,.2f}", "bbl", "green")

    # --------------------------------------------------------
    # 12. PLOTS (white background, dark text: theme-independent)
    # --------------------------------------------------------
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "#1e293b",
            "axes.labelcolor": "#1e293b",
            "axes.edgecolor": "#475569",
            "xtick.color": "#1e293b",
            "ytick.color": "#1e293b",
        }
    )

    st.header("6. مخطط مواقع الآبار")
    sym = point_symbol.strip() or "M"

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(wx, wy, s=140, c="#0369a1", label="Production Wells", zorder=3)
    ax.scatter([Mx], [My], s=320, marker="*", c="#dc2626",
               label="Monitoring Point", zorder=4)
    for name, xx, yy in zip(well_names, wx, wy):
        ax.annotate(name, (xx, yy), xytext=(7, 7), textcoords="offset points")
    ax.annotate(sym, (Mx, My), xytext=(8, 8), textcoords="offset points",
                fontsize=12, fontweight="bold")
    for xx, yy, rr in zip(wx, wy, r):
        ax.plot([Mx, xx], [My, yy], "--", lw=1, c="#64748b", alpha=0.8)
        ax.annotate(f"{rr:,.1f} ft", ((Mx + xx) / 2, (My + yy) / 2),
                    fontsize=9, color="#b45309")
    ax.set_xlabel("X [ft]")
    ax.set_ylabel("Y [ft]")
    ax.set_title("Production Wells and Monitoring Point")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.header("7. تطور الضغط والدخول المائي مع الزمن")
    p1, p2 = st.columns(2)

    with p1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(times_in, P_M, "o-", c="#0369a1", lw=2)
        ax.set_xlabel(f"Time [{time_unit}]")
        ax.set_ylabel("P(M) [psi]")
        ax.set_title("Pressure at Monitoring Point")
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with p2:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(times_in, We_pot, "o-", c="#16a34a", lw=2, label="Pot")
        ax.plot(times_in, We_sch, "s--", c="#ea580c", lw=2,
                label="Schilthuis")
        ax.plot(times_in, We_ved, "^-.", c="#7c3aed", lw=2,
                label="Van Everdingen-Hurst")
        ax.set_xlabel(f"Time [{time_unit}]")
        ax.set_ylabel("We [bbl]")
        ax.set_title("Cumulative Water Influx: Pot vs Schilthuis vs VED")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # --------------------------------------------------------
    # 13. DOWNLOAD
    # --------------------------------------------------------
    st.header("8. تصدير النتائج")
    st.download_button(
        label="تحميل الجدول الزمني المفصل CSV",
        data=detailed.to_csv(index=False).encode("utf-8-sig"),
        file_name="reservoir_time_results.csv",
        mime="text/csv",
    )

except ValueError as error:
    st.error(f"خطأ في المدخلات أو الحسابات: {error}")

except (OverflowError, ZeroDivisionError):
    st.error("حدث خطأ عددي أثناء الحساب. تحقق من قيم المدخلات.")
