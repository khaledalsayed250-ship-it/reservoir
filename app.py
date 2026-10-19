import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Reservoir Pressure Distribution",
    page_icon="🛢️",
    layout="wide",
)


# =========================================================
# 2. PAGE STYLE
# =========================================================

st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            color: #155e75;
        }

        .sub-title {
            text-align: center;
            font-size: 15px;
            color: #64748b;
            margin-bottom: 25px;
        }

        .result-card {
            background: linear-gradient(135deg, #dbeafe, #dcfce7);
            border: 2px solid #16a34a;
            border-radius: 14px;
            padding: 22px;
            text-align: center;
            margin: 15px 0;
        }

        .result-value {
            font-size: 34px;
            font-weight: bold;
            color: #166534;
        }

        .result-label {
            font-size: 17px;
            color: #14532d;
        }

        div[data-testid="stMetric"] {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 12px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="main-title">'
    'Oil Reservoir Pressure Distribution'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'Transient Pressure Analysis Using Ei(-x)'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# 3. EXPONENTIAL INTEGRAL Ei(-x)
# =========================================================

def ei_negative(x):
    """
    Calculate Ei(-x) for positive x.

    Uses a convergent series for x < 1 and a continued
    fraction for x >= 1.

    Ei(-x) = -E1(x), for x > 0.
    """

    if not math.isfinite(x) or x <= 0:
        raise ValueError("x must be finite and greater than zero.")

    gamma = 0.5772156649015329
    tolerance = 1e-14
    max_iterations = 1000

    # Series expansion for 0 < x < 1.
    if x < 1.0:
        term = 1.0
        series_sum = 0.0

        for k in range(1, max_iterations + 1):
            term *= -x / k
            addend = term / k
            series_sum += addend

            if abs(addend) < tolerance * max(
                1.0, abs(series_sum)
            ):
                break

        return -gamma - math.log(x) - series_sum

    # Continued fraction for E1(x), using modified Lentz.
    tiny = 1e-300
    b = x + 1.0
    c = 1.0 / tiny
    d = 1.0 / b
    h_cf = d

    for i in range(1, max_iterations + 1):
        a = -float(i * i)
        b += 2.0

        d = a * d + b

        if abs(d) < tiny:
            d = tiny

        c = b + a / c

        if abs(c) < tiny:
            c = tiny

        d = 1.0 / d
        delta = c * d
        h_cf *= delta

        if abs(delta - 1.0) < tolerance:
            break

    e1 = h_cf * math.exp(-x)

    return -e1


# =========================================================
# 4. INPUT VALIDATION
# =========================================================

def is_valid_positive(value):
    return (
        value is not None
        and math.isfinite(float(value))
        and float(value) > 0
    )


def all_finite(values):
    return all(
        value is not None
        and math.isfinite(float(value))
        for value in values
    )


# =========================================================
# 5. INPUT FORM
# =========================================================

st.header("القسم الأول: مدخلات البيانات")

st.caption(
    "أدخل جميع البيانات المطلوبة ثم اضغط زر حساب النتائج. "
    "الحقول الرقمية تبدأ فارغة، ولا تُجرى الحسابات قبل "
    "اكتمال البيانات والتحقق من صحتها."
)
with st.form("reservoir_input_form"):

    # -----------------------------------------------------
    # 5.1 Personal variable
    # -----------------------------------------------------

    st.subheader("1. المتغير الشخصي")

    i_value = st.number_input(
        "المتغير i",
        min_value=1,
        step=1,
        value=None,
        placeholder="أدخل قيمة i",
        key="input_i",
    )

    # -----------------------------------------------------
    # 5.2 Reservoir parameters
    # -----------------------------------------------------

    st.subheader("2. الثوابت البترولية")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        Pi = st.number_input(
            "Pi [psi]",
            min_value=0.0,
            value=None,
            placeholder="الضغط البدائي",
            key="input_Pi",
        )

        K = st.number_input(
            "K [mD]",
            min_value=0.0,
            value=None,
            placeholder="النفوذية",
            key="input_K",
        )

    with c2:
        phi_percent = st.number_input(
            "phi [%]",
            min_value=0.0,
            max_value=100.0,
            value=None,
            placeholder="المسامية %",
            key="input_phi",
        )

        ct = st.number_input(
            "ct [psi⁻¹]",
            min_value=0.0,
            value=None,
            placeholder="الانضغاطية",
            key="input_ct",
        )

    with c3:
        B = st.number_input(
            "B [bbl/STB]",
            min_value=0.0,
            value=None,
            placeholder="عامل حجم النفط",
            key="input_B",
        )

        h = st.number_input(
            "h [ft]",
            min_value=0.0,
            value=None,
            placeholder="ثخانة الطبقة",
            key="input_h",
        )

    with c4:
        mu = st.number_input(
            "mu [cP]",
            min_value=0.0,
            value=None,
            placeholder="لزوجة النفط",
            key="input_mu",
        )

        t = st.number_input(
            "t [hours]",
            min_value=0.0,
            value=None,
            placeholder="الزمن",
            key="input_t",
        )

    # -----------------------------------------------------
    # 5.3 Monitoring point
    # -----------------------------------------------------

    st.subheader("3. نقطة المراقبة M")

    m1, m2, m3 = st.columns([2, 1, 1])

    with m1:
        point_name = st.text_input(
            "اسم نقطة المراقبة",
            value="",
            placeholder="أدخل اسم النقطة",
            key="input_point_name",
        )

        point_symbol = st.text_input(
            "رمز النقطة",
            value="",
            placeholder="مثال: M",
            key="input_point_symbol",
        )

    with m2:
        Mx = st.number_input(
            "Mx [ft]",
            value=None,
            placeholder="إحداثي X",
            key="input_Mx",
        )

    with m3:
        My = st.number_input(
            "My [ft]",
            value=None,
            placeholder="إحداثي Y",
            key="input_My",
        )

    # -----------------------------------------------------
    # 5.4 Wells
    # -----------------------------------------------------

    st.subheader("4. إحداثيات الآبار ومعدلات الإنتاج")

    st.markdown("**البئر الأول**")

    w1c1, w1c2, w1c3 = st.columns(3)

    with w1c1:
        X1 = st.number_input(
            "X1 [ft]",
            value=None,
            placeholder="إحداثي X للبئر الأول",
            key="input_X1",
        )

    with w1c2:
        Y1 = st.number_input(
            "Y1 [ft]",
            value=None,
            placeholder="إحداثي Y للبئر الأول",
            key="input_Y1",
        )

    with w1c3:
        st.text_input(
            "q1 [STB/day]",
            value="يُحسب من i",
            disabled=True,
            key="display_q1",
        )

    st.markdown("**البئر الثاني**")

    w2c1, w2c2, w2c3 = st.columns(3)
    with w2c1:
        X2 = st.number_input(
            "X2 [ft]",
            value=None,
            placeholder="إحداثي X للبئر الثاني",
            key="input_X2",
        )

    with w2c2:
        Y2 = st.number_input(
            "Y2 [ft]",
            value=None,
            placeholder="إحداثي Y للبئر الثاني",
            key="input_Y2",
        )

    with w2c3:
        q2 = st.number_input(
            "q2 [STB/day]",
            min_value=0.0,
            value=None,
            placeholder="معدل إنتاج البئر الثاني",
            key="input_q2",
        )

    st.markdown("**البئر الثالث**")

    w3c1, w3c2, w3c3 = st.columns(3)

    with w3c1:
        X3 = st.number_input(
            "X3 [ft]",
            value=None,
            placeholder="إحداثي X للبئر الثالث",
            key="input_X3",
        )

    with w3c2:
        Y3 = st.number_input(
            "Y3 [ft]",
            value=None,
            placeholder="إحداثي Y للبئر الثالث",
            key="input_Y3",
        )

    with w3c3:
        q3 = st.number_input(
            "q3 [STB/day]",
            min_value=0.0,
            value=None,
            placeholder="معدل إنتاج البئر الثالث",
            key="input_q3",
        )

    submitted = st.form_submit_button(
        "حساب النتائج",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# 6. CALCULATIONS AND RESULTS
# =========================================================

if submitted:

    st.divider()
    st.header("القسم الثاني: النتائج والحسابات النهائية")

    required_values = {
        "i": i_value,
        "Pi": Pi,
        "K": K,
        "phi": phi_percent,
        "ct": ct,
        "B": B,
        "h": h,
        "mu": mu,
        "t": t,
        "Mx": Mx,
        "My": My,
        "X1": X1,
        "Y1": Y1,
        "X2": X2,
        "Y2": Y2,
        "q2": q2,
        "X3": X3,
        "Y3": Y3,
        "q3": q3,
    }

    missing_fields = [
        name
        for name, value in required_values.items()
        if value is None
    ]

    if missing_fields:
        st.error(
            "يرجى تعبئة الحقول التالية: "
            + ", ".join(missing_fields)
        )
        st.stop()

    if not point_name.strip():
        st.error("يرجى إدخال اسم نقطة المراقبة.")
        st.stop()

    if not point_symbol.strip():
        st.error("يرجى إدخال رمز نقطة المراقبة.")
        st.stop()

    positive_parameters = {
        "Pi": Pi,
        "K": K,
        "phi": phi_percent,
        "ct": ct,
        "B": B,
        "h": h,
        "mu": mu,
        "t": t,
    }

    invalid_parameters = [
        name
        for name, value in positive_parameters.items()
        if not is_valid_positive(value)
    ]

    if invalid_parameters:
        st.error(
            "يجب أن تكون القيم أكبر من الصفر للمتغيرات: "
            + ", ".join(invalid_parameters)
        )
        st.stop()

    if not all_finite(
        [
            Mx, My,
            X1, Y1,
            X2, Y2,
            X3, Y3,
            q2, q3,
        ]
    ):
        st.error("توجد قيمة غير صالحة أو غير محدودة.")
        st.stop()

    if q2 < 0 or q3 < 0:
        st.error("معدلات الإنتاج لا يمكن أن تكون سالبة.")
        st.stop()

    # -----------------------------------------------------
    # 6.1 Prepare values
    # -----------------------------------------------------

    phi = float(phi_percent) / 100.0

    q1 = 100.0 + 10.0 * (int(i_value) - 1)

    wells = pd.DataFrame(
        {
            "اسم البئر": [
                "البئر 1",
                "البئر 2",
                "البئر 3",
            ],
            "X [ft]": [
                float(X1),
                float(X2),
                float(X3),
            ],
            "Y [ft]": [
                float(Y1),
                float(Y2),
                float(Y3),
            ],
            "q [STB/day]": [
                q1,
                float(q2),
                float(q3),
            ],
        }
    )
    # -----------------------------------------------------
    # 6.2 Diffusivity coefficient and investigation radius
    # -----------------------------------------------------

    y = (
        0.0002637 * float(K)
        / (phi * float(mu) * float(ct))
    )

    ri = math.sqrt(y * float(t))

    # -----------------------------------------------------
    # 6.3 Real distance and x
    # -----------------------------------------------------

    dx = float(Mx) - wells["X [ft]"].to_numpy()
    dy = float(My) - wells["Y [ft]"].to_numpy()

    r = np.sqrt(dx**2 + dy**2)
    r_squared = r**2

    x = (
        948.0
        * phi
        * float(mu)
        * float(ct)
        * r_squared
        / (float(K) * float(t))
    )

    if np.any(r == 0):
        st.error(
            "نقطة المراقبة تطابق إحداثيات أحد الآبار. "
            "المعادلة تصبح مفردة عند r = 0. "
            "يجب استخدام نصف قطر بئر فعلي أو تغيير الإحداثيات."
        )
        st.stop()

    if np.any(x <= 0) or not np.all(np.isfinite(x)):
        st.error("تعذر حساب x بسبب قيم عددية غير صالحة.")
        st.stop()

    # -----------------------------------------------------
    # 6.4 Exponential integral
    # -----------------------------------------------------

    ei_values = np.array(
        [ei_negative(float(value)) for value in x]
    )

    # -----------------------------------------------------
    # 6.5 Pressure drawdown
    # -----------------------------------------------------

    # Ei(-x) is negative for x > 0.
    # Positive drawdown magnitude is therefore -C * Ei(-x).

    coefficient = (
        70.6
        * wells["q [STB/day]"].to_numpy()
        * float(mu)
        * float(B)
        / (float(K) * float(h))
    )

    delta_p = -coefficient * ei_values

    delta_p_total = float(np.sum(delta_p))
    P_M = float(Pi) + delta_p_total

    # =====================================================
    # 7. GENERAL RESULTS TABLE
    # =====================================================

    st.subheader("الجدول الأول: النتائج العامة للآبار")

    general_results = pd.DataFrame(
        {
            "اسم البئر": wells["اسم البئر"],
            "معدل الإنتاج q [STB/day]":
                wells["q [STB/day]"],
            "معامل الانتشار y [ft²/hour]":
                [y] * len(wells),
            "المسافة المؤثرة ri [ft]":
                [ri] * len(wells),
        }
    )

    st.dataframe(
        general_results.style.format(
            {
                "معدل الإنتاج q [STB/day]": "{:,.3f}",
                "معامل الانتشار y [ft²/hour]": "{:,.5f}",
                "المسافة المؤثرة ri [ft]": "{:,.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # =====================================================
    # 8. DETAILED RESULTS TABLE
    # =====================================================

    st.subheader("الجدول الثاني: الحسابات التفصيلية")

    detailed_results = pd.DataFrame(
        {
            "اسم البئر": wells["اسم البئر"],
            "r [ft]": r,
            "r² [ft²]": r_squared,
            "x [-]": x,
            "Ei(-x)": ei_values,
            "q [STB/day]": wells["q [STB/day]"],
            "Delta P [psi]": delta_p,
        }
    )

    st.dataframe(
        detailed_results.style.format(
            {
                "r [ft]": "{:,.4f}",
                "r² [ft²]": "{:,.4f}",
                "x [-]": "{:.8g}",
                "Ei(-x)": "{:.8f}",
                "q [STB/day]": "{:,.3f}",
                "Delta P [psi]": "{:,.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # =====================================================
    # 9. TOTAL DRAWDOWN
    # =====================================================

    st.subheader("مجموع هبوطات الضغط")

    st.latex(
        r"\Delta P_{\mathrm{total}}"
        r"=\sum_{j=1}^{3}\Delta P_j"
    )

    # استخدام st.markdown مع تصميم مخصص لتجنب المربع الأبيض وجعل النص أسود واضح
    st.markdown(
        f"""
        <div style="
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin: 10px 0;
        ">
            <span style="font-size: 16px; color: #334155; font-weight: bold;">Delta P total [psi]</span><br>
            <span style="font-size: 28px; color: #000000; font-weight: bold;">{delta_p_total:,.4f}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =====================================================
    # 10. FINAL PRESSURE
    # =====================================================

    st.subheader("النتيجة النهائية عند نقطة المراقبة")

    st.latex(
        r"P(M)=P_i-\Delta P_{\mathrm{total}}"
    )

    st.success(
        f"النقطة: {point_name.strip()} "
        f"({point_symbol.strip()})\n\n"
        f"الضغط الآني P(M) = {P_M:,.4f} psi"
    )

    if delta_p_total < 0 and P_M < float(Pi):
        st.caption(
            "تحقق الشرط: الضغط المحسوب أقل من الضغط البدائي."
        )
    elif delta_p_total == 0:
        st.warning(
            "مجموع هبوط الضغط يساوي صفرًا، "
            "لذلك لم يحدث هبوط محسوب في هذه الحالة."
        )
    else:
        st.warning(
            "الضغط المحسوب ليس أقل من الضغط البدائي. "
            "تحقق من المدخلات والمعادلات والوحدات."
        )

    # =====================================================
    # 11. SMALL WELL LOCATION MAP WITH DISTANCES
    # =====================================================

    st.subheader("خريطة الآبار ونقطة المراقبة")

    fig, ax = plt.subplots(figsize=(7, 4.8))

    well_x = wells["X [ft]"].to_numpy()
    well_y = wells["Y [ft]"].to_numpy()

    # Plot production wells.
    ax.scatter(
        well_x,
        well_y,
        s=75,
        marker="o",
        label="Production wells",
        zorder=3,
    )

    # Plot monitoring point.
    ax.scatter(
        [float(Mx)],
        [float(My)],
        s=180,
        marker="*",
        label="Monitoring point",
        zorder=4,
    )

    # Plot labels, connections, and numerical distances.
    for j, row in wells.iterrows():

        wx = float(row["X [ft]"])
        wy = float(row["Y [ft]"])
        distance = float(r[j])

        ax.annotate(
            str(row["اسم البئر"]),
            (wx, wy),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
        )

        # Connect each well to M.
        ax.plot(
            [wx, float(Mx)],
            [wy, float(My)],
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
        )

        # Put distance label near the midpoint.
        mid_x = (wx + float(Mx)) / 2.0
        mid_y = (wy + float(My)) / 2.0

        ax.annotate(
            f"r{j + 1} = {distance:.2f} ft",
            (mid_x, mid_y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "alpha": 0.85,
            },
        )

    ax.annotate(
        point_symbol.strip(),
        (float(Mx), float(My)),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlabel("X [ft]", fontsize=9)
    ax.set_ylabel("Y [ft]", fontsize=9)
    ax.set_title(
        "Well Locations and Distances to M",
        fontsize=11,
    )

    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()

    st.pyplot(fig, use_container_width=False)

    plt.close(fig)

    # =====================================================
    # 12. EXPORT
    # =====================================================

    st.subheader("تصدير النتائج")

    export_data = detailed_results.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="تحميل النتائج بصيغة CSV",
        data=export_data,
        file_name="reservoir_pressure_results.csv",
        mime="text/csv",
    )

else:
    st.info(
        "أدخل البيانات في القسم الأول، ثم اضغط "
        "«حساب النتائج» لإظهار الجداول والحسابات والرسم."
    )
