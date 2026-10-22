import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from asmt_core import ASMT_Engine

# ==========================================
# 页面配置与初始化
# ==========================================
st.set_page_config(page_title="ASMT Emergency Response Tool", layout="wide")
engine = ASMT_Engine()

st.title("🌊 Ammonia Spill Modeling Tool (ASMT)")
st.markdown("""
**An integrated Lagrangian framework for screening-level consequence assessment.**  
Coupling dynamic pH speciation, two-film volatilization, and Gaussian puff/plume dispersion.
""")

# ==========================================
# 侧边栏：场景控制
# ==========================================
st.sidebar.header("Scenario Configuration")
mode = st.sidebar.radio("Select Analysis Mode:", 
                       ("Custom Emergency Response (Universal Tool)",
                        "RS-1G Field Evaluation", 
                        "Coastal Bunkering (Case 3)"))

# =========================================================================
# 模式 1: 自定义应急响应 (导师要求的通用工具)
# =========================================================================
if mode == "Custom Emergency Response (Universal Tool)":
    st.header("🚨 Custom Emergency Response Tool")
    st.markdown("Fully customizable ASMT engine for operational emergency response. Enter spill parameters below to generate immediate mass budget and hazard assessments.")
    
    # --- 1. 精确输入区 ---
    st.subheader("1. Input Parameters")
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        M_total = st.number_input("Total Released Mass (kg)", min_value=1.0, value=1000.0, step=100.0)
        storage = st.selectbox("Storage Condition", ["Refrigerated (-33.3 °C)", "Semi-refrigerated (-10 °C)", "Pressurized (15 °C)"])
        dist_m = st.number_input("Receptor Distance (m)", min_value=10.0, value=100.0, step=10.0)
        
    with col_in2:
        U10 = st.number_input("Wind Speed (m/s)", min_value=0.1, max_value=30.0, value=5.0, step=0.1)
        T_C = st.number_input("Water/Air Temp (°C)", min_value=-2.0, max_value=40.0, value=15.0, step=0.5)
        pH_bg = st.number_input("Background pH", min_value=6.0, max_value=9.0, value=8.1, step=0.1)
        
    with col_in3:
        S = st.number_input("Salinity (PSU)", min_value=0.0, max_value=40.0, value=35.0, step=1.0)
        X = st.number_input("Wind Fetch (m)", min_value=1.0, value=10000.0, step=100.0)
        sim_days = st.number_input("Simulation Time (Days)", min_value=0.1, value=5.0, step=0.5)

    # --- 核心计算 ---
    if "Pressurized" in storage: T_s = 15.0
    elif "Semi" in storage: T_s = -10.0
    else: T_s = -33.3
    
    phi_total, M_flash, M_diss_initial = engine.calc_source_partition(M_total, T_s)
    
    res = engine.run_fate_simulation(
        M_dissolved_0=M_diss_initial, sim_time_hours=sim_days * 24.0, U10=U10, 
        T_C=T_C, S=S, pH_input=pH_bg, X=X, k20=0.01, dynamic_pH=True
    )
    
    vol_mass = res['volatilized'][-1]
    nit_mass = res['nitrified'][-1]
    res_mass = res['residual'][-1]
    
    puff_ppm = engine.calc_gaussian_puff(Mass_kg=M_flash, U_wind=U10, x=dist_m, y=0, z=2, t_sec=dist_m/U10)
    Q_plume_kg_s = vol_mass / (sim_days * 24 * 3600)
    plume_ppm = engine.calc_gaussian_plume(Q_kg_s=Q_plume_kg_s, U_wind=U10, x=dist_m, y=0, z=2)

    st.markdown("---")
    # --- 2. 图形展示区 ---
    st.subheader("2. Graphical Results")
    
    # 第一排：饼图和柱状图 (端点静态数据)
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        labels = ['Flash (Initial Air)', 'Volatilized (Delayed Air)', 'Nitrified (Aquatic Sink)', 'Residual Dissolved']
        sizes = [M_flash, vol_mass, nit_mass, res_mass]
        colors = ['#f39c12', '#f1c40f', '#27ae60', '#3498db']
        
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax1.axis('equal')
        ax1.set_title(f"Mass Budget ({sim_days} Days)")
        st.pyplot(fig1)

    with chart_col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(["Initial Flash Puff", "Prolonged Plume"], [puff_ppm, plume_ppm], color=['#e67e22', '#f1c40f'])
        ax2.axhline(160, color='red', linestyle=':', label='AEGL-2 (160 ppm)')
        ax2.axhline(1100, color='darkred', linestyle='--', label='AEGL-3 (1100 ppm)')
        ax2.set_ylabel(f"NH3 Conc. at {dist_m}m (ppm)")
        ax2.set_yscale('symlog')
        ax2.set_title(f"Dual-Hazard Airborne Exposure")
        ax2.legend()
        st.pyplot(fig2)

    # 第二排：时间演化曲线与空间衰减曲线 (动态趋势数据)
    chart_col3, chart_col4 = st.columns(2)
    
    with chart_col3:
        # 质量随时间的演化
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        days_array = res['time_hr'] / 24.0
        ax3.plot(days_array, res['residual'], label='Residual Dissolved', color='#3498db', linewidth=2)
        ax3.plot(days_array, res['volatilized'], label='Volatilized (Cumulative)', color='#f1c40f', linewidth=2)
        ax3.plot(days_array, res['nitrified'], label='Nitrified (Cumulative)', color='#27ae60', linewidth=2)
        ax3.axhline(M_flash, label='Instant Flash', color='#f39c12', linestyle='--', linewidth=2)
        ax3.set_xlabel('Time (Days)')
        ax3.set_ylabel('Mass (kg)')
        ax3.set_title('Mass Evolution over Time')
        ax3.legend()
        ax3.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig3)

    with chart_col4:
        # 浓度随距离的衰减
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        # 生成 10m 到 目标距离两倍 的空间点
        x_max = max(1000, int(dist_m * 2))
        x_vals = np.linspace(10, x_max, 100)
        plume_vals = [engine.calc_gaussian_plume(Q_kg_s=Q_plume_kg_s, U_wind=U10, x=xv, y=0, z=2) for xv in x_vals]
        
        ax4.plot(x_vals, plume_vals, label='Prolonged Plume (z=2m)', color='#f1c40f', linewidth=2)
        ax4.axhline(160, color='red', linestyle=':', label='AEGL-2 (160 ppm)')
        ax4.axhline(1100, color='darkred', linestyle='--', label='AEGL-3 (1100 ppm)')
        ax4.axvline(dist_m, color='grey', linestyle='-.', label=f'Target ({dist_m}m)')
        ax4.set_xlabel('Downwind Distance (m)')
        ax4.set_ylabel('NH3 Concentration (ppm)')
        ax4.set_yscale('symlog')
        ax4.set_title('Concentration Decay vs. Distance')
        ax4.legend()
        ax4.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig4)

    st.markdown("---")
    # --- 3. 详细数值区 ---
    st.subheader("3. Detailed Numerical Output")
    num_col1, num_col2 = st.columns(2)
    
    with num_col1:
        st.info(f"**📊 Aquatic Mass Partitioning**\n\n"
                f"- **Total Released:** {M_total:.1f} kg\n"
                f"- **Instant Flash to Air:** {M_flash:.1f} kg ({(M_flash/M_total)*100:.1f}%)\n"
                f"- **Volatilized to Air:** {vol_mass:.1f} kg ({(vol_mass/M_total)*100:.1f}%)\n"
                f"- **Nitrified (Degraded):** {nit_mass:.1f} kg ({(nit_mass/M_total)*100:.1f}%)\n"
                f"- **Residual in Water:** {res_mass:.1f} kg ({(res_mass/M_total)*100:.1f}%)")
        
    with num_col2:
        st.warning(f"**☠️ Airborne Exposure at {dist_m}m**\n\n"
                   f"- **Instantaneous Puff Peak:** {puff_ppm:.1f} ppm\n"
                   f"  *(Note: AEGL-3 Lethality Threshold is 1100 ppm)*\n\n"
                   f"- **Prolonged Plume Average:** {plume_ppm:.1f} ppm\n"
                   f"  *(Note: AEGL-2 Irreversible Effect is 160 ppm)*")

# =========================================================================
# 模式 2 & 3: 审稿人验证模型
# =========================================================================
elif mode == "RS-1G Field Evaluation":
    st.header("📊 RS-1G Field Test: Wind Speed Correction")
    st.markdown("Evaluating the impact of corrected first-hour mean wind speed (8.0 m/s) vs. puff-period wind speed (5.8 m/s).")
    col1, col2 = st.columns([1, 2])
    with col1:
        wind_speed = st.number_input("First-Hour Wind Speed (m/s)", min_value=3.0, max_value=12.0, value=8.0, step=0.1)
        temp_c = st.number_input("Air/Water Temperature (°C)", value=13.0)
        ph_val = st.number_input("Pool pH (Constant)", value=10.5)
        
        res = engine.run_fate_simulation(
            M_dissolved_0=78.0, sim_time_hours=1.0, U10=wind_speed, T_C=temp_c, 
            S=0.0, pH_input=ph_val, X=4.0, pool_depth=0.0625, dynamic_pH=False)
        
        volatilized = res['volatilized'][-1]
        error = ((volatilized - 20.0) / 20.0) * 100
        
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Observed (Field)", "ASMT Predicted"], [20.0, volatilized], color=['#2c3e50', '#e74c3c'])
        ax.axhline(20.0, color='grey', linestyle='--')
        ax.set_ylabel("Volatilized Mass (kg)")
        ax.set_ylim(0, 30)
        ax.text(1, volatilized + 1, f"Error: {error:+.1f}%", ha='center', fontsize=12, fontweight='bold', color='green' if abs(error) < 10 else 'red')
        st.pyplot(fig)

elif mode == "Coastal Bunkering (Case 3)":
    st.header("🚢 Coastal Bunkering Spills (5-Day Mass Budget)")
    col1, col2, col3 = st.columns(3)
    with col1:
        scenario = st.selectbox("Select Scenario:", ["S1 (Summer, 5 m/s)", "S3 (Pressurized, 5 m/s)", "S4 (Summer Calm, 2 m/s)", "S5 (Winter, 5 m/s)"])
    with col2:
        use_dynamic_ph = st.toggle("Enable Dynamic Alkalinity pH", value=True)
    with col3:
        dist_m = st.number_input("Downwind Receptor Distance (m)", min_value=10, max_value=1000, value=100, step=10)

    params = {"M_diss": 700.0, "U10": 5.0, "T_C": 15.0, "S": 35.0, "phi": 0.3}
    if "S3" in scenario: params.update({"M_diss": 600.0, "phi": 0.4})
    if "S4" in scenario: params["U10"] = 2.0
    if "S5" in scenario: params["T_C"] = 5.0
        
    res = engine.run_fate_simulation(
        M_dissolved_0=params["M_diss"], sim_time_hours=120.0, U10=params["U10"], 
        T_C=params["T_C"], S=params["S"], pH_input=8.1, X=10000.0, k20=0.01, dynamic_pH=use_dynamic_ph)
    
    M_total = params["M_diss"] / (1 - params["phi"])
    flash_mass = M_total * params["phi"]
    vol_mass = res['volatilized'][-1]
    nit_mass = res['nitrified'][-1]
    res_mass = res['residual'][-1]
    
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.pie([flash_mass, vol_mass, nit_mass, res_mass], labels=['Flash', 'Volatilized', 'Nitrified', 'Residual'], autopct='%1.1f%%', startangle=90, colors=['#f39c12', '#f1c40f', '#27ae60', '#3498db'])
        ax1.axis('equal')
        st.pyplot(fig1)

    with chart_col2:
        puff_ppm = engine.calc_gaussian_puff(Mass_kg=flash_mass, U_wind=params["U10"], x=dist_m, y=0, z=2, t_sec=dist_m/params["U10"])
        plume_ppm = engine.calc_gaussian_plume(Q_kg_s=vol_mass/(120*3600), U_wind=params["U10"], x=dist_m, y=0, z=2)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(["Puff", "Plume"], [puff_ppm, plume_ppm], color=['#e67e22', '#f1c40f'])
        ax2.axhline(160, color='red', linestyle=':')
        ax2.axhline(1100, color='darkred', linestyle='--')
        ax2.set_ylabel(f"NH3 Conc. at {dist_m}m (ppm)")
        ax2.set_yscale('symlog')
        st.pyplot(fig2)