"""
=======================================================================================
Ammonia Spill Modeling Tool (ASMT) - Core Computational Engine
=======================================================================================

【一、 核心控制方程与物理机理 (Governing Equations)】
本引擎严格基于 Zhaoyang 手稿 (File 2) 中的 13 个核心方程构建：

1. 初始源项与闪蒸分配 (Source Term & Flashing) [Eq. 10]
   - ϕ_fl = 1 - exp(-c_{p,l} * (T_s - T_b) / h_{fg})
   - ϕ = ϕ_fl + (1 - ϕ_fl) * ϕ_w
   - 机制：冷冻液氨泄漏时发生绝热闪蒸 (ϕ_fl)，接触水面后发生沸腾汽化 (ϕ_w)。

2. 氨酸碱形态分布 (Speciation) [Eq. 5]
   - f = 1 / (1 + 10^(pKa - pH))
   - pKa = -0.467 + 0.00113 * S + 2887.9 / T_K
   - 机制：游离态氨 (NH3) 比例 f 受温度 (T)、盐度 (S) 和 pH 严格控制。

3. 双膜理论水面挥发 (Two-Film Volatilization) [Eq. 3, 6-8]
   - dm/dt_vol = - K * f * m / d
   - 1/K = 1/k_L + 1/(K_H * k_G)
   - k_L: 液相传质系数 (由摩擦流速 u* 和 Schmidt 数 Sc_L 决定，临界风区 X=16m)
   - k_G: 气相传质系数 (由 u*, 气体 Sc_G, 雷诺数 Re* 决定)
   - 机制：溶解态氨克服水膜阻力和气膜阻力向大气逃逸。挥发深度 d 受限于波高和湍流扩散。

4. 硝化降解动力学 (Nitrification) [Eq. 9]
   - dm/dt_nit = - k_bio * m
   - k_bio = k_20 * θ^(T_w - 20)
   - 机制：微生物将氨氧化，受一阶衰减动力学及 Arrhenius 温度修正控制。

5. 大气高斯扩散 (Gaussian Dispersion) [Eq. 11-13]
   - Puff (烟团): C_puff = (ϕ*M / (2π^1.5 * σx*σy*σz)) * exp(...)
   - Plume (烟羽): C_plume = (Q / (2π*u*σy*σz)) * exp(...)
   - 机制：Puff 用于计算泄漏瞬间闪蒸带来的急性毒性峰值；Plume 用于计算水面持续挥发带来的长期暴露浓度。

【二、 全局变量与参数范围考量 (Parameters & Justifications)】
依据手稿 Table 2, Table 3 及正文描述设定：

* M (Released Mass): 泄漏总质量 (kg)。
* T_s (Storage Temp): 储存温度 (°C)。冷冻储罐为 -33.3°C (ϕ_fl=0)；加压储罐常温 15°C (引发剧烈闪蒸)。
* U_10 (Wind Speed): 10米高风速 (0.1 ~ 25 m/s)。涵盖静风到风暴(蒲福风级10级)，直接决定 u* 和波高。
* T_w (Water Temp): 水温 (1 ~ 35 °C)。涵盖极地到热带海域，影响 pKa, K_H 和 硝化速率。
* S (Salinity): 盐度 (0 ~ 37 PSU)。淡水为0，海水典型值为35。微弱影响 pKa。
* pH: 系统酸碱度 (7.0 ~ 11.0)。自然水体通常为 7.0-8.4，但在氨泄漏核心区 (Spill Core)，氨的强碱性溶解会驱使 pH 飙升至 10.5-11.0 (ITOPF, 2024)。
* X (Fetch): 风区长度 (1 ~ 100 m)。Red Squirrel 实验为 4m (受限风区)，开阔海域为 unlimited。
* k_20: 20°C 基础硝化速率 (0 ~ 0.01 h^-1)。0代表高浓度游离氨抑制了微生物(近场)；0.01代表沿海最高代谢率(远场)。
* θ (Theta): 温度修正系数 (1.02 ~ 1.14)。废水处理标准经验值 (Metcalf & Eddy, 2014)。
* Sc_L, Sc_G: 施密特数。气体恒定 0.6；液体默认 360 (25°C)，受温度影响。
=======================================================================================
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

class ASMT_Engine:
    def __init__(self):
        # 物理常数与属性 (Constants)
        self.R = 8.314          # 理想气体常数 J/(mol K)
        self.M_NH3 = 17.03      # 氨气摩尔质量 g/mol
        self.T_b = -33.3        # 氨标准沸点 °C
        self.c_pl = 4.6         # 液氨比热容 kJ/(kg K)
        self.h_fg = 1370.0      # 氨汽化潜热 kJ/kg
        self.Sc_G = 0.6         # 气相施密特数 (恒定)
        self.Sc_L = 360.0       # 液相施密特数 (默认25°C)
        self.g = 9.81           # 重力加速度 m/s^2
        # 【修复】标准亨利常数 (Sander 2023): 59 mol/(L atm) -> 转换至 mol/(m3 Pa) 需要 * 1000
        self.H_cp_std = (59.0 * 1000.0) / 101325.0
    # -------------------------------------------------------------------------
    # 模块 1: 初始闪蒸与源项分配 (Source Term Partitioning)
    # -------------------------------------------------------------------------
    def calc_source_partition(self, M_total, T_s, phi_w=0.30):
        """
        计算绝热闪蒸与接触水面汽化分配比例 (Eq. 10)
        M_total: 总泄漏质量 (kg)
        T_s: 储罐温度 (°C)
        phi_w: 接触水面沸腾比例 (默认 0.30, Raj and Reid 1978)
        返回: (phi_total, M_flash, M_dissolved)
        """
        if T_s <= self.T_b:
            phi_fl = 0.0  # 完全冷冻状态无绝热闪蒸
        else:
            # 绝热闪蒸比例公式
            phi_fl = 1.0 - math.exp(-self.c_pl * (T_s - self.T_b) / self.h_fg)
        
        # 总体逃逸到大气的比例
        phi_total = phi_fl + (1.0 - phi_fl) * phi_w
        
        M_flash = M_total * phi_total
        M_dissolved = M_total * (1.0 - phi_total)
        return phi_total, M_flash, M_dissolved

    # -------------------------------------------------------------------------
    # 模块 2: 水生化学与热力学 (Aquatic Chemistry & Thermodynamics)
    # -------------------------------------------------------------------------
    def calc_pKa(self, T_K, S):
        """计算解离常数 pKa (Bell et al., 2008)"""
        return -0.467 + 0.00113 * S + 2887.9 / T_K

    def calc_unionized_fraction(self, pH, T_K, S):
        """计算游离态氨比例 f (Eq. 5)"""
        pKa = self.calc_pKa(T_K, S)
        return 1.0 / (1.0 + 10**(pKa - pH))

    def dynamic_ph_solver(self, TAN_mgL, is_seawater=True, T_C=15.0):
        """
        动态 pH 求解器 (基于 Supp Fig S1b 的碱度守恒近似)
        此方法用于响应 Reviewer 3 的要求：稀释过程中 pH 的动态下降。
        TAN_mgL: 总氨氮浓度 mg/L
        """
        if TAN_mgL <= 0:
            return 8.1 if is_seawater else 7.5 # 背景 pH

        T_K = T_C + 273.15
        S = 35.0 if is_seawater else 0.0
        pKa = self.calc_pKa(T_K, S)
        
        # 初始海洋碱度参数 (mmol/kg)
        TA = 2.3 if is_seawater else 1.0
        
        TAN_mM = TAN_mgL / 17.03 # 近似转换 mmol/L (这里等效 mmol/kg)
        
        # 定义碱度守恒的误差函数 (简化版，仅考虑水、氨和平均碳酸盐缓冲)
        def alk_balance(pH_guess):
            f_nh3 = 1.0 / (1.0 + 10**(pKa - pH_guess))
            NH3_alk = TAN_mM * f_nh3
            # 极度简化的碳酸缓冲抵抗 (仅作趋势演示以满足模型动态计算要求)
            buffer_resist = (pH_guess - (8.1 if is_seawater else 7.5)) * (TA / 2.0)
            return NH3_alk - buffer_resist

        # 使用 Brentq 方法求解 7.0 到 12.0 之间的 pH 根
        try:
            pH_sol = brentq(alk_balance, 7.0, 11.5)
            return pH_sol
        except ValueError:
            return 11.0 # 如果极高浓度导致无根，强制返回强碱极值

    # -------------------------------------------------------------------------
    # 模块 3: 双膜理论挥发动力学 (Two-Film Volatilization Transfer)
    # -------------------------------------------------------------------------

    def calc_transfer_coefficients(self, U10, X, T_K):
        """
        计算整体气液传质系数 K (m/s) 以及辅助参数 (Eq. 6-8)
        """
        T_C = T_K - 273.15
        
        # 1. 计算摩擦流速 u* (Prata et al., 2017)
        u_star = 1e-4 * (333.9 * U10 + 4.356 * U10**2)
        
        # 2. 空气运动黏度 ν
        nu_air = 1e-6 * (9e-4 * T_C**2 + 0.0911 * T_C + 13.28)
        
        # 3. 无量纲参数
        X_star = (X * self.g) / (u_star**2)
        Re_star = (u_star * X) / nu_air
        
        # 4. 液相传质系数 k_L (Eq. 7)
        if X < 16.0:
            k_L = 4.31e-3 * u_star * (self.Sc_L**-0.5)
        else:
            k_L = (1.191 + 2.551 * math.log10(X)) * 1e-3 * u_star * (self.Sc_L**-0.5)
            
        # 5. 气相传质系数 k_G (Eq. 8)
        k_G = 0.0994 * (Re_star**-0.0894) * (X_star**0.0178) * (self.Sc_G**(-2/3)) * u_star
        
        # 6. Henry 定律常数计算 (Sander, 2023)
        # 【核心修复：59 mol/(L atm) -> 转换至 mol/(m3 Pa) 需要 * 1000.0】
        H_cp_std = (59.0 * 1000.0) / 101325.0  
        H_cp = H_cp_std * math.exp(4200.0 * (1.0/T_K - 1.0/298.15))
        K_H = 1.0 / (H_cp * self.R * T_K) # 无量纲 Henry 常数
        
        # 7. 整体传质阻力 (Eq. 6)
        K_overall = 1.0 / ((1.0 / k_L) + (1.0 / (K_H * k_G)))
        
        return K_overall, u_star

    def calc_diffusive_depth(self, U10, X, delta_t_sec, force_depth=None):
        """计算挥发混合深度 d (m)"""
        if force_depth is not None:
            return force_depth # 用于 Red Squirrel 水池实验的固定深度 0.0625m

        # 波高计算 (USACE, 1984) Eq. 4
        tau = 0.71 * U10**1.23
        # 简单判定 fetch-limited (这里设定如果 fetch < 10km 视为受限)
        if X < 10000:
            H_s = 5.112e-4 * tau * math.sqrt(X)
        else:
            H_s = 0.0248 * tau**2
        H_w = 1.27 * H_s
        
        # 垂直扩散 Dz
        D_z = 0.0009 * U10
        
        # d = min(sqrt(2 * Dz * t), 1.5 * Hw)
        # 注意: 审稿人1提出此处 t 应该是时间累计，而不是数值时间步。这里直接用累计时间。
        return min(math.sqrt(2.0 * D_z * delta_t_sec), 1.5 * H_w)

    # -------------------------------------------------------------------------
    # 模块 4: ODE 命运演化积分 (Fate Evolution Solver)
    # -------------------------------------------------------------------------
    def run_fate_simulation(self, M_dissolved_0, sim_time_hours, U10, T_C, S, pH_input, X, k20=0.005, theta=1.08, pool_depth=None, dynamic_pH=False):
        """
        使用 scipy BDF 隐式积分求解质量衰减。
        y[0]: Residual Dissolved Mass
        y[1]: Cumulative Volatilized Mass
        y[2]: Cumulative Nitrified Mass
        """
        T_K = T_C + 273.15
        K_overall, _ = self.calc_transfer_coefficients(U10, X, T_K)
        k_bio = k20 * (theta ** (T_C - 20.0)) # 硝化速率 (h^-1) 转成 s^-1
        k_bio_sec = k_bio / 3600.0

        def odes(t_sec, y):
            m_res = max(y[0], 0.0) # 剩余溶解量
            
            # 为了计算动态 pH，需要估算局部水体体积，以求得局部浓度 (mg/L)
            # 这是一个近似：假设水体横向扩散面积为 A_approx (例如 1000m2)，乘以深度 d
            t_for_depth = max(t_sec, 1.0)
            d = self.calc_diffusive_depth(U10, X, t_for_depth, force_depth=pool_depth)
            
            current_pH = pH_input
            if dynamic_pH:
                # 近似局部体积以获得浓度 (用于驱动 pH 变化)
                vol_L = (1000.0 * d) * 1000.0 if pool_depth is None else 1000.0
                tan_mgL = (m_res * 1e6) / vol_L
                current_pH = self.dynamic_ph_solver(tan_mgL, is_seawater=(S>0), T_C=T_C)

            f_nh3 = self.calc_unionized_fraction(current_pH, T_K, S)
            
            # 挥发通量方程 (Eq. 3)
            dm_dt_vol = (K_overall * f_nh3 * m_res) / d
            
            # 硝化方程 (Eq. 9)
            dm_dt_nit = k_bio_sec * m_res
            
            # 主守恒方程
            dm_dt_res = - dm_dt_vol - dm_dt_nit
            
            return [dm_dt_res, dm_dt_vol, dm_dt_nit]

        # 设置积分时间
        t_span = (0, sim_time_hours * 3600.0)
        t_eval = np.linspace(0, sim_time_hours * 3600.0, 200) # 提取 200 个时间点
        y0 = [M_dissolved_0, 0.0, 0.0]

        # 隐式 BDF 求解 (应对僵硬方程)
        sol = solve_ivp(odes, t_span, y0, method='BDF', t_eval=t_eval, rtol=1e-5, atol=1e-7)
        
        return {
            'time_hr': sol.t / 3600.0,
            'residual': sol.y[0],
            'volatilized': sol.y[1],
            'nitrified': sol.y[2]
        }

    # -------------------------------------------------------------------------
    # 模块 5: 大气高斯扩散模型 (Atmospheric Dispersion)
    # -------------------------------------------------------------------------
    def get_briggs_sigmas(self, distance, stability_class='D'):
        """Briggs (1973) 乡村/开阔水域扩散参数 (横向 σy, 垂直 σz)"""
        x = max(distance, 1.0)
        if stability_class == 'D':
            sy = 0.08 * x / math.sqrt(1 + 0.0001 * x)
            sz = 0.06 * x / math.sqrt(1 + 0.0015 * x)
        elif stability_class == 'F':
            sy = 0.04 * x / math.sqrt(1 + 0.0001 * x)
            sz = 0.016 * x / math.sqrt(1 + 0.0003 * x)
        else: # 默认 D
            sy = 0.08 * x / math.sqrt(1 + 0.0001 * x)
            sz = 0.06 * x / math.sqrt(1 + 0.0015 * x)
        return sy, sz

    def calc_gaussian_plume(self, Q_kg_s, U_wind, x, y, z, H=0.0, stab='D', T_C=15.0, sy_0=0.93, sz_0=1.15):
        """
        稳态连续泄漏烟羽 (Eq. 12 & 13)
        加入 sy_0 和 sz_0 修正近场(10m)盲区。
        4x4m 水池的初始水平扩散 sy_0 ≈ 4/4.3 = 0.93m。
        挡水墙和湍流带来的初始垂直扩散 sz_0 取经验值 1.15m 以对齐实验。
        """
        if x <= 0: return 0.0
        sy, sz = self.get_briggs_sigmas(x, stab)
        
        # 加入初始源尺寸修正 (Virtual Source)
        sy = math.sqrt(sy**2 + sy_0**2)
        sz = math.sqrt(sz**2 + sz_0**2)
        
        G_z = math.exp(-((z - H)**2) / (2*sz**2)) + math.exp(-((z + H)**2) / (2*sz**2))
        C_kg_m3 = (Q_kg_s / (2 * math.pi * U_wind * sy * sz)) * math.exp(-(y**2) / (2*sy**2)) * G_z
        
        # 转换为 ppm
        P_Pa = 101325.0
        C_ppm = 1e9 * C_kg_m3 * self.R * (T_C + 273.15) / (P_Pa * self.M_NH3)
        return C_ppm

    def calc_gaussian_puff(self, Mass_kg, U_wind, x, y, z, t_sec, H=0.0, stab='D', T_C=15.0):
        """
        瞬间泄漏烟团 (Eq. 11 & 13) - 模拟接触瞬间闪蒸爆破
        """
        if t_sec <= 0: return 0.0
        sy, sz = self.get_briggs_sigmas(x, stab)
        sx = sy # Puff 假设纵向扩散等同于横向
        
        G_z = math.exp(-((z - H)**2) / (2*sz**2)) + math.exp(-((z + H)**2) / (2*sz**2))
        term_x = math.exp(-((x - U_wind * t_sec)**2) / (2*sx**2))
        term_y = math.exp(-(y**2) / (2*sy**2))
        
        C_kg_m3 = (Mass_kg / ((2 * math.pi)**1.5 * sx * sy * sz)) * term_x * term_y * G_z
        
        P_Pa = 101325.0
        C_ppm = 1e9 * C_kg_m3 * self.R * (T_C + 273.15) / (P_Pa * self.M_NH3)
        return C_ppm
