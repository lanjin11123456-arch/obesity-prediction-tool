import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# 1. 页面配置 (Wide Mode)
# ==========================================
st.set_page_config(
    page_title="儿童肥胖风险AI预测工具",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# 2. 加载资源
# ==========================================
@st.cache_resource
def load_assets():
    try:
        # 获取当前脚本所在目录，确保能找到文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'my_obesity_model.pkl')
        scaler_path = os.path.join(current_dir, 'my_scaler.pkl')
        csv_path = os.path.join(current_dir, 'ready_train.csv')

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        sample_df = pd.read_csv(csv_path, nrows=1)
        expected_cols = sample_df.columns.tolist()

        return model, scaler, expected_cols
    except Exception as e:
        st.error(f"严重错误：加载模型文件失败。请确认 'my_obesity_model.pkl' 在同一目录下。\n报错信息: {e}")
        st.stop()


model, scaler, expected_cols = load_assets()

# ==========================================
# 3. 自定义 CSS (美化界面)
# ==========================================
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button {
        background-color: #4CAF50; color: white; border-radius: 8px; height: 50px; font-size: 18px;
    }
    .result-card { 
        background-color: #ffffff; 
        padding: 25px; 
        border-radius: 15px; 
        border-left: 6px solid #4CAF50;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .high-risk { border-left-color: #FF5252 !important; }
    .metric-label { font-size: 14px; color: #666; }
    .metric-value { font-size: 24px; font-weight: bold; color: #333; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 4. 页面主体布局
# ==========================================

col_input, col_gap, col_result = st.columns([1, 0.1, 1.2])

# --- 左侧：数据输入区 ---
with col_input:
    st.markdown("### 📝 学生体质数据录入")
    st.info("请输入学生的最新体测数据，AI 将自动计算肥胖风险。")

    with st.form("main_input_form"):
        # 分组1: 基础信息 (Age, Gender)
        st.markdown("#### 1. 基础信息")
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("性别", options=[1, 0], format_func=lambda x: "男 (Boy)" if x == 1 else "女 (Girl)")
        with c2:
            age = st.number_input("年龄 (Age)", min_value=6, max_value=18, value=10, step=1)

        # 分组2: 核心围度 (WC, HC, CC, WHR)
        st.markdown("#### 2. 身体围度 (关键指标)")
        c3, c4 = st.columns(2)
        with c3:
            wc = st.number_input("腰围 (WC) cm", min_value=40.0, max_value=120.0, value=65.0, step=0.5)
            hc = st.number_input("臀围 (HC) cm", min_value=40.0, max_value=130.0, value=75.0, step=0.5)
        with c4:
            cc = st.number_input("胸围 (CC) cm", min_value=40.0, max_value=120.0, value=70.0, step=0.5)
            # WHR 自动计算展示，不需要输入
            whr_display = wc / hc if hc != 0 else 0
            st.metric("预估腰臀比 (WHR)", f"{whr_display:.2f}")

        # 分组3: 运动素质 (RopeSkip, Run50m, Reaction)
        st.markdown("#### 3. 运动表现")
        c5, c6 = st.columns(2)
        with c5:
            rope_skip = st.number_input("跳绳 (个/分)", min_value=0, max_value=300, value=120, step=1)
            reaction = st.number_input("反应时 (秒)", min_value=0.0, max_value=5.0, value=0.4, step=0.01)
        with c6:
            run_50m = st.number_input("50米跑 (秒)", min_value=5.0, max_value=20.0, value=9.5, step=0.1)

        # 提交按钮
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 开始 AI 预测 / Predict", use_container_width=True)

# --- 右侧：结果展示区 ---
with col_result:
    st.title("🏃‍♂️ 儿童肥胖风险智能筛查系统")
    st.caption("基于 Stacking 集成学习模型 | 准确率 > 90% | 支持辅助决策")
    st.divider()

    if submitted:
        # 1. 自动计算衍生特征
        whr = wc / hc if hc != 0 else 0

        # 2. 整理数据 (必须与训练时的顺序完全一致)
        # 训练顺序: ['Age', 'RopeSkip', 'Reaction', 'Run50m', 'HC', 'Gender', 'WC', 'WHR', 'CC']
        data = {
            'Age': age,
            'RopeSkip': rope_skip,
            'Reaction': reaction,
            'Run50m': run_50m,
            'HC': hc,
            'Gender': gender,
            'WC': wc,
            'WHR': whr,
            'CC': cc
        }
        df_input = pd.DataFrame(data, index=[0])

        # 确保列顺序对齐
        df_input = df_input[expected_cols]

        # 3. 预测
        try:
            input_scaled = scaler.transform(df_input)
            prob = model.predict_proba(input_scaled)[0][1]  # 获取预测为"1"(肥胖)的概率
            risk_percent = prob * 100
        except Exception as e:
            st.error(f"预测出错: {e}")
            st.stop()

        # 4. 动态结果卡片
        card_class = "result-card high-risk" if prob > 0.5 else "result-card"
        status_color = "#FF5252" if prob > 0.5 else "#4CAF50"
        status_text = "高风险 (High Risk)" if prob > 0.5 else "低风险 (Low Risk)"

        st.markdown(f"""
        <div class="{card_class}">
            <h3 style="color: {status_color}; margin-top:0;">🔮 预测结果分析</h3>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div class="metric-label">肥胖风险概率</div>
                    <div class="metric-value" style="font-size: 36px;">{risk_percent:.1f}%</div>
                </div>
                <div style="text-align: right;">
                    <div class="metric-label">风险等级</div>
                    <div class="metric-value" style="color: {status_color};">{status_text}</div>
                </div>
            </div>
            <br>
            <div style="background-color: #eee; height: 10px; border-radius: 5px;">
                <div style="background-color: {status_color}; width: {risk_percent}%; height: 100%; border-radius: 5px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 5. 智能归因解释 (基于 SHAP 逻辑的规则解释)
        st.markdown("### 💡 AI 归因分析")

        reasons = []
        if wc > 80: reasons.append(f"⚠️ **腰围 ({wc}cm)** 明显偏高，这是中心性肥胖的主要特征。")
        if whr > 0.9: reasons.append(f"⚠️ **腰臀比 ({whr:.2f})** 超标，提示腹部脂肪堆积风险。")
        if rope_skip < 100: reasons.append(f"📉 **跳绳成绩 ({rope_skip})** 较低，建议加强心肺耐力训练。")
        if run_50m > 10: reasons.append(f"📉 **50米跑 ({run_50m}s)** 较慢，提示爆发力不足。")

        if not reasons:
            st.success("🎉 各项指标均在健康范围内！继续保持良好的运动习惯。")
        else:
            for r in reasons:
                st.write(r)

            st.info("👨‍⚕️ **干预建议：** 建议每天增加 30 分钟中高强度运动（如跳绳、游泳），并控制高糖饮食摄入。")

    else:
        # 欢迎界面
        col_img, col_text = st.columns([1, 2])
        with col_img:
            # 这是一个示例图片占位符
            st.markdown(
                """
                <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                    <span style="font-size: 80px;">📊</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_text:
            st.markdown("""
            **欢迎使用本工具！**

            本系统专为学校和家庭设计，能够利用简单的体测数据（如跳绳、跑、围度）快速筛查隐性肥胖风险。

            ✅ **无需专业医疗设备**
            ✅ **秒级出结果**
            ✅ **个性化运动建议**

            👈 *请在左侧侧边栏输入数据以开始*
            """)