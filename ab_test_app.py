import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. データ生成関数 ---
def generate_ab_test_data(
    n_a: int = 10000,
    n_b: int = 10000,
    p_a: float = 0.10,
    p_b: float = 0.11
) -> pd.DataFrame:
    """
    A/Bテスト用の仮想データを生成する関数
    """
    data_a = np.random.binomial(n=1, p=p_a, size=n_a)
    data_b = np.random.binomial(n=1, p=p_b, size=n_b)

    df_a = pd.DataFrame({
        'user_id': [f'A_{i}' for i in range(n_a)],
        'group': 'A',
        'converted': data_a
    })

    df_b = pd.DataFrame({
        'user_id': [f'B_{i}' for i in range(n_b)],
        'group': 'B',
        'converted': data_b
    })

    df = pd.concat([df_a, df_b], ignore_index=True)
    return df

def generate_realistic_ab_data(scenario: str = "e_commerce", n_a: int = None, n_b: int = None) -> pd.DataFrame:
    """
    実在する業界の平均的なCVRに基づいたA/Bテスト仮想データ生成関数
    """
    scenarios = {
        'e_commerce': {
            'name': 'ECサイトの購買コンバージョン',
            'p_a': 0.020,
            'p_b': 0.023,
            'n_a': 50000,
            'n_b': 50000
        },
        'saas_signup': {
            'name': 'SaaSサービスの無料トライアル登録',
            'p_a': 0.050,
            'p_b': 0.056,
            'n_a': 20000,
            'n_b': 20000
        },
        'media_lead': {
            'name': 'メディアサイトの資料請求・会員登録',
            'p_a': 0.008,
            'p_b': 0.010,
            'n_a': 100000,
            'n_b': 100000
        }
    }

    if scenario not in scenarios:
        st.error(f"Unknown scenario. Choose from {list(scenarios.keys())}")
        return pd.DataFrame()

    config = scenarios[scenario]
    size_a = n_a if n_a is not None else config['n_a']
    size_b = n_b if n_b is not None else config['n_b']
    p_a = config['p_a']
    p_b = config['p_b']

    st.write(f"**【シナリオ設定: {config['name']}】**")
    st.write(f"  - A群 (現行): CVR {p_a*100:.1f}% / サンプル数 {size_a:,}人")
    st.write(f"  - B群 (改善): CVR {p_b*100:.1f}% / サンプル数 {size_b:,}人")

    data_a = np.random.binomial(n=1, p=p_a, size=size_a)
    data_b = np.random.binomial(n=1, p=p_b, size=size_b)

    df_a = pd.DataFrame({'user_id': [f'A_{i}' for i in range(size_a)], 'group': 'A', 'converted': data_a})
    df_b = pd.DataFrame({'user_id': [f'B_{i}' for i in range(size_b)], 'group': 'B', 'converted': data_b})

    return pd.concat([df_a, df_b], ignore_index=True)

# --- 2. 仮説検定ロジック ---
def evaluate_ab_test_ttest(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    A/Bテストのデータフレームを受け取り、Welchのt検定を行って有意差を判定する関数
    """
    data_a = df[df['group'] == 'A']['converted']
    data_b = df[df['group'] == 'B']['converted']

    if data_a.empty or data_b.empty:
        return {
            'cvr_a': 0.0,
            'cvr_b': 0.0,
            'diff': 0.0,
            't_statistic': np.nan,
            'p_value': np.nan,
            'is_significant': False,
            'conclusion': "データがありません"
        }

    t_stat, p_value = stats.ttest_ind(data_b, data_a, equal_var=False)

    cvr_a = data_a.mean()
    cvr_b = data_b.mean()

    is_significant = p_value < alpha

    result = {
        'cvr_a': cvr_a,
        'cvr_b': cvr_b,
        'diff': cvr_b - cvr_a,
        't_statistic': t_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'conclusion': "有意差あり" if is_significant else "有意差なし"
    }

    return result

# --- 3. 複数パターン検証シミュレーション ---
def run_simulation(n_samples: list, p_a: float, p_b_list: list, n_simulations: int = 1000, alpha: float = 0.05) -> pd.DataFrame:
    """
    複数パターンのサンプルサイズとCVRでA/Bテストのシミュレーションを実行する関数
    """
    results = []

    for n in n_samples:
        for p_b in p_b_list:
            significant_count = 0

            for _ in range(n_simulations):
                data_a = np.random.binomial(n=1, p=p_a, size=n)
                data_b = np.random.binomial(n=1, p=p_b, size=n)

                _, p_value = stats.ttest_ind(data_b, data_a, equal_var=False)

                if p_value < alpha:
                    significant_count += 1

            power = significant_count / n_simulations
            results.append({
                'sample_size_per_group': n,
                'true_p_a': p_a,
                'true_p_b': p_b,
                'true_diff': p_b - p_a,
                'power_or_fpr': power
            })

    return pd.DataFrame(results)

# --- 4. 結果レポート生成関数 ---
def generate_ab_test_report(df_sim_results: pd.DataFrame):
    """
    シミュレーション結果を受け取り、テキストレポートと可視化グラフを出力する関数（文字化け防止のため英語ラベル適用）
    """
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    st.subheader("【1. 第1種過誤（偽陽性率）の検証】（理論値: 5%）")
    df_fpr = df_sim_results[df_sim_results['true_diff'] == 0.0]
    for _, row in df_fpr.iterrows():
        st.write(f"  - サンプルサイズ: {row['sample_size_per_group']:>6}人/群 | 偽陽性率: {row['power_or_fpr']*100:.1f}%")

    st.subheader("【2. 検出力（Statistical Power）の検証】（目標: 80%以上）")
    df_power = df_sim_results[df_sim_results['true_diff'] > 0.0]
    for _, row in df_power.iterrows():
        st.write(f"  - サンプルサイズ: {row['sample_size_per_group']:>6}人/群 | 差分: +{row['true_diff']*100:.1f}%pt | 検出力: {row['power_or_fpr']*100:.1f}%")

    st.markdown("""---""")

    # グラフの描画（英語ラベル化）
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df_power,
        x='sample_size_per_group',
        y='power_or_fpr',
        hue='true_diff',
        marker='o',
        linewidth=2.5,
        ax=ax
    )

    ax.axhline(0.80, color='red', linestyle='--', alpha=0.7, label='Target Power (80%)')

    ax.set_title('Sample Size vs. Statistical Power in A/B Testing', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Sample Size per Group', fontsize=12)
    ax.set_ylabel('Statistical Power', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(title='True CVR Difference', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    st.pyplot(fig)

# --- Streamlit UI ---
st.title("A/Bテスト シミュレーションツール")
st.markdown("""
このツールは、A/Bテストの仮想データを生成し、仮説検定（Welchのt検定）に基づいて有意差を判定し、
さらに様々な条件でのシミュレーションを通じて検出力分析を行うことができます。
""")

# サイドバーの設定
st.sidebar.header("シミュレーション設定")

analysis_mode = st.sidebar.radio(
    "モード選択",
    ("単一A/Bテスト", "検出力シミュレーション")
)

p_a_base = st.sidebar.number_input("A群の基準CVR (例: 0.10)", min_value=0.001, max_value=0.999, value=0.10, step=0.001, format="%.3f")
alpha_level = st.sidebar.number_input("有意水準 (alpha) (例: 0.05)", min_value=0.001, max_value=0.1, value=0.05, step=0.001, format="%.3f")
np.random.seed(42)

if analysis_mode == "単一A/Bテスト":
    st.header("単一A/Bテストの実行")

    # アプリの使い方の解説セクション
    with st.expander("単一A/Bテストシミュレーションの使い方", expanded=False):
        st.markdown("""
        ###### この機能でできること
        指定したサンプルサイズとコンバージョン率（CVR）に基づき、1回分のA/Bテストのデータ（仮想データ）をランダムに生成します。そのデータに対して「Welchのt検定」を実行し、統計的に有意な差があるかを判定します。

        ###### 主要な用語・入力項目の解説
        * 真のCVR (True Conversion Rate):
          現実のビジネスでは知り得ない「そのグループが本来持っている本当のコンバージョン率（確率）」の仮置き値です。例えばA群を `10%`、B群を `11%` に設定すると、「本当は1%差がある世界」を再現してデータを作ります。
        * サンプルサイズ (Sample Size):
          A群・B群それぞれに割り当てるユーザー数です。サンプルサイズを増やすとデータのブレ（ノイズ）が小さくなり、本当の差を検出しやすくなります。逆に少なすぎると、本当は差があっても偶然のバラツキに埋もれて見逃しやすくなります。

        ###### 結果の見方
        * p値 (p-value): 「偶然この差が生じた確率」を示します。一般的に `0.05 (5%)` 未満であれば、「統計的に意味のある差（有意差がある）」と判断します。
        * 判定: 実際のビジネスで「この施策を採用すべきか、偶然の誤差として見送るべきか」を判断する目安になります。
        """)

    data_gen_method = st.radio(
        "データ生成方法",
        ("カスタム設定", "業界シナリオから選択")
    )

    df_ab = pd.DataFrame()

    if data_gen_method == "カスタム設定":
        st.subheader("データ生成 (カスタム設定)")
        col1, col2 = st.columns(2)
        with col1:
            n_a = st.number_input("A群のサンプルサイズ", min_value=100, value=10000, step=1000)
            p_a = st.number_input("A群の真のCVR", min_value=0.001, max_value=0.999, value=p_a_base, step=0.001, format="%.3f")
        with col2:
            n_b = st.number_input("B群のサンプルサイズ", min_value=100, value=10000, step=1000)
            p_b = st.number_input("B群の真のCVR", min_value=0.001, max_value=0.999, value=p_a_base + 0.01, step=0.001, format="%.3f")

        if st.button("仮想データ生成と検定実行"):
            df_ab = generate_ab_test_data(n_a=n_a, n_b=n_b, p_a=p_a, p_b=p_b)
            st.success("データ生成完了！")

    elif data_gen_method == "業界シナリオから選択":
        st.subheader("データ生成 (業界シナリオ)")
        scenario_choice = st.selectbox(
            "シナリオを選択",
            ('e_commerce', 'saas_signup', 'media_lead'),
            index=0
        )
        custom_n = st.checkbox("シナリオのサンプルサイズをカスタマイズする")
        n_a_scenario, n_b_scenario = None, None
        if custom_n:
            col1_s, col2_s = st.columns(2)
            with col1_s:
                n_a_scenario = st.number_input("A群のサンプルサイズ (シナリオ)", min_value=100, value=50000, step=1000)
            with col2_s:
                n_b_scenario = st.number_input("B群のサンプルサイズ (シナリオ)", min_value=100, value=50000, step=1000)

        if st.button("シナリオデータ生成と検定実行"):
            df_ab = generate_realistic_ab_data(scenario=scenario_choice, n_a=n_a_scenario, n_b=n_b_scenario)
            st.success("データ生成完了！")

    if not df_ab.empty:
        st.subheader("--- 生成データ集計サマリー ---")
        summary = df_ab.groupby('group')['converted'].agg(['count', 'sum', 'mean']).rename(
            columns={'count': 'サンプル数', 'sum': 'コンバージョン数', 'mean': 'CVR'}
        )
        st.dataframe(summary)

        st.subheader("--- 仮説検定の結果 (Welchのt検定) ---")
        test_result = evaluate_ab_test_ttest(df_ab, alpha=alpha_level)
        st.write(f"A群 CVR: {test_result['cvr_a']:.4f}")
        st.write(f"B群 CVR: {test_result['cvr_b']:.4f}")
        st.write(f"差分: {test_result['diff']:.4f}")
        st.write(f"t統計量: {test_result['t_statistic']:.4f}")
        st.write(f"p値: {test_result['p_value']:.4f}")
        st.write(f"**判定: {test_result['conclusion']}**")

elif analysis_mode == "検出力シミュレーション":
    st.header("検出力シミュレーション")

    # アプリの使い方の解説セクション
    with st.expander("検出力シミュレーションの使い方", expanded=False):
        st.markdown("""
        ###### この機能でできること
        「どれくらいのサンプルサイズを集めれば、本当に効果のある施策を見落とさずに検出できるか（＝検出力）」を、さまざまな条件でシミュレーションしてグラフ化します。

        ###### 各設定項目の意味と入力方法
        * A群の基準CVR / 有意水準 (alpha) (左サイドバーで設定):
          * A群の基準CVR: 現行案の基準となるコンバージョン率（例: `0.10` = 10%）です。
          * 有意水準 (alpha): 「本当は効果がないのに、偶然効果があると誤判定してしまう確率（第1種過誤）」の許容値です（通常は `0.05` / 5% に設定します）。
        * B群のCVRバリエーション (カンマ区切り):
          * 改善案（B群）として比較したい複数の真のCVRをカンマ区切りで指定します。
          * 例: `0.10, 0.11, 0.12` と入力すると、「A群が10%のとき、B群が10%, 11%, 12%（差が0%, 1%, 2%）の場合それぞれでどうなるか」を同時に比較・グラフ化できます。
        * サンプルサイズのバリエーション (カンマ区切り):
          * テストで集める1群あたりのユーザー数の候補を複数指定します。
          * 例: `1000, 3000, 5000, 10000` と指定すると、サンプル数が増えるにつれて検出力がどう伸びるかがわかります。
        * シミュレーション試行回数:
          * 確率的なバラつきを正確に測るため、各条件で何回テストを仮想的に繰り返すかの回数です（多いほど正確ですが、計算に少し時間がかかります）。

        ###### 活用のコツ
        * 目標とする検出力（一般的には `80%` 以上）に達するために、何人のユーザーを集めなければならないか（必要なサンプルサイズ）を見極めるために使用します。
        """)

    st.markdown("""
    様々なサンプルサイズとCVR差におけるA/Bテストの検出力（真の差を検出できる確率）をシミュレーションします。
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"A群の基準CVR: {p_a_base*100:.1f}%")
        st.write(f"有意水準 (alpha): {alpha_level}")
        target_p_b_str = st.text_input("B群のCVRバリエーション (カンマ区切り, 例: 0.10,0.11,0.12)", value="0.10,0.11,0.12")
        target_p_b_list = [float(p) for p in target_p_b_str.split(',')]
    with col2:
        sample_sizes_str = st.text_input("サンプルサイズのバリエーション (カンマ区切り, 例: 1000,5000,10000)", value="1000,3000,5000,10000,20000,40000")
        sample_sizes = [int(n) for n in sample_sizes_str.split(',')]
        n_simulations = st.number_input("シミュレーション試行回数", min_value=100, value=300, step=100)

    if st.button("シミュレーション実行"):
        st.info(f"シミュレーションを実行中... (試行回数: {n_simulations})")
        df_sim_results = run_simulation(
            n_samples=sample_sizes,
            p_a=p_a_base,
            p_b_list=target_p_b_list,
            n_simulations=n_simulations,
            alpha=alpha_level
        )

        st.subheader("--- シミュレーション結果レポート ---")
        generate_ab_test_report(df_sim_results)

        st.download_button(
            label="シミュレーション結果をCSVでダウンロード",
            data=df_sim_results.to_csv(index=False).encode('utf-8'),
            file_name='ab_test_simulation_results.csv',
            mime='text/csv'
        )
