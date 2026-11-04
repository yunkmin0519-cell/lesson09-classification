from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------
# 기본 설정 및 데이터
# ---------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
RANDOM_STATE = 42

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]

st.set_page_config(
    page_title=f"{APP_TITLE} - 성능 높이기",
    page_icon="🧠",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """기존 페이지와 같은 GitHub 원본 데이터를 UTF-8로 읽습니다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


def make_train_test_split(data: pd.DataFrame):
    """
    id 오름차순으로 정렬한 뒤 10명씩 묶어,
    각 묶음의 앞 3명은 테스트, 뒤 7명은 훈련 데이터로 사용합니다.
    """
    ordered = data.sort_values("id", kind="mergesort").reset_index(drop=True)
    test_mask = pd.Series(
        [(position % 10) < 3 for position in range(len(ordered))],
        index=ordered.index,
    )
    train = ordered.loc[~test_mask].copy()
    test = ordered.loc[test_mask].copy()
    return train, test


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
):
    """BMI 결측값은 훈련 중앙값으로 채우고, 표준화도 훈련 자료에만 맞춥니다."""
    x_train = train[features].astype(float).copy()
    x_test = test[features].astype(float).copy()
    bmi_median = None

    if "bmi" in features:
        bmi_median = x_train["bmi"].median()
        if pd.isna(bmi_median):
            bmi_median = 0.0
        x_train["bmi"] = x_train["bmi"].fillna(bmi_median)
        x_test["bmi"] = x_test["bmi"].fillna(bmi_median)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train, x_test, x_train_scaled, x_test_scaled, scaler, bmi_median


def confusion_counts(y_true, y_pred):
    actual = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    return {
        "TN": int(((actual == 0) & (predicted == 0)).sum()),
        "FP": int(((actual == 0) & (predicted == 1)).sum()),
        "FN": int(((actual == 1) & (predicted == 0)).sum()),
        "TP": int(((actual == 1) & (predicted == 1)).sum()),
    }


def metric_values(counts: dict[str, int]):
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1,
    }


def format_metric(value):
    return "계산할 수 없음" if value is None else f"{value * 100:.2f}%"


def make_comparison_table(before, after):
    rows = ["TN(정답 음성)", "FP(헛짚음)", "FN(놓침)", "TP(찾아냄)",
            "정확도", "재현율", "정밀도", "F1"]
    before_metrics = metric_values(before)
    after_metrics = metric_values(after)
    before_values = [
        f"{before['TN']:,}명", f"{before['FP']:,}명",
        f"{before['FN']:,}명", f"{before['TP']:,}명",
        *[format_metric(before_metrics[name]) for name in ["정확도", "재현율", "정밀도", "F1"]],
    ]
    after_values = [
        f"{after['TN']:,}명", f"{after['FP']:,}명",
        f"{after['FN']:,}명", f"{after['TP']:,}명",
        *[format_metric(after_metrics[name]) for name in ["정확도", "재현율", "정밀도", "F1"]],
    ]
    return pd.DataFrame(
        {"가중치 조정 전": before_values, "가중치 조정 후": after_values},
        index=rows,
    )


def model_probability(model, points):
    return model.predict_proba(points)[:, list(model.classes_).index(1)]


def make_2d_plot(
    test_frame, y_test, feature_x, feature_y,
    models, scaler, features, fixed_values,
):
    """실제 정답 산점도와 두 모델의 0.5 확률 경계선을 겹쳐 그립니다."""
    x_min, x_max = test_frame[feature_x].min(), test_frame[feature_x].max()
    y_min, y_max = test_frame[feature_y].min(), test_frame[feature_y].max()
    x_pad = max((x_max - x_min) * 0.06, 0.05)
    y_pad = max((y_max - y_min) * 0.06, 0.05)
    gx = np.linspace(x_min - x_pad, x_max + x_pad, 100)
    gy = np.linspace(y_min - y_pad, y_max + y_pad, 100)
    xx, yy = np.meshgrid(gx, gy)

    fig = go.Figure()
    labels = np.asarray(y_test, dtype=int)
    for value, label, color, symbol in [
        (0, "실제 뇌졸중 아님", "#2878B5", "circle"),
        (1, "실제 뇌졸중", "#D62728", "circle"),
    ]:
        mask = labels == value
        fig.add_trace(go.Scatter(
            x=test_frame.loc[mask, feature_x],
            y=test_frame.loc[mask, feature_y],
            mode="markers",
            name=label,
            marker=dict(color=color, size=7, symbol=symbol, opacity=0.72),
            hovertemplate=(
                f"{FEATURE_LABELS[feature_x]}: %{{x:.2f}}<br>"
                f"{FEATURE_LABELS[feature_y]}: %{{y:.2f}}<br>"
                f"{label}<extra></extra>"
            ),
        ))

    line_styles = [
        ("가중치 조정 전", "#16A085", "solid"),
        ("가중치 조정 후", "#8E44AD", "dash"),
    ]
    for (model_name, model), (label, color, dash) in zip(models.items(), line_styles):
        grid = pd.DataFrame(
            {feature: np.full(xx.size, fixed_values[feature], dtype=float)
             for feature in features}
        )
        grid[feature_x] = xx.ravel()
        grid[feature_y] = yy.ravel()
        probabilities = model_probability(model, scaler.transform(grid[features])).reshape(xx.shape)

        crosses = probabilities.min() <= 0.5 <= probabilities.max()
        if crosses:
            fig.add_trace(go.Contour(
                x=gx, y=gy, z=probabilities,
                contours=dict(start=0.5, end=0.5, size=1, coloring="none", showlabels=False),
                line=dict(color=color, width=3, dash=dash),
                showscale=False,
                name=label,
                hoverinfo="skip",
            ))
        else:
            st.info(
                f"{label}의 0.5 경계선은 현재 그림 범위 밖에 있습니다. "
                f"(그림 범위에서 예측 확률 {probabilities.min():.3f}~{probabilities.max():.3f})"
            )

    fig.update_layout(
        title="실제 뇌졸중 여부와 가중치 조정 전·후 경계선",
        xaxis_title=FEATURE_LABELS[feature_x],
        yaxis_title=FEATURE_LABELS[feature_y],
        legend_title="점·경계선 구분",
        height=650,
        template="plotly_white",
    )
    return fig


def make_3d_plot(
    test_frame, y_test, axes, models, scaler, features, fixed_values,
):
    """실제 정답 산점도와 두 모델의 0.5 등확률면을 3차원으로 표시합니다."""
    axis_ranges = []
    for feature in axes:
        low, high = test_frame[feature].min(), test_frame[feature].max()
        pad = max((high - low) * 0.05, 0.05)
        axis_ranges.append(np.linspace(low - pad, high + pad, 22))

    xx, yy, zz = np.meshgrid(*axis_ranges, indexing="ij")
    fig = go.Figure()
    labels = np.asarray(y_test, dtype=int)

    for value, label, color in [
        (0, "실제 뇌졸중 아님", "#2878B5"),
        (1, "실제 뇌졸중", "#D62728"),
    ]:
        mask = labels == value
        fig.add_trace(go.Scatter3d(
            x=test_frame.loc[mask, axes[0]],
            y=test_frame.loc[mask, axes[1]],
            z=test_frame.loc[mask, axes[2]],
            mode="markers",
            name=label,
            marker=dict(color=color, size=4, opacity=0.75),
            hovertemplate=(
                f"{FEATURE_LABELS[axes[0]]}: %{{x:.2f}}<br>"
                f"{FEATURE_LABELS[axes[1]]}: %{{y:.2f}}<br>"
                f"{FEATURE_LABELS[axes[2]]}: %{{z:.2f}}<br>{label}<extra></extra>"
            ),
        ))

    line_styles = [
        ("가중치 조정 전", "#16A085", 0.28),
        ("가중치 조정 후", "#8E44AD", 0.28),
    ]
    for (model_name, model), (label, color, opacity) in zip(models.items(), line_styles):
        grid = pd.DataFrame(
            {feature: np.full(xx.size, fixed_values[feature], dtype=float)
             for feature in features}
        )
        grid[axes[0]] = xx.ravel()
        grid[axes[1]] = yy.ravel()
        grid[axes[2]] = zz.ravel()
        probabilities = model_probability(model, scaler.transform(grid[features])).reshape(xx.shape)

        if probabilities.min() <= 0.5 <= probabilities.max():
            fig.add_trace(go.Isosurface(
                x=xx.ravel(), y=yy.ravel(), z=zz.ravel(),
                value=probabilities.ravel(),
                isomin=0.499, isomax=0.501, surface_count=1,
                caps=dict(x_show=False, y_show=False, z_show=False),
                opacity=opacity,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                name=label,
                showlegend=True,
                hoverinfo="skip",
            ))
        else:
            st.info(
                f"{label}의 0.5 경계면은 현재 3차원 그림 범위 밖에 있습니다. "
                f"(그림 범위에서 예측 확률 {probabilities.min():.3f}~{probabilities.max():.3f})"
            )

    fig.update_layout(
        title="실제 뇌졸중 여부와 가중치 조정 전·후 0.5 경계면",
        scene=dict(
            xaxis_title=FEATURE_LABELS[axes[0]],
            yaxis_title=FEATURE_LABELS[axes[1]],
            zaxis_title=FEATURE_LABELS[axes[2]],
        ),
        height=750,
        template="plotly_white",
        legend_title="점·경계면 구분",
    )
    return fig


# ---------------------------------------------------------
# 화면 제목, 데이터 읽기, 입력 속성 선택
# ---------------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.header("성능 높이기")
st.caption(
    "뇌졸중 환자가 드문 자료에서는 모든 사람을 뇌졸중이 아니라고 예측해도 "
    "정확도가 높게 나올 수 있습니다. 클래스 가중치를 조정하기 전과 후를 비교합니다."
)

try:
    data = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.exception(error)
    st.stop()

required_columns = ["id", "stroke", *FEATURE_LABELS.keys()]
missing_columns = [column for column in required_columns if column not in data.columns]
if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {', '.join(missing_columns)}")
    st.stop()

selected_features = st.multiselect(
    "모델 입력 속성 선택",
    options=list(FEATURE_LABELS.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda column: FEATURE_LABELS[column],
    help="두 개 이상 선택해야 모델을 학습하고 비교할 수 있습니다.",
)
if len(selected_features) < 2:
    st.info("모델 입력 속성을 두 개 이상 골라 주세요.")
    st.stop()

# ---------------------------------------------------------
# 기존 채점표와 동일한 데이터 분할 및 전처리
# ---------------------------------------------------------
train_data, test_data = make_train_test_split(data)
x_train, x_test, x_train_scaled, x_test_scaled, scaler, bmi_median = (
    prepare_features(train_data, test_data, selected_features)
)
y_train = train_data["stroke"].astype(int)
y_test = test_data["stroke"].astype(int)

if y_train.nunique() < 2:
    st.error("훈련 데이터에 뇌졸중과 비뇌졸중 두 종류가 모두 있어야 합니다.")
    st.stop()

if bmi_median is not None:
    st.caption(
        f"체질량지수 결측값은 훈련 데이터의 중앙값 {bmi_median:.2f}으로 채웠습니다. "
        "테스트 데이터의 중앙값은 사용하지 않았습니다."
    )

st.caption(
    f"전체 {len(data):,}명 · 훈련 {len(train_data):,}명 · 테스트 {len(test_data):,}명. "
    "id 오름차순으로 정렬한 뒤 10명씩 묶어 앞 3명은 테스트, 뒤 7명은 훈련에 사용했습니다."
)

# ---------------------------------------------------------
# 가중치 조정 전·후 모델 학습
# ---------------------------------------------------------
models = {
    "로지스틱 회귀(확률로 답하는 모델)": (
        LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
        LogisticRegression(
            random_state=RANDOM_STATE, max_iter=2000, class_weight="balanced"
        ),
    ),
    "의사결정트리(질문으로 답하는 모델)": (
        DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=5,
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
    ),
}

predictions = {}
probability_models = {}
for display_name, (before_model, after_model) in models.items():
    before_model.fit(x_train_scaled, y_train)
    after_model.fit(x_train_scaled, y_train)
    predictions[display_name] = {
        "전": before_model.predict(x_test_scaled),
        "후": after_model.predict(x_test_scaled),
    }
    probability_models[display_name] = {
        "전": before_model,
        "후": after_model,
    }

# 시각화 함수가 받기 쉽도록 모델 두 개를 순서대로 구성
logistic_pair = probability_models["로지스틱 회귀(확률로 답하는 모델)"]
tree_pair = probability_models["의사결정트리(질문으로 답하는 모델)"]

# ---------------------------------------------------------
# 전후 성능 비교표
# ---------------------------------------------------------
st.subheader("가중치 조정 전·후 성능 비교")
st.write(
    "가중치 조정 후에는 드문 뇌졸중 사례의 오답에 더 큰 벌점을 주도록 학습합니다. "
    "테스트 자료와 판정 기준(확률 0.5)은 전후 모두 같습니다."
)

for display_name in models:
    before_counts = confusion_counts(y_test, predictions[display_name]["전"])
    after_counts = confusion_counts(y_test, predictions[display_name]["후"])
    st.markdown(f"**{display_name}**")
    st.dataframe(
        make_comparison_table(before_counts, after_counts),
        use_container_width=True,
    )
    st.caption(
        "찾아낸 환자(TP): "
        f"{before_counts['TP']:,}명 → {after_counts['TP']:,}명  |  "
        "놓친 환자(FN): "
        f"{before_counts['FN']:,}명 → {after_counts['FN']:,}명  |  "
        "헛짚은 사람(FP): "
        f"{before_counts['FP']:,}명 → {after_counts['FP']:,}명"
    )

st.caption(
    "TN은 뇌졸중이 아닌 사람을 아니라고 맞힌 수, FP는 뇌졸중이 아닌 사람을 "
    "뇌졸중이라고 잘못 짚은 수, FN은 실제 환자를 놓친 수, TP는 실제 환자를 찾아낸 수입니다. "
    "정확도·재현율·정밀도·F1은 백분율입니다."
)

# ---------------------------------------------------------
# 실제 정답 산점도 및 전후 경계선/경계면
# ---------------------------------------------------------
st.subheader("실제 환자 분포와 0.5 경계 비교")
st.caption(
    "점 색은 실제 뇌졸중 여부를 뜻합니다. 실선은 가중치 조정 전, "
    "점선(또는 보라색 경계면)은 가중치 조정 후입니다. "
    "선택하지 않은 속성은 테스트 자료의 중앙값으로 고정합니다."
)

fixed_values = {
    feature: float(x_test[feature].median())
    for feature in selected_features
}

if len(selected_features) == 2:
    axis_x, axis_y = selected_features
    st.plotly_chart(
        make_2d_plot(
            x_test, y_test, axis_x, axis_y,
            {
                "가중치 조정 전": logistic_pair["전"],
                "가중치 조정 후": logistic_pair["후"],
            },
            scaler, selected_features, fixed_values,
        ),
        use_container_width=True,
    )
    st.markdown("**의사결정트리의 경계선**")
    st.plotly_chart(
        make_2d_plot(
            x_test, y_test, axis_x, axis_y,
            {
                "가중치 조정 전": tree_pair["전"],
                "가중치 조정 후": tree_pair["후"],
            },
            scaler, selected_features, fixed_values,
        ),
        use_container_width=True,
    )
else:
    default_axes = selected_features[:3]
    axis_columns = st.columns(3)
    axis_choices = []
    for index, (column, default_feature) in enumerate(zip(axis_columns, default_axes)):
        remaining = [feature for feature in selected_features if feature not in axis_choices]
        chosen = column.selectbox(
            ["가로축", "세로축", "깊이축"][index],
            options=remaining,
            index=remaining.index(default_feature) if default_feature in remaining else 0,
            format_func=lambda feature: FEATURE_LABELS[feature],
            key=f"axis_{index}",
        )
        axis_choices.append(chosen)

    st.plotly_chart(
        make_3d_plot(
            x_test, y_test, axis_choices,
            {
                "가중치 조정 전": logistic_pair["전"],
                "가중치 조정 후": logistic_pair["후"],
            },
            scaler, selected_features, fixed_values,
        ),
        use_container_width=True,
    )
    st.markdown("**의사결정트리의 3차원 경계면**")
    st.plotly_chart(
        make_3d_plot(
            x_test, y_test, axis_choices,
            {
                "가중치 조정 전": tree_pair["전"],
                "가중치 조정 후": tree_pair["후"],
            },
            scaler, selected_features, fixed_values,
        ),
        use_container_width=True,
    )

st.info(
    "가중치 균형은 환자를 더 잘 찾아내는 데 도움이 될 수 있지만, "
    "헛짚는 사람(FP)이 늘거나 정확도가 낮아질 수도 있습니다. "
    "어느 한 지표만 보지 말고 재현율·정밀도·F1과 혼동행렬을 함께 살펴보세요."
)
