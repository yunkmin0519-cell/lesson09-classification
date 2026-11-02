import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ==========================================
# 1. 기본 설정
# ==========================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

RANDOM_STATE = 42

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

FEATURES = list(FEATURE_NAMES.keys())


# ==========================================
# 2. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()


# ==========================================
# 3. 화면 제목
# ==========================================

st.title("🤖 분류 모델")

st.markdown(
    """
    ### 뇌졸중을 예측하는 분류 모델 만들기

    다섯 가지 속성 가운데 원하는 속성을 골라
    로지스틱 회귀와 의사결정트리 모델을 학습하고 비교합니다.

    - 뇌졸중(1): 양성
    - 뇌졸중 아님(0): 음성
    """
)

st.divider()


# ==========================================
# 4. 입력 속성 선택
# ==========================================

st.header("1. 입력 속성 선택")

st.write(
    "모델 학습에 사용할 속성을 선택하세요. "
    "처음에는 체질량지수를 제외한 네 가지가 선택되어 있습니다."
)

selected_features = st.multiselect(
    "학습에 사용할 속성",
    options=FEATURES,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: FEATURE_NAMES[x]
)

# 두 개 미만이면 학습하지 않음
if len(selected_features) < 2:
    st.warning(
        "모델을 만들려면 입력 속성을 두 개 이상 선택해야 합니다."
    )
    st.stop()

st.write(
    "선택한 속성:",
    ", ".join(
        FEATURE_NAMES[x] for x in selected_features
    )
)

st.divider()


# ==========================================
# 5. 훈련 데이터와 테스트 데이터 분리
# ==========================================

st.header("2. 훈련 데이터와 테스트 데이터")

st.write(
    """
    데이터를 id 순서로 정렬한 뒤 10명씩 묶습니다.
    각 묶음의 앞 3명은 테스트 데이터로,
    나머지 7명은 훈련 데이터로 사용합니다.
    """
)

# id 순서대로 정렬
sorted_df = df.sort_values(
    "id",
    ascending=True,
    kind="stable"
).reset_index(drop=True)

# 10명씩 묶었을 때 앞 3명은 테스트 데이터
test_mask = (
    sorted_df.index % 10 < 3
)

train_df = sorted_df.loc[
    ~test_mask
].copy().reset_index(drop=True)

test_df = sorted_df.loc[
    test_mask
].copy().reset_index(drop=True)

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

# 데이터 분할 결과
split_col1, split_col2, split_col3 = st.columns(3)

with split_col1:
    st.metric(
        "전체 사람 수",
        f"{len(sorted_df):,}명"
    )

with split_col2:
    st.metric(
        "훈련 데이터",
        f"{len(train_df):,}명"
    )

with split_col3:
    st.metric(
        "테스트 데이터",
        f"{len(test_df):,}명"
    )

if len(test_df) != 1533:
    st.warning(
        f"테스트 데이터가 1,533명이 아닙니다. "
        f"현재 데이터는 {len(test_df):,}명입니다. "
        "데이터 원본의 행 수를 확인하세요."
    )
else:
    st.success(
        "테스트 데이터 1,533명으로 분리되었습니다."
    )


# ==========================================
# 6. 결측값 처리
# ==========================================

st.header("3. 결측값 처리")

# BMI를 선택한 경우에만 중앙값으로 채움
bmi_median = None

if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    train_bmi_missing = X_train["bmi"].isna().sum()
    test_bmi_missing = X_test["bmi"].isna().sum()

    X_train["bmi"] = X_train["bmi"].fillna(
        bmi_median
    )

    X_test["bmi"] = X_test["bmi"].fillna(
        bmi_median
    )

    st.write(
        f"체질량지수(BMI)의 훈련 데이터 중앙값: "
        f"**{bmi_median:.2f}**"
    )

    st.write(
        f"훈련 데이터 결측값: {train_bmi_missing:,}개 / "
        f"테스트 데이터 결측값: {test_bmi_missing:,}개"
    )

    st.caption(
        "훈련 데이터에서 계산한 중앙값을 두 데이터에 동일하게 적용했습니다."
    )

else:

    st.info(
        "체질량지수(BMI)를 선택하지 않았으므로 "
        "BMI 결측값 처리를 하지 않았습니다."
    )


# ==========================================
# 7. 크기 맞추기
# ==========================================

st.subheader("훈련 데이터만 사용한 크기 맞추기")

scaler = StandardScaler()

# 훈련 데이터로만 평균과 표준편차 계산
X_train_scaled = scaler.fit_transform(X_train)

# 테스트 데이터에는 훈련 데이터의 기준 적용
X_test_scaled = scaler.transform(X_test)

st.write(
    "훈련 데이터의 평균과 표준편차를 이용해 "
    "훈련 데이터와 테스트 데이터의 크기를 맞췄습니다."
)

st.divider()


# ==========================================
# 8. 로지스틱 회귀 모델
# ==========================================

st.header("4. 모델 만들기")

st.subheader(
    "① 로지스틱 회귀(확률로 답하는 모델)"
)

st.write(
    """
    입력 속성을 바탕으로 뇌졸중일 확률을 계산합니다.
    확률이 0.5 이상이면 뇌졸중(양성),
    0.5 미만이면 뇌졸중 아님(음성)으로 판단합니다.
    """
)

logistic_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=2000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

# 훈련 및 테스트 정확도
logistic_train_pred = logistic_model.predict(
    X_train_scaled
)

logistic_test_pred = logistic_model.predict(
    X_test_scaled
)

logistic_train_accuracy = accuracy_score(
    y_train,
    logistic_train_pred
)

logistic_test_accuracy = accuracy_score(
    y_test,
    logistic_test_pred
)

logistic_test_probability = logistic_model.predict_proba(
    X_test_scaled
)[:, list(logistic_model.classes_).index(1)]


# ==========================================
# 9. 의사결정트리 모델
# ==========================================

st.subheader(
    "② 의사결정트리(질문으로 답하는 모델)"
)

st.write(
    """
    속성에 관한 질문을 차례대로 던져
    뇌졸중 여부를 판단합니다.

    - 질문은 최대 세 번
    - 마지막 마디에 사람이 다섯 명 미만이면 더 나누지 않음
    - 난수는 42로 고정
    """
)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)

# 의사결정트리는 크기 맞추기 전 데이터를 사용
tree_model.fit(
    X_train,
    y_train
)

tree_train_pred = tree_model.predict(
    X_train
)

tree_test_pred = tree_model.predict(
    X_test
)

tree_train_accuracy = accuracy_score(
    y_train,
    tree_train_pred
)

tree_test_accuracy = accuracy_score(
    y_test,
    tree_test_pred
)


# ==========================================
# 10. 아무것도 보지 않는 기준 모델
# ==========================================

st.subheader(
    "③ 입력을 하나도 보지 않는 기준 모델"
)

st.write(
    """
    입력 속성을 전혀 사용하지 않고,
    훈련 데이터에서 더 많은 쪽으로만 답하는 모델입니다.
    """
)

# 훈련 데이터에서 많은 클래스
majority_class = int(
    y_train.value_counts().idxmax()
)

baseline_train_pred = [
    majority_class
] * len(y_train)

baseline_test_pred = [
    majority_class
] * len(y_test)

baseline_train_accuracy = accuracy_score(
    y_train,
    baseline_train_pred
)

baseline_test_accuracy = accuracy_score(
    y_test,
    baseline_test_pred
)

majority_name = (
    "뇌졸중(양성)"
    if majority_class == 1
    else "뇌졸중 아님(음성)"
)

st.info(
    f"이 모델은 모든 사람을 **{majority_name}**으로 답합니다."
)


# ==========================================
# 11. 정확도 비교 카드
# ==========================================

st.divider()

st.header("5. 모델 정확도 비교")

st.write(
    """
    큰 숫자는 테스트 데이터의 정확도입니다.
    아래에는 훈련 데이터와 테스트 데이터의 정확도를 함께 표시합니다.
    """
)

card1, card2, card3 = st.columns(3)

with card1:

    with st.container(border=True):

        st.markdown(
            "#### 로지스틱 회귀"
        )

        st.metric(
            "테스트 데이터 정확도",
            f"{logistic_test_accuracy * 100:.2f}%"
        )

        st.caption(
            f"훈련 데이터: {logistic_train_accuracy * 100:.2f}%"
        )

        st.caption(
            f"테스트 데이터: {logistic_test_accuracy * 100:.2f}%"
        )


with card2:

    with st.container(border=True):

        st.markdown(
            "#### 의사결정트리"
        )

        st.metric(
            "테스트 데이터 정확도",
            f"{tree_test_accuracy * 100:.2f}%"
        )

        st.caption(
            f"훈련 데이터: {tree_train_accuracy * 100:.2f}%"
        )

        st.caption(
            f"테스트 데이터: {tree_test_accuracy * 100:.2f}%"
        )


with card3:

    with st.container(border=True):

        st.markdown(
            "#### 입력을 보지 않는 모델"
        )

        st.metric(
            "테스트 데이터 정확도",
            f"{baseline_test_accuracy * 100:.2f}%"
        )

        st.caption(
            f"훈련 데이터: {baseline_train_accuracy * 100:.2f}%"
        )

        st.caption(
            f"테스트 데이터: {baseline_test_accuracy * 100:.2f}%"
        )


st.divider()


# ==========================================
# 12. 두 속성 선택
# ==========================================

st.header("6. 테스트 데이터 산점도와 모델의 판단")

st.write(
    """
    선택한 속성 중 두 가지를 골라 가로축과 세로축으로 사용합니다.

    테스트 데이터의 실제 뇌졸중 여부를 색으로 구분하고,
    두 모델이 공간을 어떻게 나누는지 살펴봅니다.
    """
)

axis_col1, axis_col2 = st.columns(2)

with axis_col1:

    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )

with axis_col2:

    y_options = [
        feature
        for feature in selected_features
        if feature != x_feature
    ]

    y_feature = st.selectbox(
        "세로축 속성",
        options=y_options,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )


# ==========================================
# 13. 나머지 속성 고정값
# ==========================================

st.subheader("나머지 속성의 고정값")

st.write(
    """
    산점도는 두 개의 축만 표현할 수 있으므로,
    나머지 속성은 테스트 데이터의 중앙값에 고정합니다.
    """
)

fixed_values = {}

for feature in selected_features:

    if feature not in [x_feature, y_feature]:

        fixed_value = float(
            X_test[feature].median()
        )

        fixed_values[feature] = fixed_value

        st.write(
            f"- **{FEATURE_NAMES[feature]}**: "
            f"{fixed_value:.3f}"
        )

if len(fixed_values) == 0:

    st.write(
        "선택한 속성이 두 개이므로 고정할 나머지 속성이 없습니다."
    )


# ==========================================
# 14. 산점도용 격자 데이터 생성
# ==========================================

# 테스트 데이터의 범위를 기준으로 격자 생성
x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

# 범위가 같은 경우를 대비
if x_min == x_max:
    x_min -= 0.5
    x_max += 0.5

if y_min == y_max:
    y_min -= 0.5
    y_max += 0.5

GRID_SIZE = 80

x_values = [
    x_min + (x_max - x_min) * i / (GRID_SIZE - 1)
    for i in range(GRID_SIZE)
]

y_values = [
    y_min + (y_max - y_min) * i / (GRID_SIZE - 1)
    for i in range(GRID_SIZE)
]

grid_records = []

for y_value in y_values:

    for x_value in x_values:

        row = {}

        for feature in selected_features:

            if feature == x_feature:
                row[feature] = x_value

            elif feature == y_feature:
                row[feature] = y_value

            else:
                row[feature] = fixed_values[feature]

        grid_records.append(row)

grid_df = pd.DataFrame(
    grid_records,
    columns=selected_features
)


# ==========================================
# 15. 격자에서 두 모델 예측
# ==========================================

# 로지스틱 회귀
grid_scaled = scaler.transform(grid_df)

grid_logistic_probability = logistic_model.predict_proba(
    grid_scaled
)[:, list(logistic_model.classes_).index(1)]

logistic_z = [
    grid_logistic_probability[
        i * GRID_SIZE:(i + 1) * GRID_SIZE
    ].tolist()
    for i in range(GRID_SIZE)
]


# 의사결정트리
grid_tree_prediction = tree_model.predict(
    grid_df
)

tree_z = [
    grid_tree_prediction[
        i * GRID_SIZE:(i + 1) * GRID_SIZE
    ].tolist()
    for i in range(GRID_SIZE)
]


# ==========================================
# 16. 모델 판단 영역 산점도
# ==========================================

fig_boundary = go.Figure()

# 의사결정트리의 판단 영역을 옅은 색으로 표시
fig_boundary.add_trace(
    go.Contour(
        x=x_values,
        y=y_values,
        z=tree_z,
        name="의사결정트리 판단 영역",
        showscale=False,
        opacity=0.20,
        colorscale=[
            [0.0, "#5DADE2"],
            [0.499, "#5DADE2"],
            [0.5, "#F1948A"],
            [1.0, "#F1948A"]
        ],
        contours=dict(
            start=0,
            end=1,
            size=1,
            coloring="fill",
            showlines=False
        ),
        hoverinfo="skip"
    )
)

# 로지스틱 회귀의 0.5 경계선
fig_boundary.add_trace(
    go.Contour(
        x=x_values,
        y=y_values,
        z=logistic_z,
        name="로지스틱 회귀 0.5 경계선",
        showscale=False,
        opacity=1,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            coloring="none",
            showlines=True
        ),
        line=dict(
            color="#222222",
            width=3
        ),
        hoverinfo="skip"
    )
)


# 실제 테스트 데이터 표시
stroke_labels = {
    0: "뇌졸중 아님(실제)",
    1: "뇌졸중(실제)"
}

for actual_class in [0, 1]:

    subset = test_df[
        test_df["stroke"] == actual_class
    ]

    fig_boundary.add_trace(
        go.Scatter(
            x=subset[x_feature],
            y=subset[y_feature],
            mode="markers",
            name=stroke_labels[actual_class],
            marker=dict(
                size=7,
                opacity=0.75,
                color=(
                    "#2874A6"
                    if actual_class == 0
                    else "#C0392B"
                ),
                line=dict(
                    width=0.5,
                    color="white"
                )
            ),
            hovertemplate=(
                f"{FEATURE_NAMES[x_feature]}: %{{x:.2f}}"
                f"<br>{FEATURE_NAMES[y_feature]}: %{{y:.2f}}"
                "<extra>%{fullData.name}</extra>"
            )
        )
    )


fig_boundary.update_layout(
    title="테스트 데이터와 두 분류 모델의 판단 영역",
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    height=650,
    legend_title="실제 뇌졸중 여부",
    hovermode="closest"
)

st.plotly_chart(
    fig_boundary,
    use_container_width=True
)


# ==========================================
# 17. 로지스틱 회귀 경계선 확인
# ==========================================

prob_min = min(
    min(row) for row in logistic_z
)

prob_max = max(
    max(row) for row in logistic_z
)

st.subheader("로지스틱 회귀의 0.5 경계선")

if prob_min <= 0.5 <= prob_max:

    st.success(
        "0.5 경계선이 현재 그림의 범위 안에 있습니다."
    )

else:

    if prob_max < 0.5:

        st.info(
            "현재 그림의 모든 격자에서 뇌졸중 확률이 "
            "0.5보다 작습니다. 경계선은 그림 밖에 있습니다."
        )

    else:

        st.info(
            "현재 그림의 모든 격자에서 뇌졸중 확률이 "
            "0.5 이상입니다. 경계선은 그림 밖에 있습니다."
        )


st.caption(
    "검은 선은 로지스틱 회귀 모델이 뇌졸중 확률 0.5를 기준으로 "
    "나누는 경계입니다. 옅은 파란색과 붉은색 영역은 "
    "의사결정트리 모델의 판단 영역입니다."
)


st.divider()


# ==========================================
# 18. 의사결정트리 가지 그림
# ==========================================

st.header("7. 의사결정트리가 던진 질문")

st.write(
    """
    의사결정트리가 어떤 질문을 던지고,
    어떤 과정을 거쳐 최종 답을 내리는지 살펴봅니다.
    """
)

tree_structure = tree_model.tree_

children_left = tree_structure.children_left
children_right = tree_structure.children_right

feature_indices = tree_structure.feature
thresholds = tree_structure.threshold

# 훈련 데이터가 각 마디에 몇 명 도달했는지 확인
decision_path = tree_model.decision_path(X_train)

# 실제 훈련 데이터의 마디별 인원과 양성 인원 계산
node_counts = {}
node_positive_counts = {}

for node_id in range(tree_structure.node_count):

    row_indices = decision_path[:, node_id].nonzero()[0]

    node_y = y_train.iloc[row_indices]

    node_counts[node_id] = len(node_y)

    node_positive_counts[node_id] = int(
        node_y.sum()
    )


# ==========================================
# 19. DOT 문자열 생성
# ==========================================

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="white", nodesep=0.45, ranksep=0.65];',
    'node [shape=box, style="rounded,filled", fontname="Malgun Gothic", fontsize=10, margin="0.18,0.12"];',
    'edge [fontname="Malgun Gothic", fontsize=10, color="#555555"];'
]

leaf_nodes = []
negative_leaf_nodes = []
used_tree_features = set()


def build_dot(node_id):

    n_people = node_counts[node_id]
    n_positive = node_positive_counts[node_id]

    positive_rate = (
        n_positive / n_people * 100
        if n_people > 0 else 0
    )

    # 현재 마디가 말단 마디인지 확인
    is_leaf = (
        children_left[node_id] == children_right[node_id]
    )

    if is_leaf:

        leaf_nodes.append(node_id)

        # 다수결로 최종 답 결정
        predicted_class = int(
            tree_structure.value[node_id][0].argmax()
        )

        if predicted_class == 0:

            negative_leaf_nodes.append(node_id)

            answer = "뇌졸중 아님 (음성)"
            fill_color = "#AED6F1"

        else:

            answer = "뇌졸중 (양성)"
            fill_color = "#F5B7B1"

        label = (
            f"마디 {node_id}\\n"
            f"사람 수: {n_people}명\\n"
            f"뇌졸중: {n_positive}명\\n"
            f"뇌졸중 비율: {positive_rate:.1f}%\\n"
            f"최종 답: {answer}"
        )

        dot_lines.append(
            f'N{node_id} [label="{label}", '
            f'fillcolor="{fill_color}", color="#555555"];'
        )

        return

    # 질문에 사용된 속성
    feature_index = feature_indices[node_id]

    feature = selected_features[feature_index]

    used_tree_features.add(feature)

    feature_name = FEATURE_NAMES[feature]

    threshold = thresholds[node_id]

    label = (
        f"마디 {node_id}\\n"
        f"{feature_name} ≤ {threshold:.3f}?\\n"
        f"사람 수: {n_people}명\\n"
        f"뇌졸중: {n_positive}명\\n"
        f"뇌졸중 비율: {positive_rate:.1f}%"
    )

    dot_lines.append(
        f'N{node_id} [label="{label}", '
        f'fillcolor="#FCF3CF", color="#B7950B"];'
    )

    left_id = children_left[node_id]
    right_id = children_right[node_id]

    # 왼쪽 가지: 조건이 참
    dot_lines.append(
        f'N{node_id} -> N{left_id} [label="예"];'
    )

    # 오른쪽 가지: 조건이 거짓
    dot_lines.append(
        f'N{node_id} -> N{right_id} [label="아니요"];'
    )

    build_dot(left_id)
    build_dot(right_id)


# 나무 생성 시작
build_dot(0)

dot_lines.append("}")

dot_string = "\n".join(dot_lines)


# ==========================================
# 20. 의사결정트리 출력
# ==========================================

st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# ==========================================
# 21. 의사결정트리 요약
# ==========================================

st.subheader("의사결정트리 요약")

total_leaf_count = len(leaf_nodes)
negative_leaf_count = len(negative_leaf_nodes)

st.write(
    f"- **답을 내는 마디 수:** {total_leaf_count}칸"
)

st.write(
    f"- **뇌졸중 아님(음성)으로 답하는 마디:** "
    f"{negative_leaf_count}칸"
)

st.write(
    "- **실제로 질문한 속성:** "
    + (
        ", ".join(
            FEATURE_NAMES[x]
            for x in FEATURES
            if x in used_tree_features
        )
        if used_tree_features
        else "없음"
    )
)


# ==========================================
# 22. 마무리
# ==========================================

st.divider()

st.markdown(
    """
    ### 🧠 실습 정리

    이번 실습에서는 세 가지 판단 방법을 비교했습니다.

    1. **로지스틱 회귀:** 확률을 계산하여 0.5를 기준으로 판단합니다.
    2. **의사결정트리:** 속성에 관한 질문을 이어 가며 판단합니다.
    3. **기준 모델:** 입력 속성을 보지 않고 훈련 데이터에서 많은 쪽으로만 판단합니다.

    정확도뿐 아니라 모델이 어떤 방식으로 판단하는지도 함께 살펴보세요.
    """
)

st.caption(
    "뇌졸중 예측 실습실 | 분류 모델"
)
