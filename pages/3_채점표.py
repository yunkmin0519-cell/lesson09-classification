from __future__ import annotations

import pandas as pd
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
    page_title=f"{APP_TITLE} - 채점표",
    page_icon="🧠",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    """분류 모델 페이지와 같은 GitHub 원본 데이터를 UTF-8로 읽습니다."""
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
    """
    BMI를 선택했을 때에만 훈련 데이터의 중앙값으로 결측값을 채웁니다.
    표준화는 훈련 데이터에만 맞춘 뒤 테스트 데이터에 적용합니다.
    """
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
    """양성은 stroke=1로 두고 TN, FP, FN, TP를 계산합니다."""
    actual = pd.Series(y_true).astype(int).to_numpy()
    predicted = pd.Series(y_pred).astype(int).to_numpy()

    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tp = int(((actual == 1) & (predicted == 1)).sum())

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def calculate_metrics(counts: dict[str, int]) -> dict[str, str]:
    """혼동행렬의 네 칸으로 네 가지 평가 지표를 계산합니다."""
    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]
    tn = counts["TN"]

    total = tp + fp + fn + tn

    accuracy = (tp + tn) / total if total else 0.0

    # 실제 뇌졸중인 사람 수가 분모
    recall = tp / (tp + fn) if (tp + fn) else None

    # 뇌졸중이라고 예측한 사람 수가 분모
    precision = tp / (tp + fp) if (tp + fp) else None

    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    def formatted(value):
        return "계산할 수 없음" if value is None else f"{value * 100:.2f}%"

    return {
        "정확도": formatted(accuracy),
        "재현율": formatted(recall),
        "정밀도": formatted(precision),
        "F1": formatted(f1),
    }


def confusion_table(counts: dict[str, int]) -> pd.DataFrame:
    """행은 실제, 열은 예측이며 행·열 합계를 함께 표시합니다."""
    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]
    tn = counts["TN"]

    actual_negative = tn + fp
    actual_positive = fn + tp
    predicted_negative = tn + fn
    predicted_positive = fp + tp
    total = tn + fp + fn + tp

    return pd.DataFrame(
        [
            [
                f"TN: {tn:,}명",
                f"FP: {fp:,}명",
                f"{actual_negative:,}명",
            ],
            [
                f"FN: {fn:,}명",
                f"TP: {tp:,}명",
                f"{actual_positive:,}명",
            ],
            [
                f"{predicted_negative:,}명",
                f"{predicted_positive:,}명",
                f"{total:,}명",
            ],
        ],
        index=["실제: 아님", "실제: 뇌졸중", "열 합계"],
        columns=["예측: 아님", "예측: 뇌졸중", "행 합계"],
    )


def make_error_list(
    test_data: pd.DataFrame,
    y_true,
    y_pred,
    error_type: str,
) -> pd.DataFrame:
    """FN 또는 FP에 해당하는 사람을 나이 내림차순으로 정렬합니다."""
    actual = pd.Series(y_true, index=test_data.index).astype(int)
    predicted = pd.Series(y_pred, index=test_data.index).astype(int)

    if error_type == "FN":
        mask = (actual == 1) & (predicted == 0)
    else:
        mask = (actual == 0) & (predicted == 1)

    selected = test_data.loc[mask].copy()

    result = pd.DataFrame(
        {
            "번호": selected["id"].astype(int),
            "나이": selected["age"],
            "평균 혈당": selected["avg_glucose_level"],
            "고혈압 여부": selected["hypertension"].map(
                {0: "아님", 1: "있음"}
            ).fillna(selected["hypertension"].astype(str)),
            "심장병 여부": selected["heart_disease"].map(
                {0: "아님", 1: "있음"}
            ).fillna(selected["heart_disease"].astype(str)),
        }
    )

    result = result.sort_values(
        "나이",
        ascending=False,
        kind="mergesort",
    )

    return result


def show_error_list(title: str, frame: pd.DataFrame):
    """목록 전체 인원수를 제목에 표시하고 최대 20명까지만 보여 줍니다."""
    st.markdown(f"**{title} — 전체 {len(frame):,}명**")

    if frame.empty:
        st.write("해당하는 사람이 없습니다.")
        return

    st.dataframe(
        frame.head(20),
        hide_index=True,
        use_container_width=True,
    )

    if len(frame) > 20:
        st.caption(
            f"전체 {len(frame):,}명 가운데 나이가 많은 순서로 "
            f"앞의 20명만 표시했습니다."
        )


# ---------------------------------------------------------
# 화면 제목 및 데이터 읽기
# ---------------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.header("채점표")

st.caption(
    "뇌졸중(stroke=1)을 양성, 뇌졸중이 아님(stroke=0)을 음성으로 평가합니다. "
    "혼동행렬의 행은 실제 정답, 열은 모델의 예측입니다."
)

try:
    data = load_data()
except Exception as error:
    st.error(
        "데이터를 불러오지 못했습니다. "
        "인터넷 연결과 데이터 주소를 확인해 주세요."
    )
    st.exception(error)
    st.stop()


required_columns = [
    "id",
    "stroke",
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:
    st.error(f"데이터에 필요한 열이 없습니다: {', '.join(missing_columns)}")
    st.stop()


# ---------------------------------------------------------
# 입력 속성 선택: 분류 모델 페이지와 동일한 기본 선택
# ---------------------------------------------------------
selected_features = st.multiselect(
    "모델 입력 속성 선택",
    options=list(FEATURE_LABELS.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda column: FEATURE_LABELS[column],
    help="분류 모델 페이지와 마찬가지로 두 개 이상 선택해야 합니다.",
)

if len(selected_features) < 2:
    st.info("모델 입력 속성을 두 개 이상 골라 주세요.")
    st.stop()


# ---------------------------------------------------------
# 분류 모델 페이지와 동일한 분할 및 전처리
# ---------------------------------------------------------
train_data, test_data = make_train_test_split(data)

if len(test_data) != 1533:
    st.warning(
        f"현재 테스트 데이터는 {len(test_data):,}명입니다. "
        "원본 데이터의 행 수가 실습에서 안내한 데이터와 다른지 확인해 주세요."
    )

(
    x_train,
    x_test,
    x_train_scaled,
    x_test_scaled,
    scaler,
    bmi_median,
) = prepare_features(
    train_data,
    test_data,
    selected_features,
)

y_train = train_data["stroke"].astype(int)
y_test = test_data["stroke"].astype(int)

if y_train.nunique() < 2:
    st.error(
        "훈련 데이터에 뇌졸중과 뇌졸중이 아닌 사람이 모두 있어야 "
        "분류 모델을 학습할 수 있습니다."
    )
    st.stop()

if bmi_median is not None:
    st.caption(
        f"체질량지수 결측값은 훈련 데이터의 중앙값 "
        f"{bmi_median:.2f}으로 채웠습니다."
    )


# ---------------------------------------------------------
# 두 모델 학습: 분류 모델 페이지와 동일한 설정
# ---------------------------------------------------------
logistic_model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=2000,
)

logistic_model.fit(
    x_train_scaled,
    y_train,
)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)

tree_model.fit(
    x_train,
    y_train,
)


# 테스트 데이터 예측
logistic_predictions = logistic_model.predict(x_test_scaled)
tree_predictions = tree_model.predict(x_test)


# 기준 모델: 훈련용에서 많은 쪽으로만 답하기
majority_class = int(y_train.value_counts().idxmax())

baseline_predictions = pd.Series(
    majority_class,
    index=y_test.index,
)


# ---------------------------------------------------------
# 테스트 데이터 구성 안내
# ---------------------------------------------------------
test_positive_count = int((y_test == 1).sum())

st.subheader("테스트 데이터")

st.write(
    f"테스트용 사람 수: **{len(test_data):,}명**"
    f"　|　그중 실제 뇌졸중인 사람 수: **{test_positive_count:,}명**"
)

st.caption(
    "id 오름차순으로 정렬한 뒤 10명씩 묶고, 각 묶음의 앞 3명을 "
    "테스트용으로 고정했습니다. 나머지 7명만 학습과 표준화에 사용했습니다."
)


# ---------------------------------------------------------
# 혼동행렬 계산
# ---------------------------------------------------------
model_predictions = {
    "로지스틱 회귀(확률로 답하는 모델)": logistic_predictions,
    "의사결정트리(질문으로 답하는 모델)": tree_predictions,
    "훈련용에서 많은 쪽으로만 답하는 모델": baseline_predictions,
}

model_counts = {
    name: confusion_counts(y_test, predictions)
    for name, predictions in model_predictions.items()
}


# ---------------------------------------------------------
# 혼동행렬 표시
# ---------------------------------------------------------
st.divider()
st.subheader("테스트 데이터 혼동행렬")

st.write(
    "각 칸에는 해당하는 사람 수와 TP·FP·FN·TN 약자를 함께 표시했습니다. "
    "오른쪽 아래의 합계는 전체 테스트 인원입니다."
)

matrix_cols = st.columns(3)

for column, (model_name, counts) in zip(matrix_cols, model_counts.items()):
    with column:
        st.markdown(f"**{model_name}**")
        st.dataframe(
            confusion_table(counts),
            use_container_width=True,
        )


# ---------------------------------------------------------
# 평가 지표 비교
# ---------------------------------------------------------
st.divider()
st.subheader("평가 지표 비교")

st.write(
    "재현율은 실제 뇌졸중인 사람 가운데 찾아낸 비율이고, "
    "정밀도는 뇌졸중이라고 예측한 사람 가운데 실제 뇌졸중인 비율입니다."
)

metric_rows = []

for model_name, counts in model_counts.items():
    metrics = calculate_metrics(counts)

    metric_rows.append(
        {
            "모델": model_name,
            "정확도": metrics["정확도"],
            "재현율": metrics["재현율"],
            "정밀도": metrics["정밀도"],
            "F1": metrics["F1"],
        }
    )

metrics_table = pd.DataFrame(metric_rows)

st.dataframe(
    metrics_table,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "정확도 = (TP + TN) / 전체, "
    "재현율 = TP / (TP + FN), "
    "정밀도 = TP / (TP + FP), "
    "F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)입니다. "
    "뇌졸중이라고 예측한 사람이 0명이면 정밀도와 F1은 "
    "'계산할 수 없음'으로 표시합니다."
)


# ---------------------------------------------------------
# 놓친 환자와 헛짚은 사람 목록
# ---------------------------------------------------------
st.divider()
st.subheader("틀리게 예측한 사람 살펴보기")

model_choice = st.selectbox(
    "목록을 확인할 모델 선택",
    options=list(model_predictions.keys()),
)

chosen_predictions = model_predictions[model_choice]
chosen_counts = model_counts[model_choice]

false_negative_list = make_error_list(
    test_data,
    y_test,
    chosen_predictions,
    "FN",
)

false_positive_list = make_error_list(
    test_data,
    y_test,
    chosen_predictions,
    "FP",
)

left_col, right_col = st.columns(2)

with left_col:
    show_error_list(
        "놓친 환자 (실제 뇌졸중인데 '아님'으로 예측)",
        false_negative_list,
    )

with right_col:
    show_error_list(
        "헛짚은 사람 (실제는 아닌데 뇌졸중으로 예측)",
        false_positive_list,
    )


st.caption(
    f"목록의 인원수는 혼동행렬의 FN·FP와 일치합니다. "
    f"놓친 환자(FN): {len(false_negative_list):,}명 "
    f"= FN {chosen_counts['FN']:,}명, "
    f"헛짚은 사람(FP): {len(false_positive_list):,}명 "
    f"= FP {chosen_counts['FP']:,}명."
)
