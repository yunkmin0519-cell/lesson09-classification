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
