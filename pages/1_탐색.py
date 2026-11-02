python
import streamlit as st
import pandas as pd
import plotly.express as px


# ==========================================
# 1. 페이지 기본 설정
# ==========================================

st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)


# ==========================================
# 2. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    """첫 화면과 동일한 주소에서 UTF-8로 데이터를 읽는다."""

    data = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    return data


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.code(str(e))
    st.stop()


# ==========================================
# 3. 페이지 제목
# ==========================================

st.title("🔎 데이터 탐색")

st.markdown(
    """
    뇌졸중 데이터를 다양한 그래프와 표로 살펴봅니다.

    나이, 평균 혈당, 고혈압, 심장병, 체질량지수,
    흡연 상태와 뇌졸중의 관계를 탐색해 봅시다.
    """
)

st.divider()


# ==========================================
# 4. 나이와 평균 혈당 분포
# ==========================================

st.header("1. 나이와 평균 혈당의 분포")

st.markdown(
    """
    전체 사람들의 나이와 평균 혈당이
    어떤 형태로 분포하는지 히스토그램으로 확인합니다.
    """
)

col1, col2 = st.columns(2)

with col1:

    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이의 분포",
        labels={
            "age": "나이",
            "count": "사람 수"
        },
        color_discrete_sequence=["#3498DB"]
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.1
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True
    )


with col2:

    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당의 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        },
        color_discrete_sequence=["#E67E22"]
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.1
    )

    st.plotly_chart(
        fig_glucose,
        use_container_width=True
    )


st.divider()


# ==========================================
# 5. 뇌졸중 경험 유무에 따른 상자그림
# ==========================================

st.header("2. 뇌졸중 경험 유무에 따른 비교")

st.markdown(
    """
    뇌졸중을 경험한 사람과 경험하지 않은 사람의
    나이와 평균 혈당 분포를 상자그림으로 비교합니다.
    """
)

# 그룹 이름 만들기
df["stroke_group"] = df["stroke"].map({
    0: "뇌졸중 경험 없음",
    1: "뇌졸중 경험 있음"
})

stroke_order = [
    "뇌졸중 경험 없음",
    "뇌졸중 경험 있음"
]

col1, col2 = st.columns(2)

with col1:

    fig_age_box = px.box(
        df,
        x="stroke_group",
        y="age",
        color="stroke_group",
        category_orders={
            "stroke_group": stroke_order
        },
        title="뇌졸중 유무에 따른 나이 비교",
        labels={
            "stroke_group": "뇌졸중 경험",
            "age": "나이"
        },
        color_discrete_map={
            "뇌졸중 경험 없음": "#3498DB",
            "뇌졸중 경험 있음": "#E74C3C"
        }
    )

    fig_age_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 경험",
        yaxis_title="나이"
    )

    st.plotly_chart(
        fig_age_box,
        use_container_width=True
    )


with col2:

    fig_glucose_box = px.box(
        df,
        x="stroke_group",
        y="avg_glucose_level",
        color="stroke_group",
        category_orders={
            "stroke_group": stroke_order
        },
        title="뇌졸중 유무에 따른 평균 혈당 비교",
        labels={
            "stroke_group": "뇌졸중 경험",
            "avg_glucose_level": "평균 혈당"
        },
        color_discrete_map={
            "뇌졸중 경험 없음": "#3498DB",
            "뇌졸중 경험 있음": "#E74C3C"
        }
    )

    fig_glucose_box.update_layout(
        showlegend=False,
        xaxis_title="뇌졸중 경험",
        yaxis_title="평균 혈당"
    )

    st.plotly_chart(
        fig_glucose_box,
        use_container_width=True
    )


# 그룹별 평균값 계산
mean_table = (
    df.groupby("stroke_group", observed=False)
    .agg(
        나이_평균=("age", "mean"),
        평균혈당=("avg_glucose_level", "mean"),
        사람수=("stroke", "count")
    )
    .reindex(stroke_order)
    .reset_index()
)

mean_table = mean_table.rename(
    columns={
        "stroke_group": "뇌졸중 경험"
    }
)

mean_table["나이_평균"] = mean_table["나이_평균"].round(2)
mean_table["평균혈당"] = mean_table["평균혈당"].round(2)

mean_table = mean_table.rename(
    columns={
        "나이_평균": "평균 나이",
        "평균혈당": "평균 혈당",
        "사람수": "사람 수"
    }
)

st.subheader("그룹별 평균값")

st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ==========================================
# 6. 고혈압 유무에 따른 뇌졸중 비율
# ==========================================

st.header("3. 고혈압과 심장병에 따른 뇌졸중 비율")

st.markdown(
    """
    고혈압이나 심장병이 있는 그룹과 없는 그룹에서
    뇌졸중을 경험한 사람의 비율을 비교합니다.
    """
)


def make_stroke_rate_table(data, column, labels):
    """특정 질환 유무에 따른 뇌졸중 비율을 계산한다."""

    result = (
        data.groupby(column, dropna=False)
        .agg(
            전체_사람수=("stroke", "size"),
            뇌졸중_경험자수=("stroke", "sum")
        )
        .reset_index()
    )

    result["뇌졸중 비율"] = (
        result["뇌졸중_경험자수"]
        / result["전체_사람수"]
        * 100
    )

    result["구분"] = result[column].map(labels)

    return result


# ------------------------------------------
# 고혈압
# ------------------------------------------

hypertension_labels = {
    0: "고혈압 없음",
    1: "고혈압 있음"
}

hypertension_table = make_stroke_rate_table(
    df,
    "hypertension",
    hypertension_labels
)

fig_hypertension = px.bar(
    hypertension_table,
    x="구분",
    y="뇌졸중 비율",
    text="뇌졸중 비율",
    title="고혈압 유무에 따른 뇌졸중 비율",
    labels={
        "구분": "고혈압 여부",
        "뇌졸중 비율": "뇌졸중 비율(%)"
    },
    color="구분",
    color_discrete_map={
        "고혈압 없음": "#3498DB",
        "고혈압 있음": "#E74C3C"
    }
)

fig_hypertension.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_hypertension.update_layout(
    showlegend=False,
    yaxis_title="뇌졸중 비율(%)",
    xaxis_title="고혈압 여부"
)

st.plotly_chart(
    fig_hypertension,
    use_container_width=True
)


# ------------------------------------------
# 심장병
# ------------------------------------------

heart_labels = {
    0: "심장병 없음",
    1: "심장병 있음"
}

heart_table = make_stroke_rate_table(
    df,
    "heart_disease",
    heart_labels
)

fig_heart = px.bar(
    heart_table,
    x="구분",
    y="뇌졸중 비율",
    text="뇌졸중 비율",
    title="심장병 유무에 따른 뇌졸중 비율",
    labels={
        "구분": "심장병 여부",
        "뇌졸중 비율": "뇌졸중 비율(%)"
    },
    color="구분",
    color_discrete_map={
        "심장병 없음": "#3498DB",
        "심장병 있음": "#E74C3C"
    }
)

fig_heart.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig_heart.update_layout(
    showlegend=False,
    yaxis_title="뇌졸중 비율(%)",
    xaxis_title="심장병 여부"
)

st.plotly_chart(
    fig_heart,
    use_container_width=True
)


st.divider()


# ==========================================
# 7. BMI 결측값과 뇌졸중 비율
# ==========================================

st.header("4. BMI 결측값과 뇌졸중 비율")

st.markdown(
    """
    체질량지수(BMI)가 비어 있는 사람들만 따로 살펴보고,
    전체 사람들의 뇌졸중 비율과 비교합니다.
    """
)

# BMI 결측자 추출
missing_bmi = df[df["bmi"].isna()]

# 전체 인원 및 뇌졸중 경험자
total_count = len(df)
total_stroke_count = int(df["stroke"].sum())

# BMI 결측 인원 및 뇌졸중 경험자
missing_count = len(missing_bmi)
missing_stroke_count = int(missing_bmi["stroke"].sum())

# 비율 계산
total_stroke_rate = (
    total_stroke_count / total_count * 100
    if total_count > 0 else 0
)

missing_stroke_rate = (
    missing_stroke_count / missing_count * 100
    if missing_count > 0 else 0
)

bmi_table = pd.DataFrame({
    "구분": [
        "BMI가 비어 있는 사람",
        "전체 사람"
    ],
    "사람 수": [
        missing_count,
        total_count
    ],
    "뇌졸중 경험자 수": [
        missing_stroke_count,
        total_stroke_count
    ],
    "뇌졸중 비율(%)": [
        round(missing_stroke_rate, 2),
        round(total_stroke_rate, 2)
    ]
})

st.dataframe(
    bmi_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ==========================================
# 8. 흡연 상태별 사람 수
# ==========================================

st.header("5. 흡연 상태별 사람 수")

st.markdown(
    """
    흡연 상태의 종류마다 몇 명의 사람이 포함되어 있는지
    확인합니다.
    """
)

smoking_table = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

# 결측값을 알아보기 쉽게 표시
smoking_table["흡연 상태"] = (
    smoking_table["흡연 상태"]
    .fillna("결측값")
)

st.dataframe(
    smoking_table,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 9. 페이지 하단
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:gray; padding:15px;">
        🔎 뇌졸중 예측 실습실 | 데이터 탐색<br>
        데이터를 관찰하고 특징을 발견해 봅시다.
    </div>
    """,
    unsafe_allow_html=True
)
