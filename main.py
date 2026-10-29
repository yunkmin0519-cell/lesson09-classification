
import json
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "yunkmin0519-cell/data/refs/heads/main/stroke.csv"
)

BASE_DIR = Path(__file__).resolve().parent
RECORD_DIR = BASE_DIR / "내_기록"
RECORD_FILE = RECORD_DIR / "기록.json"

COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]


# --------------------------------------------------
# 기록 파일 읽기 및 저장
# --------------------------------------------------
def load_record():
    """이전에 저장한 우리말 뜻과 데이터 출처를 읽습니다."""
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}

    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            record = json.load(file)

        return {
            "meanings": record.get("meanings", {}),
            "source": record.get("source", ""),
        }
    except (json.JSONDecodeError, OSError):
        return {"meanings": {}, "source": ""}


def save_record(meanings, source):
    """우리말 뜻과 데이터 출처를 JSON 파일에 저장합니다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "meanings": meanings,
        "source": source,
    }

    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    """GitHub에서 CSV를 UTF-8로 읽어옵니다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as error:
    st.set_page_config(
        page_title="뇌졸중 예측 실습실",
        page_icon="🧠",
        layout="wide",
    )
    st.error("데이터를 불러오지 못했습니다.")
    st.write("인터넷 연결과 데이터 URL을 확인해 주세요.")
    st.caption(f"오류 내용: {error}")
    st.stop()


# 필수 열 확인
missing_columns = [col for col in COLUMNS if col not in df.columns]

if missing_columns:
    st.error(f"CSV에서 필요한 열을 찾을 수 없습니다: {missing_columns}")
    st.stop()


# --------------------------------------------------
# 기존 기록 불러오기
# --------------------------------------------------
record = load_record()
saved_meanings = record.get("meanings", {})
saved_source = record.get("source", "")


# --------------------------------------------------
# 화면 상단
# --------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.markdown(
    """
    뇌졸중 관련 데이터를 살펴보고, 각 변수의 의미와 데이터의
    구조를 이해하는 학습 공간입니다.

    아래 표에서 열 이름의 우리말 뜻을 직접 작성하고,
    데이터의 특징과 빈 값의 분포를 확인해 보세요.
    """
)

st.divider()


# --------------------------------------------------
# 데이터 요약
# --------------------------------------------------
total_people = len(df)
total_columns = len(df.columns)
stroke_count = int(
    pd.to_numeric(df["stroke"], errors="coerce").eq(1).sum()
)

stroke_rate = (
    stroke_count / total_people * 100
    if total_people > 0
    else 0
)

card1, card2, card3, card4 = st.columns(4)

with card1:
    st.metric(
        label="👥 전체 사람 수",
        value=f"{total_people:,}명",
    )

with card2:
    st.metric(
        label="📋 전체 열 개수",
        value=f"{total_columns}개",
    )

with card3:
    st.metric(
        label="🩺 뇌졸중 발생 인원",
        value=f"{stroke_count:,}명",
    )

with card4:
    st.metric(
        label="📊 뇌졸중 발생 비율",
        value=f"{stroke_rate:.2f}%",
    )

st.caption(
    "뇌졸중 발생 비율은 전체 행 중 stroke 값이 1인 행의 비율입니다."
)

st.divider()


# --------------------------------------------------
# 열 이름과 우리말 뜻, 값의 종류, 빈 값 개수
# --------------------------------------------------
st.subheader("📖 데이터 열 살펴보기")

st.write(
    "우리말 뜻 열을 직접 수정하세요. "
    "값의 종류와 빈 값 개수는 원본 데이터에서 자동으로 계산됩니다."
)

column_info = []

for col in df.columns:
    unique_count = int(df[col].nunique(dropna=True))

    if pd.api.types.is_numeric_dtype(df[col]):
        value_kind = f"숫자형 · 고유값 {unique_count:,}종"
    else:
        value_kind = f"문자형 · 고유값 {unique_count:,}종"

    missing_count = int(df[col].isna().sum())

    column_info.append(
        {
            "열 이름": col,
            "우리말 뜻": saved_meanings.get(col, ""),
            "값의 종류": value_kind,
            "빈 값 개수": missing_count,
        }
    )

column_df = pd.DataFrame(column_info)

edited_df = st.data_editor(
    column_df,
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    key="column_meanings_editor",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름",
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고하여 각 열의 뜻을 입력하세요.",
            width="large",
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류",
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            format="%d",
        ),
    },
)

# 편집된 우리말 뜻을 가져옵니다.
current_meanings = {}

for _, row in edited_df.iterrows():
    current_meanings[str(row["열 이름"])] = str(
        row["우리말 뜻"]
        if pd.notna(row["우리말 뜻"])
        else ""
    )

st.caption(
    "빈 값 개수는 실제 결측값(NaN)의 수입니다. "
    "문자열 'Unknown'은 빈 값으로 계산하지 않습니다."
)

st.divider()


# --------------------------------------------------
# 데이터 미리보기
# --------------------------------------------------
st.subheader("🔎 데이터 미리보기")

st.write("원본 CSV의 처음 다섯 행입니다.")

st.dataframe(
    df.head(5),
    hide_index=True,
    use_container_width=True,
)

st.divider()


# --------------------------------------------------
# 데이터 출처 입력
# --------------------------------------------------
st.subheader("🔗 데이터 출처")

st.write(
    "교재에 있는 데이터 출처를 입력하세요. "
    "입력한 내용은 아래 저장 버튼으로 보관할 수 있습니다."
)

st.markdown(f"[원본 CSV 열기]({DATA_URL})")

source_text = st.text_area(
    "데이터 출처 기록",
    value=saved_source,
    placeholder="교재에 제시된 데이터 출처를 입력하세요.",
    height=120,
    key="source_text_editor",
)

# --------------------------------------------------
# 기록 저장
# --------------------------------------------------
if st.button("💾 우리말 뜻과 출처 저장", type="primary"):
    try:
        save_record(current_meanings, source_text)
        st.success("기록을 저장했습니다. 새로고침 후에도 다시 불러옵니다.")
    except OSError as error:
        st.error(f"기록을 저장하지 못했습니다: {error}")

st.caption(
    "저장 위치: 내_기록/기록.json"
)
