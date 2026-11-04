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
