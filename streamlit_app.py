import streamlit as st
import pandas as pd
import textwrap
import json
import os
import plotly.express as px

#############################
# 数据加载与保存视图相关函数
#############################

def load_data():
    if not os.path.exists("data.parquet"):
        st.error("数据文件 data.parquet 不存在。")
        st.stop()
    return pd.read_parquet("data.parquet")

def load_saved_views():
    if os.path.exists("saved_views.json"):
        with open("saved_views.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_view(view_name, config):
    views = load_saved_views()
    views[view_name] = config
    with open("saved_views.json", "w", encoding="utf-8") as f:
        json.dump(views, f, indent=2, ensure_ascii=False)

def load_view_callback(saved_views):
    selected_view = st.session_state.get("saved_view_select")
    if selected_view in saved_views:
        view = saved_views[selected_view]
        # 将保存的状态还原到当前 session_state
        for key in ["filter_run", "filter_step", "filter_query", "x_axis", "color_option", "facet_row_option", "facet_col_option", "agg_field", "agg_func"]:
            st.session_state[key] = view.get(key, None)

#################################
# 页面 2：聚合分析 - 视图保存/加载
#################################

def draw_view_save_load(df):
    with st.sidebar.expander("💾 保存 / 加载视图设置", expanded=True):
        saved_views = load_saved_views()
        view_names = list(saved_views.keys())

        if view_names:
            st.selectbox("选择要加载的视图:", options=view_names, key="saved_view_select")
            st.button("📂 加载视图", on_click=load_view_callback, args=(saved_views,))
        else:
            st.info("当前没有保存的视图。")

        view_name = st.text_input("输入视图名称以保存:", value="", key="view_name")
        if st.button("💾 保存当前视图"):
            config = {
                "filter_run": st.session_state.get("filter_run", sorted(df["run_name"].unique().tolist())),
                "filter_step": st.session_state.get("filter_step", sorted(df["step_num"].unique().tolist())),
                "filter_query": st.session_state.get("filter_query", sorted(df["query_index"].unique().tolist())),
                "x_axis": st.session_state.get("x_axis", "run_name"),
                "color_option": st.session_state.get("color_option", "None"),
                "facet_row_option": st.session_state.get("facet_row_option", "None"),
                "facet_col_option": st.session_state.get("facet_col_option", "None"),
                "agg_field": st.session_state.get("agg_field", "reward"),
                "agg_func": st.session_state.get("agg_func", "mean")
            }
            save_view(view_name, config)
            st.success("✅ 视图已成功保存")

#################################
# 页面 2：聚合分析 - 筛选控件
#################################

def draw_filter_widgets(df):
    with st.sidebar.expander("🔍 数据筛选", expanded=True):
        filter_run = st.multiselect("选择 run_name:", sorted(df["run_name"].unique()), key="filter_run", default=sorted(df["run_name"].unique()))
        filter_step = st.multiselect("选择 step_num:", sorted(df["step_num"].unique()), key="filter_step", default=sorted(df["step_num"].unique()))
        filter_query = st.multiselect("选择 query_index:", sorted(df["query_index"].unique()), key="filter_query", default=sorted(df["query_index"].unique()))
        return filter_run, filter_step, filter_query

#################################
# 页面 2：聚合分析 - 设置控件
#################################

def draw_aggregation_settings(df):
    with st.sidebar.expander("⚙️ 聚合设置", expanded=True):
        key_fields = ["run_name", "step_num", "query_index", "rollout_index"]
        value_fields = [col for col in df.select_dtypes(include="number").columns if col not in key_fields]

        x_axis = st.selectbox("选择 X 轴 (必选):", key_fields, key="x_axis", index=0)
        color_option = st.selectbox("选择颜色分组 (可选):", key_fields + ['None'], key="color_option", index=1)
        facet_row = st.selectbox("Facet Row (可选):", ["None"] + key_fields, key="facet_row_option")
        facet_col = st.selectbox("Facet Column (可选):", ["None"] + key_fields, key="facet_col_option")
        agg_field = st.selectbox("选择聚合字段:", value_fields, key="agg_field")
        agg_func = st.selectbox("选择聚合函数:", ["mean", "sum", "min", "max", "count"], key="agg_func")

        return x_axis, color_option, facet_row, facet_col, agg_field, agg_func

#################################
# 页面 2：聚合分析 主绘图逻辑
#################################

def draw_aggregation_analysis(df):
    st.title("📊 聚合分析")
    st.write("在聚合页面，可以对数据进行多维度的聚合和展示。")

    draw_view_save_load(df)
    filter_run, filter_step, filter_query = draw_filter_widgets(df)

    filtered_data = df[
        df["run_name"].isin(filter_run) &
        df["step_num"].isin(filter_step) &
        df["query_index"].isin(filter_query)
    ]

    st.write("### 🎯 过滤后的数据预览")
    st.dataframe(filtered_data)

    x_axis, color_option, facet_row, facet_col, agg_field, agg_func = draw_aggregation_settings(df)

    groupby_keys = [x_axis]
    for key in [color_option, facet_row, facet_col]:
        if key != "None" and key not in groupby_keys:
            groupby_keys.append(key)

    st.write("#### 当前分组键:", groupby_keys)

    if groupby_keys:
        if agg_func == "count":
            agg_df = filtered_data.groupby(groupby_keys).agg({agg_field: "count"}).reset_index()
        else:
            agg_df = filtered_data.groupby(groupby_keys).agg({agg_field: agg_func}).reset_index()
    else:
        single_value = getattr(filtered_data[agg_field], agg_func)() if agg_func != "count" else filtered_data[agg_field].count()
        agg_df = pd.DataFrame({agg_field: [single_value]})

    st.write("### 📋 聚合结果")
    st.dataframe(agg_df)

    fig = px.line(
        agg_df,
        x=x_axis,
        y=agg_field,
        color=color_option if color_option != "None" else None,
        facet_row=facet_row if facet_row != "None" else None,
        facet_col=facet_col if facet_col != "None" else None,
        markers=True
    )
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(showspikes=True, spikecolor="grey", spikethickness=1, spikedash="dot")
    st.plotly_chart(fig, use_container_width=True)

#################################
# 页面 1：Rollout 文本可视化
#################################

def save_data(df):
    df.to_parquet("data.parquet", index=False)
    st.success("✅ data.parquet 修改已保存")


def draw_rollout_visualization(df):
    st.title("📝 Rollout 文本可视化")
    st.write("你可以筛选数据、或新增计算列。")

    key_fields = ["run_name", "step_num", "query_index", "rollout_index"]
    primary_key = st.sidebar.selectbox("选择主键字段:", key_fields, key="rollout_primary_key")
    other_keys = [k for k in key_fields if k != primary_key]

    selected_primary_values = st.sidebar.multiselect(
        f"选择 {primary_key}:", ['All'] + sorted(df[primary_key].unique()), key="rollout_primary_filter"
    )
    if 'All' in selected_primary_values:
        selected_primary_values = sorted(df[primary_key].unique())

    selected_other = {}
    st.sidebar.markdown("#### 🔄 快速切换其他字段")
    for key in other_keys:
        options = sorted(df[key].unique())

        # 初始化 index
        if f"{key}_index" not in st.session_state:
            st.session_state[f"{key}_index"] = 0

        # 当前值
        current_index = st.session_state[f"{key}_index"]

        col1, col2, col3 = st.sidebar.columns([3, 0.8, 0.8], vertical_alignment="bottom")

        with col2:
            if st.button("⬅️", key=f"{key}_prev"):
                current_index = (current_index - 1) % len(options)
                st.session_state[f"{key}_index"] = current_index

        with col3:
            if st.button("➡️", key=f"{key}_next"):
                current_index = (current_index + 1) % len(options)
                st.session_state[f"{key}_index"] = current_index

        with col1:
            selected_value = st.selectbox(
                f"{key}",
                options,
                index=current_index,
                key=f"{key}_selectbox"
            )
            # 如果用户手动选择了下拉框，更新 index
            if selected_value != options[current_index]:
                st.session_state[f"{key}_index"] = options.index(selected_value)
                current_index = st.session_state[f"{key}_index"]

        selected_other[key] = options[current_index]

    if "original_df_backup" not in st.session_state:
        st.session_state["original_df_backup"] = df.copy()

    st.markdown("### ✨ 自定义列计算")
    with st.expander("🧮 使用 Python 表达式新增列"):
        new_col_name = st.text_input("新列名称:", value="new_column")
        code = st.text_area("函数代码:", "def func(row):\n    return len(row['rollout_text'])", height=120)
        if st.button("🚀 应用函数新增列"):
            try:
                local_env = {}
                exec(code, {}, local_env)
                func = local_env.get("func")
                if func is None:
                    raise ValueError("函数名必须为 func")
                df[new_col_name] = df.apply(func, axis=1)
                st.session_state["original_df_backup"] = df.copy()
                st.success(f"✅ 新列 {new_col_name} 已添加")
            except Exception as e:
                st.error(f"❌ 出错: {e}")

    with st.expander("🗑️ 删除列"):
        col_to_delete = st.multiselect("选择要删除的列:", df.columns.tolist())
        if st.button("确认删除所选列"):
            df.drop(columns=col_to_delete, inplace=True)
            st.session_state["original_df_backup"] = df.copy()
            st.success("✅ 选中列已删除")

    # 筛选
    filtered_df = df[df[primary_key].isin(selected_primary_values)]
    for key, val in selected_other.items():
        filtered_df = filtered_df[filtered_df[key] == val]

    # 对长文本字段进行手动换行处理
    if "rollout_text" in filtered_df.columns:
        filtered_df["rollout"] = filtered_df["rollout"].apply(lambda x: "\n".join(textwrap.wrap(x, width=100)))

    # 隐藏 selected_other 中所有字段
    columns_to_hide = list(selected_other.keys())
    columns_to_show = [col for col in filtered_df.columns if col not in columns_to_hide]
    filtered_df_to_display = filtered_df[columns_to_show]

    # 显示静态表格
    st.markdown("### 📊 当前筛选结果")
    st.dataframe(
        filtered_df,
        column_order=columns_to_show,
        hide_index=True,
        use_container_width=True,
        row_height=100,
        column_config={
            "rollout": st.column_config.TextColumn(
                max_chars=1000,
                help="长文本字段，使用换行显示",
                # format_func=lambda x: x.replace("\n", "<br>"),
                width=700
            )
        },
        height=1000,
    )

#################################
# 主函数入口
#################################

def main():
    if "working_df" not in st.session_state:
        df = load_data()
        st.session_state["working_df"] = df.copy()

    df = st.session_state["working_df"]
    page = st.sidebar.radio("📄 页面选择", ["Rollout 文本可视化", "聚合分析"])

    if page == "Rollout 文本可视化":
        draw_rollout_visualization(df)
    else:
        draw_aggregation_analysis(df)

if __name__ == "__main__":
    main()