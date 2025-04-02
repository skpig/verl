import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px

#############################
# 数据加载与保存视图相关函数
#############################

def load_data():
    if not os.path.exists("data.json"):
        st.error("数据文件 data.json 不存在。")
        st.stop()
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def load_saved_views():
    if os.path.exists("saved_views.json"):
        with open("saved_views.json", "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {}

def save_view(view_name, config):
    views = load_saved_views()
    views[view_name] = config
    with open("saved_views.json", "w", encoding="utf-8") as f:
        json.dump(views, f, indent=2)

def load_view_callback(saved_views):
    selected_view = st.session_state.get("saved_view_select")
    if selected_view in saved_views:
        view = saved_views[selected_view]
        st.session_state["filter_run"] = view["filter_run"]
        st.session_state["filter_step"] = view["filter_step"]
        st.session_state["filter_query"] = view["filter_query"]
        st.session_state["x_axis"] = view["x_axis"]
        st.session_state["color_option"] = view["color_option"]
        st.session_state["facet_row_option"] = view["facet_row_option"]
        st.session_state["facet_col_option"] = view["facet_col_option"]
        st.session_state["agg_field"] = view["agg_field"]
        st.session_state["agg_func"] = view["agg_func"]

#################################
# 页面 2（聚合分析）辅助函数
#################################

def draw_view_save_load(df):
    st.sidebar.header("保存/加载视图")
    saved_views = load_saved_views()
    view_names = list(saved_views.keys())
    if view_names:
        st.sidebar.selectbox("选择要加载的视图", options=view_names, key="saved_view_select")
        st.sidebar.button("加载视图", on_click=load_view_callback, args=(saved_views,))
    else:
        st.sidebar.info("当前没有保存的视图。")
    
    view_name = st.sidebar.text_input("输入视图名称", value="", key="view_name")
    if st.sidebar.button("保存视图"):
        # 构造当前视图配置
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
        save_view(st.session_state.view_name, config)
        st.sidebar.success("视图保存成功！")

def draw_filter_widgets(df):
    filter_run = st.sidebar.multiselect(
        "选择 run_name", 
        sorted(df["run_name"].unique().tolist()),
        default=st.session_state.get("filter_run", sorted(df["run_name"].unique().tolist())),
        key="filter_run"
    )
    filter_step = st.sidebar.multiselect(
        "选择 step_num", 
        sorted(df["step_num"].unique().tolist()),
        default=st.session_state.get("filter_step", sorted(df["step_num"].unique().tolist())),
        key="filter_step"
    )
    filter_query = st.sidebar.multiselect(
        "选择 query_index", 
        sorted(df["query_index"].unique().tolist()),
        default=st.session_state.get("filter_query", sorted(df["query_index"].unique().tolist())),
        key="filter_query"
    )
    return filter_run, filter_step, filter_query

def draw_aggregation_settings(df):
    available_keys = ["run_name", "step_num", "query_index", "rollout_index"]
    # only keey numeric columns as value keys
    value_keys = df.select_dtypes(include=["number"]).columns.tolist()
    value_keys = [key for key in value_keys if key not in available_keys]
    x_axis = st.sidebar.selectbox(
        "选择 X 轴 (必选)", 
        options=available_keys, 
        index=0, 
        key="x_axis"
    )
    color_option = st.sidebar.selectbox(
        "选择线条颜色 (可选)", 
        options=available_keys, 
        index=1, 
        key="color_option"
    )
    facet_row_option = st.sidebar.selectbox(
        "选择 Facet Row (可选)", 
        options=["None"] + available_keys, 
        index=0, 
        key="facet_row_option"
    )
    facet_col_option = st.sidebar.selectbox(
        "选择 Facet Column (可选)", 
        options=["None"] + available_keys, 
        index=0, 
        key="facet_col_option"
    )
    agg_field = st.sidebar.selectbox(
        "选择聚合字段", 
        options=value_keys, 
        index=0, 
        key="agg_field"
    )
    agg_func = st.sidebar.selectbox(
        "选择聚合函数", 
        options=["mean", "sum", "min", "max", "count"], 
        index=0, 
        key="agg_func"
    )
    return x_axis, color_option, facet_row_option, facet_col_option, agg_field, agg_func

#################################
# 页面 1：Rollout 文本可视化
#################################

def save_data(df):
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)
    st.success("data.json 修改已保存")

def draw_rollout_visualization(df):
    st.title("Rollout 文本可视化 & 可编辑")
    st.write("在此页面你可以编辑基础数据库中的项目，并可以通过自定义 apply 操作新增列。")


    keys_list = ["run_name", "step_num", "query_index"]
    primary_key = st.sidebar.selectbox("选择主键(Multi-choice)", keys_list, index=0, key="rollout_primary_key")
    other_keys = [k for k in keys_list if k != primary_key]

    primary_options = sorted(df[primary_key].unique().tolist())
    selected_primary = st.sidebar.multiselect(
        f"选择 {primary_key}", 
        primary_options, 
        default=primary_options, 
        key="rollout_primary_filter"
    )

    selected_other = {}
    st.sidebar.markdown("#### 非主键快速切换")
    for key in other_keys:
        if key not in st.session_state:
            st.session_state[key] = 0
        options = sorted(df[key].unique().tolist())
        col1, col2, col3 = st.sidebar.columns([1, 2, 1])
        with col1:
            if st.button("⬅️", key=f"{key}_prev_rollout"):
                st.session_state[key] = (st.session_state[key] - 1) % len(options)
        with col2:
            st.write(f"**{key}**: `{options[st.session_state[key]]}`")
        with col3:
            if st.button("➡️", key=f"{key}_next_rollout"):
                st.session_state[key] = (st.session_state[key] + 1) % len(options)
        selected_other[key] = options[st.session_state[key]]

    # 原始备份用于撤销
    if "original_df_backup" not in st.session_state:
        st.session_state["original_df_backup"] = df.copy()

    st.write("### 编辑或增加列")

    with st.expander("🆕 自定义 apply 表达式新增列"):
        st.markdown("**你可以输入一个完整的 Python 函数定义，用于对每一行生成新列**")
        new_col_name = st.text_input("新列名", value="new_column")
        code = st.text_area(
            "输入函数代码 (例如: `def func(row): return len(row['rollout_text'])`)",
            height=150,
            value="def func(row):\n    return len(row['rollout_text'])"
        )
        apply_btn = st.button("应用表达式新增列")

        if apply_btn:
            try:
                local_env = {}
                exec(code, {}, local_env)
                func = local_env.get("func")
                if func is None:
                    raise ValueError("未定义名为 'func' 的函数")
                df[new_col_name] = df.apply(func, axis=1)
                st.session_state["original_df_backup"] = df.copy()
                st.success(f"新列 `{new_col_name}` 已添加到整个数据集")
            except Exception as e:
                st.error(f"无法应用函数表达式：{e}")

    with st.expander("🗑️ 删除列"):
        col_to_delete = st.multiselect("选择要删除的列", options=df.columns.tolist())
        if st.button("确认删除所选列"):
            df.drop(columns=col_to_delete, inplace=True)
            st.session_state["original_df_backup"] = df.copy()
            st.success("选中列已从整个数据集中删除")

    # 过滤应在此之后进行，以包含新增列/删除列
    filtered_df = df[df[primary_key].isin(selected_primary)]
    for key, value in selected_other.items():
        filtered_df = filtered_df[filtered_df[key] == value]

    # 可编辑表格
    edited_df = st.data_editor(filtered_df, num_rows="dynamic", use_container_width=True, key="editable_df")

    # 提交修改
    col_save, col_reset = st.columns([1, 1])
    with col_save:
        if st.button(":inbox_tray: 保存修改到 data.json"):
            df.update(edited_df)
            save_data(df)

    with col_reset:
        if st.button("↩️ 撤销所有未保存修改"):
            st.session_state["working_df"] = st.session_state["original_df_backup"].copy()
#################################
# 页面 2：聚合分析
#################################

def draw_aggregation_analysis(df):
    st.title("聚合分析")
    st.write("在聚合页面，可以对数据进行多维度的聚合和展示。")
    
    # 先绘制“保存/加载视图”区域（确保在后续控件生成之前）
    draw_view_save_load(df)
    
    # 绘制过滤器
    filter_run, filter_step, filter_query = draw_filter_widgets(df)
    filtered_data = df[
        (df["run_name"].isin(filter_run)) &
        (df["step_num"].isin(filter_step)) &
        (df["query_index"].isin(filter_query))
    ]
    st.write("### 过滤后的数据预览")
    st.dataframe(filtered_data)
    
    # 绘制聚合设置
    x_axis, color_option, facet_row_option, facet_col_option, agg_field, agg_func = draw_aggregation_settings(df)
    
    # 自动构建分组键
    groupby_keys = [x_axis]
    if color_option != "None" and color_option not in groupby_keys:
        groupby_keys.append(color_option)
    if facet_row_option != "None" and facet_row_option not in groupby_keys:
        groupby_keys.append(facet_row_option)
    if facet_col_option != "None" and facet_col_option not in groupby_keys:
        groupby_keys.append(facet_col_option)
    st.write("#### 当前分组键：", groupby_keys)
    
    # 进行聚合计算
    if groupby_keys:
        if agg_func == "count":
            agg_df = filtered_data.groupby(groupby_keys).agg({agg_field: "count"}).reset_index()
        else:
            agg_df = filtered_data.groupby(groupby_keys).agg({agg_field: agg_func}).reset_index()
    else:
        if agg_func == "count":
            agg_value = filtered_data[agg_field].count()
        else:
            agg_value = getattr(filtered_data[agg_field], agg_func)()
        agg_df = pd.DataFrame({agg_field: [agg_value]})
    
    st.write("### 聚合结果表")
    st.dataframe(agg_df)
    
    # 绘制折线图（开启统一 hover 模式与 spike 线效果）
    st.sidebar.header("图表设置")
    fig = px.line(
        agg_df,
        x=x_axis,
        y=agg_field,
        color=color_option if color_option != "None" else None,
        facet_row=facet_row_option if facet_row_option != "None" else None,
        facet_col=facet_col_option if facet_col_option != "None" else None,
        markers=True
    )
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(showspikes=True, spikecolor="grey", spikethickness=1, spikedash="dot")
    st.plotly_chart(fig)

#############################
# 主函数
#############################

def main():
    # 初始化 session_state 中的 working_df
    if "working_df" not in st.session_state:
        df = load_data()
        st.session_state["working_df"] = df.copy()
    df = st.session_state["working_df"]
    page = st.sidebar.radio("选择页面", ["Rollout 文本可视化", "聚合分析"])
    if page == "Rollout 文本可视化":
        draw_rollout_visualization(df)
    else:
        draw_aggregation_analysis(df)

if __name__ == "__main__":
    main()
