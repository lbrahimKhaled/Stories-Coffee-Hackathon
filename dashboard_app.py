from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from dash import Dash, Input, Output, State, callback, dash_table, dcc, html
    from dash.exceptions import PreventUpdate
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "dash is not installed. Run `pip install -r requirements.txt` before starting the dashboard."
    ) from exc


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "datasets" / "outputs" / "product_clustering_14" / "by_branch"
GLOBAL_OUTPUT_DIR = ROOT_DIR / "datasets" / "outputs" / "product_clustering_14" / "global"

ASSIGNMENTS_PATH = OUTPUT_DIR / "branch_product_cluster_assignments_14.csv"
PROFILES_PATH = OUTPUT_DIR / "branch_product_cluster_profiles_14.csv"
K_SCORES_PATH = OUTPUT_DIR / "branch_product_cluster_k_scores_14.csv"
GLOBAL_ASSIGNMENTS_PATH = GLOBAL_OUTPUT_DIR / "product_cluster_assignments_14.csv"
GLOBAL_PROFILES_PATH = GLOBAL_OUTPUT_DIR / "product_cluster_profiles_14.csv"
GLOBAL_K_SCORES_PATH = GLOBAL_OUTPUT_DIR / "product_cluster_k_scores_14.csv"

ALL_BRANCHES_LABEL = "All Branches (Combined)"


def assert_required_files() -> None:
    required_paths = [
        ASSIGNMENTS_PATH,
        PROFILES_PATH,
        K_SCORES_PATH,
        GLOBAL_ASSIGNMENTS_PATH,
        GLOBAL_PROFILES_PATH,
        GLOBAL_K_SCORES_PATH,
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        missing_text = "\n".join(f"- {m}" for m in missing)
        raise FileNotFoundError(
            "Missing clustering output file(s). Run `models/product_clustering_by_branch_14.py` and "
            "`models/product_clustering_14.py` first.\n"
            f"{missing_text}"
        )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments_branch = pd.read_csv(ASSIGNMENTS_PATH)
    profiles_branch = pd.read_csv(PROFILES_PATH)
    k_scores_branch = pd.read_csv(K_SCORES_PATH)

    global_assignments = pd.read_csv(GLOBAL_ASSIGNMENTS_PATH)
    global_profiles = pd.read_csv(GLOBAL_PROFILES_PATH)
    global_k_scores = pd.read_csv(GLOBAL_K_SCORES_PATH)

    global_profiles = global_profiles.copy()
    global_profiles["SuccessRank"] = (
        global_profiles["sales_share"].rank(method="dense", ascending=False).astype(int)
    )
    global_profiles["SuccessTier"] = global_profiles["ClusterLabel"].astype(str)
    global_profiles["Branch"] = ALL_BRANCHES_LABEL

    global_tier_map = global_profiles.set_index("Cluster")["SuccessTier"].to_dict()
    global_rank_map = global_profiles.set_index("Cluster")["SuccessRank"].to_dict()

    global_assignments = global_assignments.copy()
    global_assignments["Branch"] = ALL_BRANCHES_LABEL
    global_assignments["SuccessTier"] = global_assignments["Cluster"].map(global_tier_map).astype(str)
    global_assignments["SuccessRank"] = global_assignments["Cluster"].map(global_rank_map).astype(int)

    global_k_scores = global_k_scores.copy()
    global_k_scores["Branch"] = ALL_BRANCHES_LABEL

    assignments = pd.concat([assignments_branch, global_assignments], ignore_index=True, sort=False)
    profiles = pd.concat([profiles_branch, global_profiles], ignore_index=True, sort=False)
    k_scores = pd.concat([k_scores_branch, global_k_scores], ignore_index=True, sort=False)

    for df in [assignments, profiles, k_scores]:
        df["Branch"] = df["Branch"].astype(str).str.strip()

    assignments["Product Desc"] = assignments["Product Desc"].astype(str)
    profiles["SuccessTier"] = profiles["SuccessTier"].astype(str)
    assignments["SuccessTier"] = assignments["SuccessTier"].astype(str)

    return assignments, profiles, k_scores


def tier_color(tier_name: str) -> str:
    tier = tier_name.lower()
    if "most successful" in tier:
        return "#0F8A78"
    if "least successful" in tier:
        return "#B84A2A"
    if "strong" in tier:
        return "#D8973C"
    return "#5A6F57"


def format_number(value: float) -> str:
    return f"{value:,.0f}"


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def as_currencyish(value: float) -> str:
    return f"{value:,.0f}"


def tier_axis_label(tier_name: str) -> str:
    tier_name = str(tier_name).strip()
    tier_match = re.match(r"^Tier\s*(\d+)", tier_name, flags=re.IGNORECASE)
    if tier_match:
        return f"Tier {tier_match.group(1)}"

    cluster_match = re.match(r"^(C\d+)", tier_name, flags=re.IGNORECASE)
    if cluster_match:
        return cluster_match.group(1).upper()

    words = tier_name.split()
    if len(words) <= 3:
        return tier_name
    return " ".join(words[:3])


def graph_component(fig: go.Figure, height: int = 470) -> html.Div:
    return html.Div(
        className="chart-frame",
        children=[
            dcc.Graph(
                figure=fig,
                config={"displayModeBar": False, "responsive": True},
                style={"height": f"{height}px"},
            )
        ],
    )


def get_branch_slice(df: pd.DataFrame, branch: str) -> pd.DataFrame:
    return df[df["Branch"] == branch].copy()


def get_selected_k_row(k_branch: pd.DataFrame) -> pd.Series | None:
    if k_branch.empty:
        return None
    selected = k_branch[k_branch["selected"] == True]  # noqa: E712
    if selected.empty:
        selected = k_branch.sort_values("silhouette", ascending=False).head(1)
    return selected.iloc[0]


def metric_card(title: str, value: str, subtitle: str) -> html.Div:
    return html.Div(
        className="metric-card reveal",
        children=[
            html.P(title, className="metric-title"),
            html.H3(value, className="metric-value"),
            html.P(subtitle, className="metric-subtitle"),
        ],
    )


def apply_plot_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.82)",
        font=dict(family="'Sora', 'Avenir Next', sans-serif", color="#2E241A"),
        margin=dict(l=44, r=96, t=62, b=68),
        legend=dict(
            title="",
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.4)",
        ),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(gridcolor="rgba(57,41,23,0.14)", zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="rgba(57,41,23,0.14)", zeroline=False, automargin=True)
    return fig


def build_scatter_figure(assignments_branch: pd.DataFrame) -> go.Figure:
    scatter_df = assignments_branch.copy()
    scatter_df["TierColor"] = scatter_df["SuccessTier"].map(tier_color)
    tier_order = (
        scatter_df[["SuccessTier", "SuccessRank"]].drop_duplicates().sort_values("SuccessRank")["SuccessTier"].tolist()
    )
    color_map = {tier: tier_color(tier) for tier in tier_order}

    fig = px.scatter(
        scatter_df,
        x="Sales",
        y="Profit",
        color="SuccessTier",
        size="Qty",
        color_discrete_map=color_map,
        hover_name="Product Desc",
        hover_data={
            "Cluster": True,
            "MarginPct": ":.1%",
            "Sales": ":,.0f",
            "Profit": ":,.0f",
            "Qty": ":,.0f",
            "TakeAwayShare": ":.1%",
            "TableShare": ":.1%",
            "TotersShare": ":.1%",
            "SuccessTier": False,
        },
        title="Product Performance Map",
        size_max=24,
        category_orders={"SuccessTier": tier_order},
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="rgba(255,255,255,0.35)"), opacity=0.82))
    fig.update_layout(xaxis_title="Sales", yaxis_title="Profit")
    return apply_plot_theme(fig)


def build_cluster_mix_figure(profiles_branch: pd.DataFrame) -> go.Figure:
    cluster_df = profiles_branch.sort_values("SuccessRank").copy()
    cluster_df["TierColor"] = cluster_df["SuccessTier"].map(tier_color)
    cluster_df["TierAxis"] = cluster_df["SuccessTier"].map(tier_axis_label)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=cluster_df["TierAxis"],
            y=cluster_df["sales_share"],
            name="Sales Share",
            marker=dict(color=cluster_df["TierColor"]),
            hovertext=cluster_df["SuccessTier"],
            hovertemplate="%{hovertext}<br>Sales Share: %{y:.1%}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=cluster_df["TierAxis"],
            y=cluster_df["profit_share"],
            name="Profit Share",
            mode="lines+markers",
            marker=dict(color="#2E241A", size=9),
            line=dict(color="#2E241A", width=3),
            hovertext=cluster_df["SuccessTier"],
            hovertemplate="%{hovertext}<br>Profit Share: %{y:.1%}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=cluster_df["TierAxis"],
            y=cluster_df["n_products"],
            name="# Products",
            mode="lines+markers",
            marker=dict(color="#A25E2A", size=8),
            line=dict(color="#A25E2A", width=2, dash="dot"),
            hovertext=cluster_df["SuccessTier"],
            hovertemplate="%{hovertext}<br>Products: %{y}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(title="Cluster Tier Mix")
    fig.update_xaxes(title_text="Success Tier", tickangle=0)
    fig.update_yaxes(title_text="Sales/Profit Share", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(title_text="Products", secondary_y=True)
    return apply_plot_theme(fig)


def build_channel_heatmap(profiles_branch: pd.DataFrame) -> go.Figure:
    channel_df = profiles_branch.sort_values("SuccessRank").copy()
    matrix = channel_df[["avg_takeaway_share", "avg_table_share", "avg_toters_share"]].values
    labels_x = ["Take Away", "Table", "Toters"]
    labels_y = channel_df["SuccessTier"].map(tier_axis_label).tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels_x,
            y=labels_y,
            colorscale=[
                [0.0, "#F2E5CF"],
                [0.5, "#DFA56D"],
                [1.0, "#A34A2E"],
            ],
            zmin=0,
            zmax=1,
            colorbar=dict(title="Share", tickformat=".0%"),
            hovertemplate="%{y}<br>%{x}: %{z:.1%}<extra></extra>",
        )
    )
    fig.update_layout(title="Channel Dominance by Tier", xaxis_title="", yaxis_title="")
    fig.update_xaxes(tickangle=0)
    return apply_plot_theme(fig)


def build_k_quality_figure(k_branch: pd.DataFrame) -> go.Figure:
    k_df = k_branch.sort_values("k").copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=k_df["k"],
            y=k_df["silhouette"],
            mode="lines+markers",
            name="Silhouette",
            line=dict(color="#0F8A78", width=3),
            marker=dict(size=8),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=k_df["k"],
            y=k_df["min_cluster_ratio"],
            mode="lines+markers",
            name="Smallest Cluster Ratio",
            line=dict(color="#A25E2A", width=2, dash="dash"),
            marker=dict(size=7),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=k_df["k"],
            y=k_df["inertia"],
            name="Inertia",
            marker=dict(color="rgba(87,64,44,0.26)"),
        ),
        secondary_y=True,
    )

    selected = get_selected_k_row(k_df)
    if selected is not None:
        fig.add_vrect(
            x0=selected["k"] - 0.4,
            x1=selected["k"] + 0.4,
            fillcolor="rgba(15,138,120,0.15)",
            line_width=0,
            annotation_text=f"Selected k={int(selected['k'])}",
            annotation_position="top left",
        )
    fig.update_layout(title="Clustering Model Diagnostics")
    fig.update_xaxes(dtick=1, title_text="k (Number of Clusters)")
    fig.update_yaxes(title_text="Silhouette / Ratio", tickformat=".2f", secondary_y=False)
    fig.update_yaxes(title_text="Inertia", secondary_y=True)
    return apply_plot_theme(fig)


def build_top_products_bar(assignments_branch: pd.DataFrame, count: int = 12) -> go.Figure:
    top_df = assignments_branch.sort_values("Sales", ascending=False).head(count).copy()
    top_df = top_df.sort_values("Sales", ascending=True)

    fig = px.bar(
        top_df,
        x="Sales",
        y="Product Desc",
        color="SuccessTier",
        orientation="h",
        color_discrete_map={tier: tier_color(tier) for tier in top_df["SuccessTier"].unique()},
        hover_data={
            "Profit": ":,.0f",
            "MarginPct": ":.1%",
            "Qty": ":,.0f",
            "Cluster": True,
            "SuccessTier": False,
        },
        title="Top Revenue Products",
    )
    fig.update_layout(yaxis_title="", xaxis_title="Sales")
    return apply_plot_theme(fig)


def build_bottom_products_bar(assignments_branch: pd.DataFrame, count: int = 10) -> go.Figure:
    low_df = assignments_branch.sort_values("Sales", ascending=True).head(count).copy()

    fig = px.bar(
        low_df,
        x="Sales",
        y="Product Desc",
        color="SuccessTier",
        orientation="h",
        color_discrete_map={tier: tier_color(tier) for tier in low_df["SuccessTier"].unique()},
        hover_data={
            "Profit": ":,.0f",
            "MarginPct": ":.1%",
            "Qty": ":,.0f",
            "Cluster": True,
            "SuccessTier": False,
        },
        title="Least 10 Products by Sales",
    )
    fig.update_layout(yaxis_title="", xaxis_title="Sales")
    return apply_plot_theme(fig)


def build_margin_box_figure(assignments_branch: pd.DataFrame) -> go.Figure:
    df = assignments_branch.sort_values("SuccessRank").copy()
    df["TierAxis"] = df["SuccessTier"].map(tier_axis_label)
    tier_order = df[["TierAxis", "SuccessRank"]].drop_duplicates().sort_values("SuccessRank")["TierAxis"].tolist()
    tier_color_map = {
        tier_axis_label(tier): tier_color(tier)
        for tier in df["SuccessTier"].drop_duplicates().tolist()
    }

    fig = px.box(
        df,
        x="TierAxis",
        y="MarginPct",
        color="TierAxis",
        color_discrete_map=tier_color_map,
        category_orders={"TierAxis": tier_order},
        points="suspectedoutliers",
        title="Margin % Distribution by Tier",
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(tickformat=".0%", title_text="Margin %")
    fig.update_xaxes(title_text="Tier")
    return apply_plot_theme(fig)


def build_margin_scatter(assignments_branch: pd.DataFrame) -> go.Figure:
    df = assignments_branch.copy()
    color_map = {tier: tier_color(tier) for tier in df["SuccessTier"].drop_duplicates().tolist()}

    fig = px.scatter(
        df,
        x="Sales",
        y="MarginPct",
        size="Qty",
        color="SuccessTier",
        color_discrete_map=color_map,
        hover_name="Product Desc",
        hover_data={
            "Profit": ":,.0f",
            "Sales": ":,.0f",
            "MarginPct": ":.1%",
            "Cluster": True,
            "SuccessTier": False,
        },
        title="Margin % vs Sales",
        log_x=True,
        size_max=18,
    )
    fig.update_yaxes(tickformat=".0%", title_text="Margin %")
    fig.update_xaxes(title_text="Sales (log scale)")
    fig.update_traces(marker=dict(opacity=0.74, line=dict(width=0.5, color="rgba(255,255,255,0.4)")))
    return apply_plot_theme(fig)


def build_margin_histogram(assignments_branch: pd.DataFrame) -> go.Figure:
    df = assignments_branch.copy()
    fig = px.histogram(
        df,
        x="MarginPct",
        nbins=45,
        title="Margin % Histogram",
        color_discrete_sequence=["#C97A3A"],
    )
    fig.update_xaxes(tickformat=".0%", title_text="Margin %")
    fig.update_yaxes(title_text="Products")
    return apply_plot_theme(fig)


def extract_count(text: str, default: int = 5, cap: int = 10) -> int:
    match = re.search(r"\b(\d{1,2})\b", text)
    if not match:
        return default
    value = int(match.group(1))
    return max(1, min(value, cap))


def branch_summary_text(branch: str, assignments_branch: pd.DataFrame, profiles_branch: pd.DataFrame) -> str:
    if assignments_branch.empty or profiles_branch.empty:
        return f"No data available for {branch}."

    total_sales = assignments_branch["Sales"].sum()
    total_profit = assignments_branch["Profit"].sum()
    margin_pct = total_profit / total_sales if total_sales > 0 else 0.0
    top_tier = profiles_branch.sort_values("SuccessRank").iloc[0]

    if branch == ALL_BRANCHES_LABEL:
        branch_count = ASSIGNMENTS_DF[ASSIGNMENTS_DF["Branch"] != ALL_BRANCHES_LABEL]["Branch"].nunique()
        return (
            f"{branch_count} branches combined: {format_number(total_sales)} sales, "
            f"{format_number(total_profit)} profit, overall margin {format_pct(margin_pct)}. "
            f"Leading cluster: {top_tier['SuccessTier']}."
        )

    return (
        f"{branch}: {format_number(total_sales)} sales, {format_number(total_profit)} profit, "
        f"overall margin {format_pct(margin_pct)}. "
        f"Highest tier is {top_tier['SuccessTier']}."
    )


def build_insights_component(
    branch: str,
    assignments_branch: pd.DataFrame,
    profiles_branch: pd.DataFrame,
    k_branch: pd.DataFrame,
) -> html.Div:
    if assignments_branch.empty or profiles_branch.empty:
        return html.Div("No insights available.", className="insights-empty")

    best = profiles_branch.sort_values("SuccessRank").iloc[0]
    selected_k = get_selected_k_row(k_branch)
    total_sales = assignments_branch["Sales"].sum()
    total_profit = assignments_branch["Profit"].sum()
    overall_margin = total_profit / total_sales if total_sales > 0 else 0.0
    negative_profit_count = int((assignments_branch["Profit"] < 0).sum())

    channel_means = {
        "Take Away": float(assignments_branch["TakeAwayShare"].mean()),
        "Table": float(assignments_branch["TableShare"].mean()),
        "Toters": float(assignments_branch["TotersShare"].mean()),
    }
    dominant_channel = max(channel_means, key=channel_means.get)
    dominant_channel_share = channel_means[dominant_channel]

    insights = [
        f"Leading tier: {best['SuccessTier']} contributes {format_pct(best['sales_share'])} of sales and "
        f"{format_pct(best['profit_share'])} of profit.",
        f"Dominant channel: {dominant_channel} averages {format_pct(dominant_channel_share)} of product sales.",
        f"Margin watch: {negative_profit_count} products have negative profit, with portfolio margin at {format_pct(overall_margin)}.",
    ]

    if selected_k is not None:
        insights.append(
            f"Model quality: selected k={int(selected_k['k'])}, silhouette={selected_k['silhouette']:.3f}, "
            f"smallest cluster ratio={format_pct(selected_k['min_cluster_ratio'])}."
        )

    if branch == ALL_BRANCHES_LABEL:
        branch_count = ASSIGNMENTS_DF[ASSIGNMENTS_DF["Branch"] != ALL_BRANCHES_LABEL]["Branch"].nunique()
        insights.insert(0, f"Combined view across {branch_count} branches and {len(assignments_branch):,} products.")

    return html.Div(
        className="insights-block",
        children=[
            html.H4("Valuable Insights", className="insights-title"),
            html.Ul([html.Li(item) for item in insights], className="insights-list"),
        ],
    )


def answer_query(query: str, branch: str) -> str:
    q = query.lower().strip()
    assignments_branch = get_branch_slice(ASSIGNMENTS_DF, branch)
    profiles_branch = get_branch_slice(PROFILES_DF, branch).sort_values("SuccessRank")
    k_branch = get_branch_slice(K_SCORES_DF, branch)

    if assignments_branch.empty or profiles_branch.empty:
        return f"I do not have clustering data for {branch}."

    best = profiles_branch.iloc[0]
    worst = profiles_branch.iloc[-1]
    selected_k = get_selected_k_row(k_branch)

    if (("top" in q or "best" in q) and ("product" in q or "sku" in q)) or any(
        key in q for key in ["best-selling", "top selling"]
    ):
        n = extract_count(q)
        top = assignments_branch.sort_values("Sales", ascending=False).head(n)
        items = ", ".join(f"{row['Product Desc']} ({as_currencyish(row['Sales'])})" for _, row in top.iterrows())
        return f"Top {n} products in {branch}: {items}."

    if ("least" in q or "bottom" in q or "worst" in q) and ("product" in q or "sku" in q):
        n = extract_count(q)
        low = assignments_branch.sort_values("Sales", ascending=True).head(n)
        items = ", ".join(f"{row['Product Desc']} ({as_currencyish(row['Sales'])})" for _, row in low.iterrows())
        return f"Least {n} products in {branch} by sales: {items}."

    if any(key in q for key in ["best cluster", "most successful", "top tier", "strongest tier"]):
        return (
            f"{best['SuccessTier']} is leading in {branch} with {format_pct(best['sales_share'])} sales share "
            f"and {format_pct(best['profit_share'])} profit share. "
            f"Action: {best['SuggestedAction']}"
        )

    if any(key in q for key in ["least", "worst", "weak", "underperform"]):
        return (
            f"{worst['SuccessTier']} is the weakest tier in {branch}: "
            f"{format_pct(worst['sales_share'])} sales share, {format_pct(worst['profit_share'])} profit share. "
            f"Action: {worst['SuggestedAction']}"
        )

    if any(key in q for key in ["take away", "takeaway", "toters", "table", "channel"]):
        channel_cols = {
            "Take Away": "avg_takeaway_share",
            "Table": "avg_table_share",
            "Toters": "avg_toters_share",
        }
        dominant_messages = []
        for label, col in channel_cols.items():
            row = profiles_branch.sort_values(col, ascending=False).iloc[0]
            dominant_messages.append(f"{label}: {row['SuccessTier']} ({format_pct(row[col])})")
        return f"Channel leaders in {branch}: " + " | ".join(dominant_messages) + "."

    if any(key in q for key in ["k=", "selected k", "number of clusters", "cluster count", "silhouette"]):
        if selected_k is None:
            return f"No model diagnostics found for {branch}."
        return (
            f"{branch} selected k={int(selected_k['k'])} with silhouette {selected_k['silhouette']:.3f}. "
            f"Smallest-cluster ratio is {format_pct(selected_k['min_cluster_ratio'])}."
        )

    if any(key in q for key in ["summary", "overview", "snapshot"]):
        return branch_summary_text(branch, assignments_branch, profiles_branch)

    if any(key in q for key in ["margin", "profitability", "loss", "low margin"]):
        low_margin = assignments_branch.sort_values("MarginPct").head(5)
        items = ", ".join(
            f"{row['Product Desc']} ({row['MarginPct']:.1%})"
            for _, row in low_margin.iterrows()
        )
        return (
            f"{branch} overall margin is {format_pct(assignments_branch['Profit'].sum() / assignments_branch['Sales'].sum())}. "
            f"Lowest-margin products: {items}."
        )

    return (
        f"Try asking: 'top 5 products', 'most successful tier', 'weakest tier', "
        f"'channel mix', or 'selected k for {branch}'."
    )


def render_chat_messages(history: list[dict[str, str]]) -> list[html.Div]:
    messages: list[html.Div] = []
    for message in history:
        role = message.get("role", "assistant")
        text = message.get("content", "")
        messages.append(
            html.Div(
                className=f"chat-message {role}",
                children=[html.Span(text)],
            )
        )
    return messages


def build_tab_layout(branch: str, selected_tab: str) -> html.Div:
    assignments_branch = get_branch_slice(ASSIGNMENTS_DF, branch)
    profiles_branch = get_branch_slice(PROFILES_DF, branch).sort_values("SuccessRank")
    k_branch = get_branch_slice(K_SCORES_DF, branch)

    if selected_tab == "performance":
        return html.Div(
            className="grid-two",
            children=[
                dcc.Graph(figure=build_scatter_figure(assignments_branch), config={"displayModeBar": False}),
                dcc.Graph(figure=build_cluster_mix_figure(profiles_branch), config={"displayModeBar": False}),
            ],
        )

    if selected_tab == "channels":
        return html.Div(
            className="grid-two",
            children=[
                dcc.Graph(figure=build_channel_heatmap(profiles_branch), config={"displayModeBar": False}),
                dcc.Graph(figure=build_cluster_mix_figure(profiles_branch), config={"displayModeBar": False}),
            ],
        )

    if selected_tab == "diagnostics":
        return html.Div(
            className="grid-one",
            children=[dcc.Graph(figure=build_k_quality_figure(k_branch), config={"displayModeBar": False})],
        )

    if selected_tab == "products":
        top_table = (
            assignments_branch.sort_values("Sales", ascending=False)
            .head(15)[["Product Desc", "SuccessTier", "Cluster", "Sales", "Profit", "MarginPct", "Qty"]]
            .copy()
        )
        top_table["Sales"] = top_table["Sales"].map(lambda v: f"{v:,.0f}")
        top_table["Profit"] = top_table["Profit"].map(lambda v: f"{v:,.0f}")
        top_table["MarginPct"] = top_table["MarginPct"].map(lambda v: f"{v:.1%}")
        top_table["Qty"] = top_table["Qty"].map(lambda v: f"{v:,.0f}")

        return html.Div(
            className="grid-two",
            children=[
                dcc.Graph(figure=build_top_products_bar(assignments_branch), config={"displayModeBar": False}),
                html.Div(
                    className="table-wrap",
                    children=[
                        html.H4("Top 15 Products", className="table-title"),
                        dash_table.DataTable(
                            data=top_table.to_dict("records"),
                            columns=[{"name": c, "id": c} for c in top_table.columns],
                            page_size=15,
                            style_table={"overflowX": "auto"},
                            style_header={
                                "backgroundColor": "#EFDCC0",
                                "color": "#2E241A",
                                "fontWeight": "700",
                                "border": "none",
                            },
                            style_cell={
                                "backgroundColor": "rgba(255,255,255,0.72)",
                                "color": "#2E241A",
                                "padding": "10px 12px",
                                "border": "none",
                                "fontFamily": "'Manrope', 'Segoe UI', sans-serif",
                                "fontSize": "13px",
                                "textAlign": "left",
                            },
                        ),
                    ],
                ),
            ],
        )

    return html.Div(
        className="grid-two",
        children=[
            dcc.Graph(figure=build_scatter_figure(assignments_branch), config={"displayModeBar": False}),
            dcc.Graph(figure=build_channel_heatmap(profiles_branch), config={"displayModeBar": False}),
        ],
    )


assert_required_files()
ASSIGNMENTS_DF, PROFILES_DF, K_SCORES_DF = load_data()
BRANCHES = sorted(PROFILES_DF["Branch"].unique().tolist())
DEFAULT_BRANCH = BRANCHES[0]


app = Dash(__name__, title="Stories Coffee | Branch Intelligence")
server = app.server

app.layout = html.Div(
    className="app-shell",
    children=[
        dcc.Store(id="chat-open", data=False),
        dcc.Store(
            id="chat-history",
            data=[
                {
                    "role": "assistant",
                    "content": "I can answer branch-level clustering questions. Try: 'most successful tier' or 'top 5 products'.",
                }
            ],
        ),
        html.Div(
            className="app-container",
            children=[
                html.Section(
                    className="hero-card reveal",
                    children=[
                        html.P("Stories Coffee | CEO Dashboard", className="hero-kicker"),
                        html.H1("Branch Product Clustering Intelligence", className="hero-title"),
                        html.P(
                            "Switch branches and explore how product clusters behave by sales, profit, and channel mix.",
                            className="hero-subtitle",
                        ),
                    ],
                ),
                html.Section(
                    className="controls-card reveal",
                    children=[
                        html.Div(
                            className="control-row",
                            children=[
                                html.Div(
                                    className="control-block",
                                    children=[
                                        html.Label("Branch", htmlFor="branch-dropdown", className="control-label"),
                                        dcc.Dropdown(
                                            id="branch-dropdown",
                                            options=[{"label": b, "value": b} for b in BRANCHES],
                                            value=DEFAULT_BRANCH,
                                            clearable=False,
                                            searchable=True,
                                        ),
                                    ],
                                ),
                                html.Div(id="branch-summary", className="branch-summary"),
                            ],
                        ),
                        html.Div(id="kpi-grid", className="kpi-grid"),
                    ],
                ),
                html.Section(
                    className="visual-card reveal",
                    children=[
                        dcc.Tabs(
                            id="plot-tabs",
                            value="performance",
                            children=[
                                dcc.Tab(label="Performance Map", value="performance", className="plot-tab", selected_className="plot-tab--selected"),
                                dcc.Tab(label="Channel Mix", value="channels", className="plot-tab", selected_className="plot-tab--selected"),
                                dcc.Tab(label="Model Diagnostics", value="diagnostics", className="plot-tab", selected_className="plot-tab--selected"),
                                dcc.Tab(label="Top Products", value="products", className="plot-tab", selected_className="plot-tab--selected"),
                            ],
                        ),
                        html.Div(id="tab-content", className="tab-content"),
                    ],
                ),
            ],
        ),
        html.Button("AI", id="chat-toggle", n_clicks=0, className="chat-toggle"),
        html.Div(
            id="chat-panel",
            className="chat-panel closed",
            children=[
                html.Div(
                    className="chat-header",
                    children=[
                        html.H4("Dataset Copilot"),
                        html.P("Ask about the selected branch"),
                    ],
                ),
                html.Div(id="chat-messages", className="chat-messages"),
                html.Div(
                    className="chat-input-row",
                    children=[
                        dcc.Input(id="chat-input", type="text", placeholder="Ask a question...", autoComplete="off"),
                        html.Button("Send", id="chat-send", n_clicks=0),
                    ],
                ),
            ],
        ),
    ],
)


@callback(Output("branch-summary", "children"), Input("branch-dropdown", "value"))
def update_branch_summary(branch: str) -> str:
    assignments_branch = get_branch_slice(ASSIGNMENTS_DF, branch)
    profiles_branch = get_branch_slice(PROFILES_DF, branch).sort_values("SuccessRank")
    return branch_summary_text(branch, assignments_branch, profiles_branch)


@callback(Output("kpi-grid", "children"), Input("branch-dropdown", "value"))
def update_kpis(branch: str) -> list[html.Div]:
    assignments_branch = get_branch_slice(ASSIGNMENTS_DF, branch)
    profiles_branch = get_branch_slice(PROFILES_DF, branch).sort_values("SuccessRank")
    k_branch = get_branch_slice(K_SCORES_DF, branch)
    selected_k = get_selected_k_row(k_branch)

    total_sales = assignments_branch["Sales"].sum()
    total_profit = assignments_branch["Profit"].sum()
    n_products = len(assignments_branch)
    top_tier = profiles_branch.iloc[0]["SuccessTier"]
    top_tier_share = profiles_branch.iloc[0]["sales_share"]

    if selected_k is None:
        k_text = "n/a"
        sil_text = "No diagnostics"
    else:
        k_text = str(int(selected_k["k"]))
        sil_text = f"Silhouette {selected_k['silhouette']:.3f}"

    return [
        metric_card("Total Sales", format_number(total_sales), "Across all products in branch"),
        metric_card("Total Profit", format_number(total_profit), "Branch-level cluster portfolio"),
        metric_card("Clustered Products", format_number(float(n_products)), f"Top tier: {top_tier}"),
        metric_card("Selected k", k_text, sil_text),
        metric_card("Top-Tier Sales Share", format_pct(top_tier_share), "Contribution of best-performing tier"),
    ]


@callback(
    Output("tab-content", "children"),
    Input("branch-dropdown", "value"),
    Input("plot-tabs", "value"),
)
def update_tab_content(branch: str, selected_tab: str) -> html.Div:
    return build_tab_layout(branch, selected_tab)


@callback(
    Output("chat-open", "data"),
    Input("chat-toggle", "n_clicks"),
    State("chat-open", "data"),
    prevent_initial_call=True,
)
def toggle_chat_panel(_: int, is_open: bool) -> bool:
    return not bool(is_open)


@callback(
    Output("chat-panel", "className"),
    Output("chat-toggle", "children"),
    Input("chat-open", "data"),
)
def update_chat_visibility(is_open: bool) -> tuple[str, str]:
    if is_open:
        return "chat-panel open", "X"
    return "chat-panel closed", "AI"


@callback(
    Output("chat-history", "data"),
    Output("chat-input", "value"),
    Input("chat-send", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("chat-history", "data"),
    State("branch-dropdown", "value"),
    prevent_initial_call=True,
)
def process_chat_message(
    send_clicks: int,
    input_submit: int,
    user_text: str,
    chat_history: list[dict[str, str]],
    branch: str,
) -> tuple[list[dict[str, str]], str]:
    if (send_clicks is None and input_submit is None) or not user_text or not user_text.strip():
        raise PreventUpdate

    history = chat_history or []
    clean_text = user_text.strip()
    history.append({"role": "user", "content": clean_text})
    history.append({"role": "assistant", "content": answer_query(clean_text, branch)})

    if len(history) > 16:
        history = history[-16:]
    return history, ""


@callback(Output("chat-messages", "children"), Input("chat-history", "data"))
def update_chat_messages(chat_history: list[dict[str, str]]) -> list[html.Div]:
    history = chat_history or []
    return render_chat_messages(history)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
