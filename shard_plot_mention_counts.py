from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import streamlit as st
import plotly.express as px


DATA_CANDIDATES = [
    Path(__file__).parent / "data",  # ./data next to this script
    Path(__file__).parent,            # script folder
    Path(__file__).parent.parent / "data",  # one level up /data (fallback)
]
PIVOT_FILE = "alias_pivoted_brand_counts30.csv"
RAW_MATCHES = "reddit_matches_raw30.csv"

DEFAULT_TOPN = 5


def find_file(fname:str) -> Path:
    for base in DATA_CANDIDATES:
        p = base / fname
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {fname} in any of: {[str(b) for b in DATA_CANDIDATES]}"
    )


@st.cache_data(show_spinner=False)
def load_pivot() -> pd.DataFrame:
    p = find_file(PIVOT_FILE)
    df = pd.read_csv(p, parse_dates=['date'])
    for c in df.columns:
        if c!= 'date':
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df = df.sort_values('date').reset_index(drop=True)
    return df


def resample_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "D":
        return df.set_index("date").asfreq("D").fillna(0).reset_index()
    return (
        df.set_index('date')
        .resample("W")
        .sum()
        .reset_index()
    )


def to_long(df: pd.DataFrame, brands: List[str]) -> pd.DataFrame:
    cols = ['date'] + brands
    sub = df[cols].copy()
    long_df = sub.melt(id_vars='date', var_name='brand', value_name='mentions')
    return long_df


def default_top_brands(df_freq: pd.DataFrame, topn: int) -> List[str]:
    totals = df_freq.drop(columns=['date']).sum(axis=0).sort_values(ascending=False)
    return totals.head(topn).index.tolist()


def apply_rolling(long_df: pd.DataFrame, freq: str, win_daily=7, win_weekly=4) -> pd.DataFrame:
    win = win_daily if freq == "D" else win_weekly
    if win <= 1:
        long_df['smoothed'] = long_df['mentions']
        return long_df
    out = []
    for b, g in long_df.groupby("brand", as_index=False):
        g = g.sort_values('date').copy()
        g['smoothed'] = g['mentions'].rolling(win, min_periods=1).mean()
        out.append(g)
    return pd.concat(out, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_raw_matches_or_none() -> Optional[pd.DataFrame]:
    try:
        p = find_file(RAW_MATCHES)
    except FileNotFoundError:
        return None
    try:
        raw = pd.read_csv(p, parse_dates=['date'])
        keep = ['date', 'keyword', 'alias', 'subreddit', 'title', 'score', 'num_comments']
        cols = [c for c in keep if c in raw.columns]
        return raw[cols].copy().sort_values(['date', 'score'], ascending=[True, False])
    except Exception:
        return None


st.set_page_config(page_title="Reddit Brand Mentions (Sharded)", layout="wide")
st.title("Reddit Brand Mentions - Sharded (Interactive)")

try:
    df = load_pivot()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

all_brands = [c for c in df.columns if c != "date"]

with st.sidebar:
    st.header("Controls")
    freq_label = st.radio("Frequency", ["Daily", "Weekly"], index=0)
    freq = "D" if freq_label == "Daily" else "W"

    # We compute frequency FIRST, then pick defaults based on that windowed frame
    df_freq = resample_freq(df, freq)

    # Date range selector (based on available dates at this freq)
    min_d, max_d = df_freq["date"].min().date(), df_freq["date"].max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_d, max_d),
        min_value=min_d, max_value=max_d
    )
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = min_d, date_range

    # Filter by date first
    mask = (df_freq["date"].dt.date >= start_date) & (df_freq["date"].dt.date <= end_date)
    df_win = df_freq.loc[mask].copy()

    # Defaults: top-N brands by total mentions over filtered window
    defaults = default_top_brands(df_win, DEFAULT_TOPN) if not df_win.empty else []
    selected = st.multiselect("Brands", options=all_brands, default=defaults)

    # Rolling average toggle
    do_smooth = st.checkbox("Show rolling average", value=True)
    win_daily = st.number_input("Rolling window (days)", min_value=2, max_value=30, value=7) if (do_smooth and freq == "D") else 7
    win_weekly = st.number_input("Rolling window (weeks)", min_value=2, max_value=12, value=4) if (do_smooth and freq == "W") else 4

# Guard: no data window or no brands
if df_win.empty:
    st.info("No data in the selected date range.")
    st.stop()

if not selected:
    st.info("Pick at least one brand to plot from the sidebar.")
    st.stop()

# Long format for plotting
long_df = to_long(df_win, selected)

# Plot
if do_smooth:
    long_smooth = apply_rolling(long_df, freq=freq, win_daily=win_daily, win_weekly=win_weekly)
    fig = px.line(
        long_smooth,
        x="date", y="smoothed", color="brand",
        title=f"Brand mentions ({freq_label}) with rolling average",
        labels={"smoothed": "Mentions", "date": "Date"},
    )
    # Add faint raw lines underneath
    fig_raw = px.line(long_df, x="date", y="mentions", color="brand")
    for tr in fig_raw.data:
        tr.update(opacity=0.25, line={"width": 1})
        fig.add_trace(tr)
else:
    fig = px.line(
        long_df,
        x="date", y="mentions", color="brand",
        title=f"Brand mentions ({freq_label})",
        labels={"mentions": "Mentions", "date": "Date"},
    )

fig.update_layout(hovermode="x unified", legend_title_text="Brand")
st.plotly_chart(fig, use_container_width=True)

# Summary table
st.subheader("Summary (current window)")
summary = (
    long_df.groupby("brand", as_index=False)["mentions"].sum()
           .sort_values("mentions", ascending=False)
)
st.dataframe(summary, use_container_width=True)

# Top 10 brands by total mentions (entire dataset)
st.subheader("Top 10 brands by total mentions (entire dataset)")

totals_all = df.drop(columns=["date"]).sum(axis=0).sort_values(ascending=False)
top10_all = totals_all.head(10).reset_index()
top10_all.columns = ["brand", "total_mentions"]

fig_top10 = px.bar(
    top10_all,
    x="brand",
    y="total_mentions",
    title="Top 10 brands by total mentions (entire dataset)",
    labels={"brand": "Brand", "total_mentions": "Total mentions"},
)
st.plotly_chart(fig_top10, use_container_width=True)
st.dataframe(top10_all, use_container_width=True)

# Optional: drilldown from raw matches
raw = load_raw_matches_or_none()

with st.expander("Top subreddits for selected brands"):
    if raw is None:
        st.info("reddit_matches_raw.csv not found.")
    else:
        # Filter raw matches by the visible date window
        rmask = (raw["date"].dt.date >= start_date) & (raw["date"].dt.date <= end_date)
        rsub = raw.loc[rmask].copy()
        if "keyword" in rsub.columns:
            rsub = rsub[rsub["keyword"].isin(selected)]
        if rsub.empty:
            st.info("No raw matches for the selected brands/date range.")
        else:
            # Group by brand (keyword) and subreddit
            top_subs = (
                rsub.groupby(["keyword", "subreddit"])\
                    .size()\
                    .reset_index(name="posts")
            )
            # Show top 10 subreddits per selected brand
            for b in selected:
                sub_b = (
                    top_subs[top_subs["keyword"] == b]
                    .sort_values("posts", ascending=False)
                    .head(10)
                )
                if not sub_b.empty:
                    st.markdown(f"**{b}**")
                    st.dataframe(sub_b[["subreddit", "posts"]], use_container_width=True)

with st.expander("Show top posts (if raw matches available)"):
    if raw is None:
        st.info("reddit_matches_raw.csv not found.")
    else:
        # Filter by date window and selected brands
        rmask = (raw["date"].dt.date >= start_date) & (raw["date"].dt.date <= end_date)
        rsub = raw.loc[rmask].copy()
        if "keyword" in rsub.columns:
            rsub = rsub[rsub["keyword"].isin(selected)]
        # Top posts by score for the current window
        cols = [c for c in ["date", "keyword", "alias", "subreddit", "title", "score", "num_comments"] if c in rsub.columns]
        top_posts = rsub.sort_values("score", ascending=False)[cols].head(50)
        st.dataframe(top_posts, use_container_width=True)
