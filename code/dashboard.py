from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.title("LPBF Layer Viewer")

raw_dir = Path("/path/to/file/").expanduser()
coordinates_file = Path(__file__).with_name("file")
mpm_partstats = Path("/path/to/file").expanduser() / "file"
circle_radius = 75
quality_threshold = 2.01e4
quality_window = 11
quality_min_completion = 0.10
ignored_part_names = {"QC 1", "QC 2", "QC 3", "QC 4"}


def find_column(df, candidates):
    normalized = {str(col).strip().lower().replace(" ", "_"): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def part_name_key(part_name):
    value = str(part_name).strip()
    if value.isdigit():
        value = value.strip("0") or "0"
    return value.upper()


def read_table(path):
    return pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)


def quality_cell_style(value):
    styles = {
        "Pass": (
            "background-color: #dcfce7; color: #166534; font-weight: 700; "
            "border-radius: 6px; text-align: center;"
        ),
        "Fail": (
            "background-color: #fee2e2; color: #991b1b; font-weight: 700; "
            "border-radius: 6px; text-align: center;"
        ),
        "Not determined yet": (
            "background-color: #e0f2fe; color: #075985; font-weight: 600; "
            "border-radius: 6px; text-align: center;"
        ),
        "No data yet": (
            "background-color: #f1f5f9; color: #475569; font-weight: 600; "
            "border-radius: 6px; text-align: center;"
        ),
    }
    return styles.get(value, "color: #475569; font-weight: 600; text-align: center;")


def part_quality(partstats_path, build_completion, determination_layer):
    if partstats_path is None:
        return pd.DataFrame(), {}

    stats = read_table(partstats_path)
    part_col = find_column(stats, ["PART_NAME", "part_name", "part name"])
    layer_col = find_column(stats, ["LAYER_NUMBER", "layer_number", "layer number"])
    mean_col = find_column(stats, ["MEAN", "mean"])

    if part_col is None or layer_col is None or mean_col is None:
        return pd.DataFrame(), {}

    stats = stats[[part_col, layer_col, mean_col]].copy()
    stats[part_col] = stats[part_col].astype(str).str.strip()
    stats = stats[~stats[part_col].map(part_name_key).isin(ignored_part_names)]

    all_parts = sorted(stats[part_col].drop_duplicates())
    if build_completion < quality_min_completion:
        quality = pd.DataFrame(
            {
                "Part name": all_parts,
                "Quality": ["Not determined yet"] * len(all_parts),
            }
        )
        return quality, dict(zip(quality["Part name"].map(part_name_key), quality["Quality"]))

    stats[layer_col] = pd.to_numeric(stats[layer_col], errors="coerce")
    stats[mean_col] = pd.to_numeric(stats[mean_col], errors="coerce")
    stats = stats.dropna(subset=[layer_col, mean_col])
    stats = stats[stats[layer_col] <= determination_layer]

    if stats.empty:
        quality = pd.DataFrame(
            {
                "Part name": all_parts,
                "Quality": ["No data yet"] * len(all_parts),
            }
        )
        return quality, dict(zip(quality["Part name"].map(part_name_key), quality["Quality"]))

    stats = stats.sort_values([part_col, layer_col])
    stats["MEAN moving average"] = stats.groupby(part_col)[mean_col].transform(
        lambda values: values.rolling(quality_window, min_periods=quality_window).mean()
    )
    latest = stats.dropna(subset=["MEAN moving average"]).groupby(part_col, as_index=False).tail(1)

    quality = latest[[part_col, layer_col, "MEAN moving average"]].copy()
    quality["Quality"] = np.where(
        quality["MEAN moving average"] > quality_threshold,
        "Pass",
        "Fail",
    )
    quality = quality.rename(
        columns={
            part_col: "Part name",
            layer_col: "Latest layer",
        }
    )
    quality["MEAN moving average"] = quality["MEAN moving average"].round(2)

    missing_parts = sorted(set(all_parts) - set(quality["Part name"]))
    if missing_parts:
        quality = pd.concat(
            [
                quality,
                pd.DataFrame(
                    {
                        "Part name": missing_parts,
                        "Latest layer": [np.nan] * len(missing_parts),
                        "MEAN moving average": [np.nan] * len(missing_parts),
                        "Quality": ["No data yet"] * len(missing_parts),
                    }
                ),
            ],
            ignore_index=True,
        )

    quality = quality.sort_values("Part name")

    return quality, dict(zip(quality["Part name"].map(part_name_key), quality["Quality"]))

files = [
    f"SI0201448420250909082402_{i:05d}_{i * 60:07_d}.h5"
    for i in range(1, 1009)
]

st.sidebar.header("Layer control")

if "layer_idx" not in st.session_state:
    st.session_state.layer_idx = 0

n_files = len(files)

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("⬅ Previous"):
        st.session_state.layer_idx = max(0, st.session_state.layer_idx - 1)

with col2:
    if st.button("Next ➡"):
        st.session_state.layer_idx = min(n_files - 1, st.session_state.layer_idx + 1)

st.session_state.layer_idx = st.sidebar.slider(
    "Layer slider",
    min_value=0,
    max_value=n_files - 1,
    value=st.session_state.layer_idx,
)

st.session_state.layer_idx = st.sidebar.number_input(
    "Layer number",
    min_value=0,
    max_value=n_files - 1,
    value=st.session_state.layer_idx,
    step=1,
)

raw_file = raw_dir / files[st.session_state.layer_idx]
current_layer_number = st.session_state.layer_idx + 1
build_completion = current_layer_number / n_files
quality_determination_layer = max(1, int(np.ceil(n_files * quality_min_completion)))

st.sidebar.write(f"Index: `{st.session_state.layer_idx}`")
st.sidebar.write(f"Build completion: `{build_completion:.1%}`")

with h5py.File(raw_file, "r") as f:
    data16 = f["MeasurementData/Channels/0/DataUint16"][:].astype(np.uint16)
    data8 = f["MeasurementData/Channels/0/DataUint8"][:].astype(np.uint8)

i = data16[:, 0].astype(np.float32)
x = data16[:, 1].astype(np.int32)
y = data16[:, 2].astype(np.int32)
b = data8[:, 0].astype(np.float32)

bitmap = np.zeros((y.max() + 1, x.max() + 1), dtype=np.float32)
bitmap[y, x] = i * b

quality, quality_lookup = part_quality(
    mpm_partstats,
    build_completion,
    quality_determination_layer,
)

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=bitmap,
        colorscale="Hot",
        showscale=False,
        hoverinfo="skip",
    )
)

if coordinates_file.exists():
    coordinates = pd.read_csv(coordinates_file, dtype=str, keep_default_na=False)

    part_col = find_column(coordinates, ["part", "part_name", "name", "part name"])
    x_col = find_column(coordinates, ["x", "center_x", "centre_x", "circle_x", "circle centre x"])
    y_col = find_column(coordinates, ["y", "center_y", "centre_y", "circle_y", "circle centre y"])

    if part_col is None or x_col is None or y_col is None:
        st.warning(
            "Could not find part/x/y columns in coordinates.csv. "
            "Expected columns like part, x, y or part_name, center_x, center_y."
        )
    else:
        theta = np.linspace(0, 2 * np.pi, 100)

        for _, row in coordinates.iterrows():
            center_x = float(row[x_col])
            center_y = float(row[y_col])
            part_name = str(row[part_col])
            quality_status = quality_lookup.get(part_name_key(part_name), "Not applicable")
            fill_color = {
                "Pass": "rgba(22, 163, 74, 0.30)",
                "Fail": "rgba(220, 38, 38, 0.32)",
                "Not determined yet": "rgba(14, 165, 233, 0.16)",
                "No data yet": "rgba(148, 163, 184, 0.16)",
                "Not applicable": "rgba(148, 163, 184, 0.16)",
            }.get(quality_status, "rgba(0, 255, 255, 0.12)")
            hover_label_color = {
                "Pass": "#15803d",
                "Fail": "#b91c1c",
                "Not determined yet": "#0369a1",
                "No data yet": "#475569",
                "Not applicable": "#475569",
            }.get(quality_status, "#475569")
            hover_text = f"<b>{part_name}</b><br><span style='font-size:12px'>Quality</span>: <b>{quality_status}</b>"

            fig.add_trace(
                go.Scatter(
                    x=center_x + circle_radius * np.cos(theta),
                    y=center_y + circle_radius * np.sin(theta),
                    mode="lines",
                    fill="toself",
                    hoveron="fills",
                    name=hover_text,
                    text=[hover_text] * len(theta),
                    hovertext=[hover_text] * len(theta),
                    hoverinfo="text",
                    hovertemplate="%{fullData.name}<extra></extra>",
                    line=dict(color="rgba(0, 0, 0, 0)", width=0),
                    fillcolor=fill_color,
                    hoverlabel=dict(
                        bgcolor=hover_label_color,
                        bordercolor="rgba(255, 255, 255, 0.35)",
                        font=dict(color="white", size=14),
                    ),
                    showlegend=False,
                )
            )
else:
    st.warning("coordinates.csv was not found.")

fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, visible=False)
fig.update_xaxes(visible=False)
fig.update_layout(
    width=900,
    height=720,
    margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor="black",
    paper_bgcolor="black",
    hovermode="closest",
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Part quality")
if mpm_partstats is None:
    st.warning("No mpm_partstats file found next to dashboard.py.")
elif quality.empty:
    st.warning("Could not read PART_NAME, LAYER_NUMBER, and MEAN from the mpm_partstats file.")
else:
    if build_completion < quality_min_completion:
        st.info("Pass/fail is not determined until the build reaches 10% completion.")
    else:
        st.info(f"Pass/fail is fixed from layer {quality_determination_layer}, the 10% build checkpoint.")

    pass_count = (quality["Quality"] == "Pass").sum()
    fail_count = (quality["Quality"] == "Fail").sum()
    not_ready_count = quality["Quality"].isin(["Not determined yet", "No data yet"]).sum()

    metric_cols = st.columns(3)
    metric_cols[0].metric("Pass", int(pass_count))
    metric_cols[1].metric("Fail", int(fail_count))
    metric_cols[2].metric("Not determined", int(not_ready_count))

    display_quality = quality[["Part name", "Quality"]].copy()
    styled_quality = (
        display_quality.style.apply(
            lambda column: [quality_cell_style(value) for value in column],
            subset=["Quality"],
        )
        .set_properties(subset=["Part name"], **{"font-weight": "500"})
        .set_properties(subset=["Quality"], **{"text-align": "center"})
    )
    st.dataframe(styled_quality, use_container_width=True, hide_index=True)
