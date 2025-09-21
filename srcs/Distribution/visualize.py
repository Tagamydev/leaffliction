import polars as pl
import plotly.express as px


def visualize_dataset(csv_path: str) -> None:
    """
    Load dataset CSV and generate histogram + pie chart.
    """
    df = pl.read_csv(csv_path)
    pdf = df.to_pandas()

    # Histogram
    fig = px.histogram(
        pdf, x="class", text_auto=True, color="class",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.show()

    # Pie chart
    fig = px.pie(
        pdf, names="class", color="class",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.show()

