#%%


from pathlib import Path
import polars as pl

def list_images_to_csv(root: str) -> pl.DataFrame:
    exts = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff",".gif"}
    rows = []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            rows.append({
                "path": str(p.resolve()),
                "name": p.name,   
                "class": p.parent.name,
                "stem": p.stem,
                "group": p.stem.split("_")[0]
            })
    
    return pl.from_records(rows)
# %%

def train_test_val(df:pl.DataFrame)->pl.DataFrame:
    TRAIN, VAL = (0.7,0.9)

    df2 = df.select(["class","group"]).unique().sample(fraction=1,shuffle=True)
    df3 = df2.with_columns(
        idx = pl.cum_count("group").over("class"),
        tot = pl.n_unique("group").over("class"),
    )
    df4 = df3.with_columns(
        prop = pl.col("idx")/pl.col("tot")

    )
    df5 = df4.with_columns(
        split = ( pl.when(pl.col("prop") <TRAIN).then(pl.lit("train"))
                 .when(pl.col("prop") <VAL).then(pl.lit("val"))
                 .otherwise(pl.lit("test"))

        )

    )
    df5 = df5.drop(["idx", "tot","prop","class"])
    return df5



#%%
x=list_images_to_csv(".")

y = train_test_val(x)
print(y)

# %%

z = x.join(y,on=["group"])
print(z)

#%%

import plotly.express as px

pdf = z.to_pandas()

#%%
fig = px.histogram(pdf, x="class", text_auto=True, color = "class",
                   color_discrete_sequence=px.colors.qualitative.Set3)
fig.show()
#%%


fig = px.pie(pdf, names="class", color = "class",
                   color_discrete_sequence=px.colors.qualitative.Set3)
fig.show()
# %%
