from math import isclose

import cudf
import matplotlib.pyplot as plt
from cudf import DataFrame
from cupy import arcsin, cos, float64, pi, power, radians, sin, sqrt
from geodatasets import get_path
from geopandas import geopandas
from rich import print

DATASET_PATH = "/path/to/anchors/json"
OUTPUT_PATH = "/path/to/output/json"
# POP_DISTANCE_RATIO = 1 / 375400  # 11,262,000 / 30 pop / km
POP_DISTANCE_RATIO = 0.00000266  # 11,262,000 / 30 pop / km
CIR = 40075  # apx. mean circumference of the earth in km


def haversine_cupy(
    lng1: float64, lat1: float64, lng2: float64, lat2: float64
) -> float64:
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])

    dlng = lng2 - lng1
    dlat = lat2 - lat1

    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2.0) ** 2
    c = 2 * arcsin(sqrt(a))
    km = 6367 * c
    return km


def cross_merge(
    df1: DataFrame, df2: DataFrame | None = None, suffixes=("_1", "_2")
) -> DataFrame:
    """
    Take the cartesian product of a given dataframe against another.
    (since cudf hasn't implemented crossmerge yet)

    This is done by joining the dataset on another using a dummy key.
    """
    if df2 is None:
        df2 = df1

    df1["key"] = 1
    df2["key"] = 1
    result = cudf.merge(df1, df2, on="key", how="inner")
    result = result.drop(columns=["key"])
    overlapping_colums = set(df1.columns).intersection(set(df2.columns)) - {"key"}

    # rectify column names with new suffixes
    for col in overlapping_colums:
        result = result.rename(
            columns={
                f"{col}_x": f"{col}{suffixes[0]}",
                f"{col}_y": f"{col}{suffixes[1]}",
            }
        )

    assert isinstance(result, DataFrame)
    return result


def main() -> None:
    df = cudf.read_json(DATASET_PATH)
    assert isinstance(df, DataFrame)
    # print(df.head())

    # drop any data not in the UK (for now)
    df = df[df["country"] == "United Kingdom"]
    og_len = len(df["city"])
    og_df = df
    df["population"] = df["population"].fillna(0)

    df = df[df["population"] > 100000]

    # assuming London has a population of 11,262,000 and a radius of 30km,
    # using that ratio, we can calculate the radius of each circle
    res = 11262000 * POP_DISTANCE_RATIO
    assert isclose(res, 30, rel_tol=1e-2), f"POP_DISTANCE_RATIO is incorrect got {res}"
    assert isclose(
        POP_DISTANCE_RATIO, 1 / 375400, rel_tol=1e-2
    ), f"POP_DISTANCE_RATIO is incorrect got {POP_DISTANCE_RATIO}"
    df["radius"] = df["population"].apply(lambda x: (x * POP_DISTANCE_RATIO))

    # starting with the highest population, we can merge smaller circles into larger ones if more than 50% of the smaller circle is inside the larger one
    # this is done by sorting the dataframe by population, then iterating through each row
    df = df.sort_values(by=["population"], ascending=False)
    df = df.reset_index(drop=True)

    # bounding box is (X-dX, Y-dY) to (X+dX, Y+dY) where X and Y are the lat/lng center of the circle and dX and dY are the radius
    # dY = radius * C / 360
    # dX = dY*cos(rad (Y))
    df["dY"] = df["radius"] * CIR / 360
    df["dX"] = df["dY"] * cos(radians(df["lat"]))

    df["bounding_box_min_lat"] = df["lat"] - df["dY"]
    df["bounding_box_max_lat"] = df["lat"] + df["dY"]
    df["bounding_box_min_lng"] = df["lng"] - df["dX"]
    df["bounding_box_max_lng"] = df["lng"] + df["dX"]

    # vectorize data to compute the bounding box overlaps for all pairs
    # using the cartesian product (df.merge)
    df_pairs = cross_merge(df)
    assert isinstance(df_pairs, DataFrame)

    """
    check if the bounding box of the row intersects with any of the bounding 
    boxes of the rows in the new dataframe
      - if it does, merge the two circles together and update the new dataframe
      - if it doesn't, add the row to the new dataframe
    """

    # comparing the bounding boxes first acts as a preliminary filter so we
    # don't waste time computing all the haversine distances for every circle
    # in the data set
    df_pairs = df_pairs[
        (df_pairs["bounding_box_min_lat_1"] <= df_pairs["bounding_box_max_lat_2"])
        & (df_pairs["bounding_box_max_lat_1"] >= df_pairs["bounding_box_min_lat_2"])
        & (df_pairs["bounding_box_min_lng_1"] <= df_pairs["bounding_box_max_lng_2"])
        & (df_pairs["bounding_box_max_lng_1"] >= df_pairs["bounding_box_min_lng_2"])
    ]

    df_pairs["distance"] = haversine_cupy(
        df_pairs["lng_1"], df_pairs["lat_1"], df_pairs["lng_2"], df_pairs["lat_2"]
    )

    df_pairs = df_pairs[
        df_pairs["distance"] <= (df_pairs["radius_1"] + df_pairs["radius_2"])
    ]

    """
    Calculate size of intersection
    where r1, r2 are radii

    theta = 2 * arcsin(r1 / dist)
    alpha = 2 * arcsin(r2 / dist)

    s1 = 0.5 * r1^2 * (theta - sin(theta))
    s2 = 0.5 * r2^2 * (alpha - sin(alpha))
    intersection_area = s2 + s1
    second_circle_area = pi * r2^2
    """
    df_pairs["intersection_area"] = 0.5 * power(df_pairs["radius_1"], 2) * (
        2 * arcsin(df_pairs["radius_2"] / df_pairs["distance"])
        - sin(2 * arcsin(df_pairs["radius_2"] / df_pairs["distance"]))
    ) + 0.5 * power(df_pairs["radius_2"], 2) * (
        2 * arcsin(df_pairs["radius_1"] / df_pairs["distance"])
        - sin(2 * arcsin(df_pairs["radius_1"] / df_pairs["distance"]))
    )

    df_pairs = df_pairs[
        df_pairs["intersection_area"] > (0.5 * pi * power(df_pairs["radius_2"], 2))
    ]

    """
    Merge overlapping circles, updating the population of the larger circle and
    dropping the smaller circle
    """
    df_pairs = df_pairs.sort_values(by="population_1", ascending=False)
    df["population"] = df["population"].where(
        ~df["city"].isin(df_pairs["city_2"]),
        df["population"] + df_pairs["population_2"],
    )

    df = df[~df["city"].isin(df_pairs["city_2"])]  # Drop merged circles
    df.to_json(OUTPUT_PATH, orient="records", indent=4)
    print(f"pruned to {len(df["city"])}/{og_len} rows")
    gdf = geopandas.GeoDataFrame(
        df.to_pandas(),
        geometry=geopandas.points_from_xy(df["lng"].to_pandas(), df["lat"].to_pandas()),
    )
    world = geopandas.read_file(get_path("naturalearth.land"))
    ax = world.clip([-10, 48, 2, 60]).plot(color="white", edgecolor="black")
    gdf.plot(ax=ax, color="blue")

    plt.show()
    # plt.scatter([x for x in df["lng"].to_pandas()], [y for y in df["lat"].to_pandas()])
    # plt.show()


if __name__ == "__main__":
    main()
