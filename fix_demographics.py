import os
from collections import defaultdict

import cudf
from cudf import DataFrame

IN_DIR: str = "/path/to/dataset/csv"
OUT_DIR: str = "/path/to/desired/output/dir"
SEPARATOR: str = "_"
COLS: list[str] = [ "Pop_1E", "WtNHE", "BlNHE", "AIANNHE", "AsnNHE", "NHPINHE", "Rc2plNHE", "MnTrvTmE", "MdHHIncE", "NtvE", "FbE", "Fb2E", "GeoID"]


def get_files(dir: str) -> list[str]:
    print(f"getting files from {dir}")
    files: list[str] = [
        f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))
    ]
    print(f"got {len(files)} files")
    return files


def clean_demographic_datum(
    dem_file: str,
    eco_file: str,
    soc_file: str,
    count: int,
) -> DataFrame:
    dem: DataFrame = cudf.read_csv(f"{IN_DIR}/{dem_file}")
    eco: DataFrame = cudf.read_csv(f"{IN_DIR}/{eco_file}")
    soc: DataFrame = cudf.read_csv(f"{IN_DIR}/{soc_file}")

    join: DataFrame = cudf.merge(dem, eco, on="GeoID", how="outer")
    join = cudf.merge(join, soc, on="GeoID", how="outer")
    join.set_index("GeoID")

    filter = join[[col for col in COLS if col in join.columns]]
    assert isinstance(filter, DataFrame), "filtered output is not type DataFrame"

    print(f"{count} done")
    return filter


def order_files(
    files: list[str],
) -> list[list[str]]:
    normalized: list[str] = [file.lower() for file in files]
    groups: defaultdict[str, defaultdict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )

    for file in normalized:
        prefix, identifier = file.split(SEPARATOR, 1)
        category: str = prefix[0]
        if category in {"d", "e", "s"}:
            groups[identifier][category].append(file)

    matches: list[list[str]] = []
    for identifier, categories in groups.items():
        if {"d", "e", "s"} <= categories.keys():
            matches.append([categories["d"][0], categories["e"][0], categories["s"][0]])

    return matches


def process_data():
    files: list[str] = get_files(IN_DIR)
    ordered: list[list[str]] = order_files(files)

    for tup in ordered:
        from_year = int(tup[0].split(SEPARATOR, 2)[1])
        clean_demographic_datum(tup[0], tup[1], tup[2], from_year).to_csv(
            f"{OUT_DIR}/output_{from_year}"
        )


if __name__ == "__main__":
    process_data()
