import cuml
import cupy
import cuspatial
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy
from cuml.manifold import TSNE
from scipy.spatial import ConvexHull
from torch import Tensor
from torch_delaunay.functional import shull2d

DO_TSNE: bool = False
DO_DBSCAN: bool = True
PLOT_DBSCAN: bool = False


def alpha_shape(S, R):
    """
    Compute the alpha shape of a point set S given its simplices.
    Simplices can be obtained through a Delaunay Triangulation.

    Args:
        S (cupy.ndarray): A set of points in R^d.
        simplices (cp.ndarray): Array of simplices from the Delaunay triangulation.

    Returns:
        tuple: (R, B, I)
            R (cp.ndarray): List of simplices (the input simplices).
            B (dict): Dictionary mapping each simplex to its interval (a, b).
            I (dict): Dictionary mapping each simplex to its interval (b, ∞).

    Source: https://graphics.stanford.edu/courses/cs268-11-spring/handouts/AlphaShapes/as_fisher.pdf
            Page 12
    """
    d = S.shape[1] - 1  # Dimension of space (d-simplex)

    # Initialize boundary (B) and interior (I) intervals for each simplex
    B: dict = {tuple(s): (cupy.inf, cupy.inf) for s in R}
    I: dict = {tuple(s): (0, cupy.inf) for s in R}

    # Loop through dimensions from d-1 to 0
    for k in range(d - 1, -1, -1):
        # Filter simplices to the current dimension k
        k_simplex = [tuple(s) for s in R if len(s) == k + 1]

        for T in k_simplex:
            # Determine a: If B[T] is empty, use circumradius; otherwise, find min(B[U])
            if B[T] == (cupy.inf, cupy.inf):  # If no boundary interval
                a = compute_circumradius(S[T])
            else:
                a = min(
                    B[U][0] for U in find_neighboring_simplices(T, R, k + 1)
                )  # Min of neighbors' intervals

            # Determine b: If T is on the convex hull, set b = inf; otherwise, find max(B[U])
            if is_on_convex_hull(T, S):  # Check if T is on the convex hull
                b = cupy.inf
            else:
                b = max(
                    B[U][0] for U in find_neighboring_simplices(T, R, d)
                )  # Max of neighbors' intervals

            # Update intervals for T
            B[T] = (a, b)
            I[T] = (b, cupy.inf)

    return R, B, I


def compute_circumradius(simplex_points):
    """
    Compute the circumradius of a simplex on the GPU.

    Args:
        simplex_points (cp.ndarray): Points forming the simplex, shape (n, d).

    Returns:
        float: The circumradius of the simplex.

    Source(s): Guibas & Stolfi https://dl.acm.org/doi/pdf/10.1145/282918.282923
               https://ianthehenry.com/posts/delaunay/
    """
    n, d = simplex_points.shape
    centroid = simplex_points.mean(axis=0)
    dist_squared = cupy.sum((simplex_points - centroid) ** 2, axis=1)
    circumradius = cupy.sqrt(cupy.sum(dist_squared) / (n * d))
    return circumradius


def find_neighboring_simplices(T, simplices, dim):
    """
    Find neighboring simplices of the given simplex T in dimension `dim`.
    """
    return [U for U in simplices if len(set(U).intersection(T)) == dim]


def is_on_convex_hull(T, points):
    """
    Determine if a simplex T is on the convex hull of the points.

    Args:
        T (tuple): Simplex to check (tuple of point indices).
        points (cp.ndarray): Points forming the dataset, shape (n, d).

    Returns:
        bool: True if T is on the convex hull, False otherwise.
    """
    hull = ConvexHull(cupy.asnumpy(points))
    for simplex in hull.simplices:
        if set(T).issubset(simplex):
            return True
    return False


def main():
    gdf = gpd.read_file("/path/to/geonames/dataset", engine="pyogrio")
    gdf.assign(cluster=0)

    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y

    gdf = cuspatial.from_geopandas(gdf)
    # print(gdf.head())
    xy = gdf[["x", "y"]].to_cupy()
    # print(xy)

    if DO_TSNE:
        tsne = TSNE(
            n_components=2,
            method="fft",
            metric="euclidean",
            n_neighbors=90,
            perplexity=30,
        )
        x_hat = tsne.fit_transform(xy)
        plt.scatter(x_hat[:, 0].get(), x_hat[:, 1].get(), c=xy.get()[:, 1], s=0.5)
        plt.show()

    if DO_DBSCAN:
        dbscan = cuml.DBSCAN(metric="euclidean", output_type="numpy")
        weights = gdf["population"].to_cupy()
        gdf["label"] = dbscan.fit_predict(xy, out_dtype="int32", sample_weight=weights)
        if PLOT_DBSCAN:
            plt.scatter(
                gdf["x"].to_numpy(),
                gdf["y"].to_numpy(),
                c=gdf["label"].to_numpy(),
                cmap="gist_rainbow",
            )
            plt.show()
        arr = (gdf[["x", "y"]]).to_numpy()
        lab = gdf["label"].to_numpy()
        unique_lab = numpy.unique(gdf["label"].to_numpy())
        point_dict = {label: arr[lab == label] for label in unique_lab}
        for _, points in point_dict.items():
            if len(points) < 3:
                continue
            simplices = shull2d(points=Tensor(points)).numpy()
            R, B, I = alpha_shape(S=points, R=simplices)
            print("List of simplices (R):")
            print(R)
            print("\nBoundary intervals (B):")
            for simplex, interval in B.items():
                print(f"Simplex {simplex}: Interval {interval}")
            print("\nInterior intervals (I):")
            for simplex, interval in I.items():
                print(f"Simplex {simplex}: Interval {interval}")

    return


if __name__ == "__main__":
    main()
