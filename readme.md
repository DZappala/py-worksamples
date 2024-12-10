# Work Samples

## Samples \#1 and \#2

Samples 1 and 2 share a similar context as part of a larger project to map non-arbitrary administrative subdivisions around the globe. Most administrative subdivisions are drawn with biases. These ulterior motives typically overlook the natural ways in which people move and interact with each other throughout the world. For example, gerrymandering is a mapping technique that divides neighborhoods of cities with a political bias. Gerrymanderers will demarcate voting districts by hand-picking city blocks to increase the likelihood of their candidate getting a majority vote. This enables scenarios where voters in one district might oppose the construction of a new bridge across the street but have no way to vote out the elected official who greenlighted the project. What would a map look like were we to draw boundaries based on the physical and human-geographical attributes of our world?

### Dataset

Samples 1 and 2 share a dataset called "Geonames," a set of about 25M records of cities around the globe with a population > 100. Prior to sample 1, I pared the dataset down, eliminating smaller cities to make the dataset more manageable. However, sample 2 employs a more effective algorithm and is able to use the entire dataset.

### Sample 1: Circle Compress

This algorithm combines multiple cities, represented by a point, into a single record by looking for each city's nearest neighbors with a population less than some threshold. Think of each point as a gravity well, pulling in nearby cities and merging them into itself, its gravitational pull falling off logarithmically. To compute this, I process records in batches on the GPU with Nvidia CUDA via Python APIs. These libraries are familiar with respcect to their iterative counterparts, but were chosen specifically for GPU optimization.

The libraries used include:

- cuDF, a GPU-optimized replacement of pandas.
- CuPy, a GPU-optimized replacement for SciPy and NumPy.
- cuML, a library of general machine learning algorithms.
- cuSpatial, a GPU-optimized replacement for GeoPandas and scipy.spatial.

**Relevant files:**

- [circle_compress.py](circle_compress.py)
- [anchors.json](anchors.json)

### Sample 2: Hierarchical clustering

Sample 2 uses the entire geonames dataset by employing the DBSCAN algorithm. DBSCAN is a hierarchical agglomerative clustering algorithm whose weights can be tuned by various scalar values; it functions similarly to k-nearest-neighbors. Similar to Circle Compress, I use population size to weight the output. DBSCAN outputs a list of labels, which we can associate with their respective point clouds. Having these labels is nice; it lets us visualize them with Matplotlib, but for a more performant, interactive experience, we can create plane meshes from the resultant point clouds—much like a connect-the-dots picture—and view them using QGIS (an open-source mapping software).

However, planes don't require ~100 vertices; we can get rid of any points inside the plane and keep only the edge vertices that define the bounds of the mesh. This might be easy to do by looking at the shape—just connect all the perimeter dots—but doing 30,000+ connect-the-dot puzzles would take too much time and isn't fun. However, given a list of points, a computer has no way of knowing which ones are inside the plane and which ones define the outer edge. We can instead solve for the points' convex hull or alpha shape. To simplify, this process involves solving a Delaunay triangulation and iterating to find the outside edge of each triangle. To speed things along, I again pipe all the computations through the GPU. I run the Delaunay triangulation using PyTorch and solve the alpha shape by Edelsbrunner's algorithm as outlined in a paper from Stanford called, "Introduction to Alpha Shapes.".

Libraries Used:

- Matplotlib
- cuSpatial
- Torch, torch_delaunay
- GeoPandas
- NumPy
- CuPy
- cuML

**Relevant files:**

- [hierarchical2.py](hierarchical2.py)
- [geonames.geojson](data/geonames.geojson)

## Sample 3: Data cleaning

The final sample is an example of some work I needed to do for my Database Management Systems final project. The project required us to find a dataset and upload it to a database. For this project, I parsed data from the Census Bureau's American Community Survey for each year over the last 10 years and Citibike bikeshare trip data. Fix_demographics.py is one of several scripts I used to amalgamate the census data, which was distributed across multiple csv files. All told, the data set consisted of +230M records; by using GPU multi-processing, I was able to process all the records in about 15 seconds.

Libraries Used:
- cuDF

**Relevant Files:**

- [fix_demographics.py](fix_demographics.py)
- [demo_2021_nta.csv](data/demo_2021_nta.csv)
- [econ_2021_nta.csv](data/econ_2021_nta.csv)
- [soc_2021_nta.csv](data/soc_2021_nta.csv)
