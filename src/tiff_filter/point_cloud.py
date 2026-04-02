import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import sparse
import matplotlib.pyplot as plt


class PointCloudGenerator:
    def __init__(self, location):
        self.location = location

    def load_sparse_matrix(self, volume, filtered=True):
        if filtered:
            loaded_matrix = sparse.load_npz(
                f"{self.location}/total_{volume}_sparse_filtered.npz")
        else:
            loaded_matrix = sparse.load_npz(
                f"{self.location}/total_{volume}_sparse_unfiltered.npz")

        return loaded_matrix

    def create_point_cloud(self, volume, filtered=True):
        loaded_matrix = self.load_sparse_matrix(volume, filtered)

        points = loaded_matrix.coords.T.astype(np.float64)  # faster in float64

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd

    def write_point_cloud(self, pcd, outfile_name):
        if ".pcd" in outfile_name:
            o3d.io.write_point_cloud(f"{outfile_name}", pcd)
            print(f"point cloud written to {outfile_name}")
        else:
            o3d.io.write_point_cloud(f"{outfile_name}.pcd", pcd)
            print(f"point cloud written to {outfile_name}.pcd")

    def general_denoise(
            self, pcd, nb_neighbours=10, std_ratio=1.0
    ):
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbours, std_ratio=std_ratio)
        inlier_cloud = pcd.select_by_index(ind)
        return inlier_cloud

    def find_tree(
            self, pcd, eps=12, min_points=10, print_progress=True, cluster_id=0
    ):
        labels = np.array(pcd.cluster_dbscan(
            eps=eps, min_points=min_points, print_progress=print_progress))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")

        # Change '0' to the correct cluster ID
        tree_indices = np.where(labels == cluster_id)[0]
        tree_cloud = pcd.select_by_index(tree_indices)

        return tree_cloud
