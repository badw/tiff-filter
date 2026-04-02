import rasterio
import numpy as np
import warnings
import glob
import tqdm_pathos
import natsort

import sparse
from scipy.ndimage import label, sum as scipy_sum
from scipy import ndimage
import copy
import tqdm
import os
from collections import defaultdict


class TiffFilter:

    def __init__(self, tiff_path, out_path):
        warnings.simplefilter("ignore")

        self.tiff_files = sorted(
            glob.glob(f"{tiff_path}/*.tiff"), key=natsort.natsort_key)
        self.out_path = out_path

    @staticmethod
    def filter_tiff(tiff_array: np.array, threshold: float, null_value=0):
        return np.where(tiff_array <= threshold, tiff_array, null_value)

    def remove_large_structures(self, islice, footprint=(50, 50), area_threshold=5000):
        mask = islice > 0
        struct = np.ones(footprint)
        dilated_mask = ndimage.binary_dilation(
            mask, structure=struct, iterations=1
        )

        labeled_array, num_features = label(dilated_mask)
        sizes = scipy_sum(mask, labeled_array, range(num_features + 1))

        remove_ids = np.where(sizes > area_threshold)[0]
        remove_ids = remove_ids[remove_ids != 0]

        remove_mask = np.isin(labeled_array, remove_ids)
        islice[remove_mask] = 0
        return (islice)
        # sparse.save_npz(f"./sparse_filtered_arrays/{index}.npz",sparse.COO.from_numpy(islice))

    def multiprocessing_function(self, file, threshold=0.8, footprint=(50, 50), area_threshold=5000):
        img = rasterio.open(file)
        arr = img.read()[0]
        arr = self.filter_tiff(arr, threshold*np.max(arr), 0)
        arr[arr > 0] = 1
        outfilename = file.replace(".tiff", ".npz").split("/")[1]
        # save for comparison to extra filtering
        sparse_arr = sparse.COO.from_numpy(1-arr)
        sparse.save_npz(
            f"{self.out_path}/sparse_unfiltered_{outfilename}", sparse_arr)

        # np.savez_compressed(f"{self.out_path}/unfiltered_{outfilename}",1-arr)
        # extra filtering
        extra_filter = self.remove_large_structures(
            islice=1-arr, footprint=footprint, area_threshold=area_threshold)
        sparse.save_npz(f"{self.out_path}/sparse_filtered_{outfilename}",
                        sparse.COO.from_numpy(extra_filter))

    def run(self, n_cpus=4, threshold=0.8, footprint=(50, 50), area_threshold=5000):
        tqdm_pathos.map(
            self.multiprocessing_function,
            self.tiff_files,
            threshold=threshold,
            footprint=footprint,
            area_threshold=area_threshold,
            n_cpus=n_cpus
        )

    def split_files(self, files: list, filtered=True) -> defaultdict:
        """
        split a list of files into a dictionary with {scan_no:{slice_no:filename.npz}}    

        RETURNS
        dict
        """
        datadict = defaultdict(dict)
        for file in files:
            if filtered:
                if "sparse_filtered" in file:
                    loc1 = int([x for x in file.split("_")
                               if "pag" in x][0].replace("pag", ""))

                    loc2 = int([x for x in file.split("_")
                               if ".npz" in x][0].replace(".npz", ""))

                    datadict[loc1][loc2] = file

            else:
                if "sparse_unfiltered" in file:
                    loc1 = int([x for x in file.split("_")
                               if "pag" in x][0].replace("pag", ""))
                    loc2 = int([x for x in file.split("_")
                               if ".npz" in x][0].replace(".npz", ""))

                    datadict[loc1][loc2] = file

              # np.load(file)["arr_0"][0]
        self.datadict = datadict

    def join_data(self, volume=0, filtered=True):
        files = sorted(glob.glob(f"{self.out_path}/*.npz"),
                       key=natsort.natsort.natsort_key)
        self.split_files(files=files, filtered=filtered)
        stacked = sparse.stack([sparse.load_npz(file)
                               for i, file in self.datadict[volume].items()])
        if filtered:
            sparse.save_npz(f"total_{volume}_sparse_filtered.npz", stacked)
        else:
            sparse.save_npz(f"total_{volume}_sparse_unfiltered.npz", stacked)
