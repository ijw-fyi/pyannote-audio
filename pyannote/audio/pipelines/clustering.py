# The MIT License (MIT)
#
# Copyright (c) 2021- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Clustering pipelines"""


import random
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from einops import rearrange
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Categorical, Integer, Uniform
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import umap
import hdbscan

from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import oracle_segmentation
from pyannote.audio.utils.permutation import permutate


class BaseClustering(Pipeline):
    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = 1000,
        constrained_assignment: bool = False,
    ):
        super().__init__()
        self.metric = metric
        self.max_num_embeddings = max_num_embeddings
        self.constrained_assignment = constrained_assignment

    def set_num_clusters(
        self,
        num_embeddings: int,
        num_clusters: Optional[int] = None,
        min_clusters: Optional[int] = None,
        max_clusters: Optional[int] = None,
    ):
        min_clusters = num_clusters or min_clusters or 1
        min_clusters = max(1, min(num_embeddings, min_clusters))
        max_clusters = num_clusters or max_clusters or num_embeddings
        max_clusters = max(1, min(num_embeddings, max_clusters))

        if min_clusters > max_clusters:
            raise ValueError(
                f"min_clusters must be smaller than (or equal to) max_clusters "
                f"(here: min_clusters={min_clusters:g} and max_clusters={max_clusters:g})."
            )

        if min_clusters == max_clusters:
            num_clusters = min_clusters

        return num_clusters, min_clusters, max_clusters

    def filter_embeddings(
        self,
        embeddings: np.ndarray,
        segmentations: Optional[SlidingWindowFeature] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter NaN embeddings and downsample embeddings

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.

        Returns
        -------
        filtered_embeddings : (num_embeddings, dimension) array
        chunk_idx : (num_embeddings, ) array
        speaker_idx : (num_embeddings, ) array
        """

        # whether speaker is active
        active = np.sum(segmentations.data, axis=1) > 0
        # whether speaker embedding extraction went fine
        valid = ~np.any(np.isnan(embeddings), axis=2)

        # indices of embeddings that are both active and valid
        chunk_idx, speaker_idx = np.where(active * valid)

        # sample max_num_embeddings embeddings
        num_embeddings = len(chunk_idx)
        if num_embeddings > self.max_num_embeddings:
            indices = list(range(num_embeddings))
            random.shuffle(indices)
            indices = sorted(indices[: self.max_num_embeddings])
            chunk_idx = chunk_idx[indices]
            speaker_idx = speaker_idx[indices]

        return embeddings[chunk_idx, speaker_idx], chunk_idx, speaker_idx

    def constrained_argmax(self, soft_clusters: np.ndarray) -> np.ndarray:
        soft_clusters = np.nan_to_num(soft_clusters, nan=np.nanmin(soft_clusters))
        num_chunks, num_speakers, num_clusters = soft_clusters.shape
        # num_chunks, num_speakers, num_clusters

        hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)

        for c, cost in enumerate(soft_clusters):
            speakers, clusters = linear_sum_assignment(cost, maximize=True)
            for s, k in zip(speakers, clusters):
                hard_clusters[c, s] = k

        return hard_clusters

    def assign_embeddings(
        self,
        embeddings: np.ndarray,
        train_chunk_idx: np.ndarray,
        train_speaker_idx: np.ndarray,
        train_clusters: np.ndarray,
        constrained: bool = False,
    ):
        """Assign embeddings to the closest centroid

        Cluster centroids are computed as the average of the train embeddings
        previously assigned to them.

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension)-shaped array
            Complete set of embeddings.
        train_chunk_idx : (num_embeddings,)-shaped array
        train_speaker_idx : (num_embeddings,)-shaped array
            Indices of subset of embeddings used for "training".
        train_clusters : (num_embedding,)-shaped array
            Clusters of the above subset
        constrained : bool, optional
            Use constrained_argmax, instead of (default) argmax.

        Returns
        -------
        soft_clusters : (num_chunks, num_speakers, num_clusters)-shaped array
        hard_clusters : (num_chunks, num_speakers)-shaped array
        centroids : (num_clusters, dimension)-shaped array
            Clusters centroids
        """

        # TODO: option to add a new (dummy) cluster in case num_clusters < max(frame_speaker_count)

        num_clusters = np.max(train_clusters) + 1
        num_chunks, num_speakers, dimension = embeddings.shape

        train_embeddings = embeddings[train_chunk_idx, train_speaker_idx]

        centroids = np.vstack(
            [
                np.mean(train_embeddings[train_clusters == k], axis=0)
                for k in range(num_clusters)
            ]
        )

        # compute distance between embeddings and clusters
        e2k_distance = rearrange(
            cdist(
                rearrange(embeddings, "c s d -> (c s) d"),
                centroids,
                metric=self.metric,
            ),
            "(c s) k -> c s k",
            c=num_chunks,
            s=num_speakers,
        )
        soft_clusters = 2 - e2k_distance

        # assign each embedding to the cluster with the most similar centroid
        if constrained:
            hard_clusters = self.constrained_argmax(soft_clusters)
        else:
            hard_clusters = np.argmax(soft_clusters, axis=2)

        # NOTE: train_embeddings might be reassigned to a different cluster
        # in the process. based on experiments, this seems to lead to better
        # results than sticking to the original assignment.

        return hard_clusters, soft_clusters, centroids

    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: Optional[SlidingWindowFeature] = None,
        num_clusters: Optional[int] = None,
        min_clusters: Optional[int] = None,
        max_clusters: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply clustering

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.
        num_clusters : int, optional
            Number of clusters, when known. Default behavior is to use
            internal threshold hyper-parameter to decide on the number
            of clusters.
        min_clusters : int, optional
            Minimum number of clusters. Has no effect when `num_clusters` is provided.
        max_clusters : int, optional
            Maximum number of clusters. Has no effect when `num_clusters` is provided.

        Returns
        -------
        hard_clusters : (num_chunks, num_speakers) array
            Hard cluster assignment (hard_clusters[c, s] = k means that sth speaker
            of cth chunk is assigned to kth cluster)
        soft_clusters : (num_chunks, num_speakers, num_clusters) array
            Soft cluster assignment (the higher soft_clusters[c, s, k], the most likely
            the sth speaker of cth chunk belongs to kth cluster)
        centroids : (num_clusters, dimension) array
            Centroid vectors of each cluster
        """

        train_embeddings, train_chunk_idx, train_speaker_idx = self.filter_embeddings(
            embeddings,
            segmentations=segmentations,
        )

        num_embeddings, _ = train_embeddings.shape

        num_clusters, min_clusters, max_clusters = self.set_num_clusters(
            num_embeddings,
            num_clusters=num_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )

        if max_clusters < 2:
            # do NOT apply clustering when min_clusters = max_clusters = 1
            num_chunks, num_speakers, _ = embeddings.shape
            hard_clusters = np.zeros((num_chunks, num_speakers), dtype=np.int8)
            soft_clusters = np.ones((num_chunks, num_speakers, 1))
            centroids = np.mean(train_embeddings, axis=0, keepdims=True)
            return hard_clusters, soft_clusters, centroids

        train_clusters = self.cluster(
            train_embeddings,
            min_clusters,
            max_clusters,
            num_clusters=num_clusters,
        )

        hard_clusters, soft_clusters, centroids = self.assign_embeddings(
            embeddings,
            train_chunk_idx,
            train_speaker_idx,
            train_clusters,
            constrained=self.constrained_assignment,
        )

        return hard_clusters, soft_clusters, centroids


class AgglomerativeClustering(BaseClustering):
    """Agglomerative clustering

    Parameters
    ----------
    metric : {"cosine", "euclidean", ...}, optional
        Distance metric to use. Defaults to "cosine".

    Hyper-parameters
    ----------------
    method : {"average", "centroid", "complete", "median", "single", "ward"}
        Linkage method.
    threshold : float in range [0.0, 2.0]
        Clustering threshold.
    min_cluster_size : int in range [1, 20]
        Minimum cluster size
    """

    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = np.inf,
        constrained_assignment: bool = False,
    ):
        super().__init__(
            metric=metric,
            max_num_embeddings=max_num_embeddings,
            constrained_assignment=constrained_assignment,
        )

        self.threshold = Uniform(0.0, 2.0)  # assume unit-normalized embeddings
        self.method = Categorical(
            ["average", "centroid", "complete", "median", "single", "ward", "weighted"]
        )

        # minimum cluster size
        self.min_cluster_size = Integer(1, 20)

    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        num_clusters: Optional[int] = None,
    ):
        """

        Parameters
        ----------
        embeddings : (num_embeddings, dimension) array
            Embeddings
        min_clusters : int
            Minimum number of clusters
        max_clusters : int
            Maximum number of clusters
        num_clusters : int, optional
            Actual number of clusters. Default behavior is to estimate it based
            on values provided for `min_clusters`,  `max_clusters`, and `threshold`.

        Returns
        -------
        clusters : (num_embeddings, ) array
            0-indexed cluster indices.
        """

        num_embeddings, _ = embeddings.shape

        # heuristic to reduce self.min_cluster_size when num_embeddings is very small
        # (0.1 value is kind of arbitrary, though)
        min_cluster_size = min(
            self.min_cluster_size, max(1, round(0.1 * num_embeddings))
        )

        # linkage function will complain when there is just one embedding to cluster
        if num_embeddings == 1:
            return np.zeros((1,), dtype=np.uint8)

        # centroid, median, and Ward method only support "euclidean" metric
        # therefore we unit-normalize embeddings to somehow make them "euclidean"
        if self.metric == "cosine" and self.method in ["centroid", "median", "ward"]:
            with np.errstate(divide="ignore", invalid="ignore"):
                embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric="euclidean"
            )

        # other methods work just fine with any metric
        else:
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric=self.metric
            )

        # apply the predefined threshold
        clusters = fcluster(dendrogram, self.threshold, criterion="distance") - 1

        # split clusters into two categories based on their number of items:
        # large clusters vs. small clusters
        cluster_unique, cluster_counts = np.unique(
            clusters,
            return_counts=True,
        )
        large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
        num_large_clusters = len(large_clusters)

        # force num_clusters to min_clusters in case the actual number is too small
        if num_large_clusters < min_clusters:
            num_clusters = min_clusters

        # force num_clusters to max_clusters in case the actual number is too large
        elif num_large_clusters > max_clusters:
            num_clusters = max_clusters

        # look for perfect candidate if necessary
        if num_clusters is not None and num_large_clusters != num_clusters:
            # switch stopping criterion from "inter-cluster distance" stopping to "iteration index"
            _dendrogram = np.copy(dendrogram)
            _dendrogram[:, 2] = np.arange(num_embeddings - 1)

            best_iteration = num_embeddings - 1
            best_num_large_clusters = 1

            # traverse the dendrogram by going further and further away
            # from the "optimal" threshold

            for iteration in np.argsort(np.abs(dendrogram[:, 2] - self.threshold)):
                # only consider iterations that might have resulted
                # in changing the number of (large) clusters
                new_cluster_size = _dendrogram[iteration, 3]
                if new_cluster_size < min_cluster_size:
                    continue

                # estimate number of large clusters at considered iteration
                clusters = fcluster(_dendrogram, iteration, criterion="distance") - 1
                cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
                large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
                num_large_clusters = len(large_clusters)

                # keep track of iteration that leads to the number of large clusters
                # as close as possible to the target number of clusters.
                if abs(num_large_clusters - num_clusters) < abs(
                    best_num_large_clusters - num_clusters
                ):
                    best_iteration = iteration
                    best_num_large_clusters = num_large_clusters

                # stop traversing the dendrogram as soon as we found a good candidate
                if num_large_clusters == num_clusters:
                    break

            # re-apply best iteration in case we did not find a perfect candidate
            if best_num_large_clusters != num_clusters:
                clusters = (
                    fcluster(_dendrogram, best_iteration, criterion="distance") - 1
                )
                cluster_unique, cluster_counts = np.unique(clusters, return_counts=True)
                large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
                num_large_clusters = len(large_clusters)
                print(
                    f"Found only {num_large_clusters} clusters. Using a smaller value than {min_cluster_size} for `min_cluster_size` might help."
                )

        if num_large_clusters == 0:
            clusters[:] = 0
            return clusters

        small_clusters = cluster_unique[cluster_counts < min_cluster_size]
        if len(small_clusters) == 0:
            return clusters

        # re-assign each small cluster to the most similar large cluster based on their respective centroids
        large_centroids = np.vstack(
            [
                np.mean(embeddings[clusters == large_k], axis=0)
                for large_k in large_clusters
            ]
        )
        small_centroids = np.vstack(
            [
                np.mean(embeddings[clusters == small_k], axis=0)
                for small_k in small_clusters
            ]
        )
        centroids_cdist = cdist(large_centroids, small_centroids, metric=self.metric)
        for small_k, large_k in enumerate(np.argmin(centroids_cdist, axis=0)):
            clusters[clusters == small_clusters[small_k]] = large_clusters[large_k]

        # re-number clusters from 0 to num_large_clusters
        _, clusters = np.unique(clusters, return_inverse=True)
        return clusters


class OracleClustering(BaseClustering):
    """Oracle clustering"""

    def __call__(
        self,
        embeddings: Optional[np.ndarray] = None,
        segmentations: Optional[SlidingWindowFeature] = None,
        file: Optional[AudioFile] = None,
        frames: Optional[SlidingWindow] = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply oracle clustering

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array, optional
            Sequence of embeddings. When provided, compute speaker centroids
            based on these embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.
        file : AudioFile
        frames : SlidingWindow

        Returns
        -------
        hard_clusters : (num_chunks, num_speakers) array
            Hard cluster assignment (hard_clusters[c, s] = k means that sth speaker
            of cth chunk is assigned to kth cluster)
        soft_clusters : (num_chunks, num_speakers, num_clusters) array
            Soft cluster assignment (the higher soft_clusters[c, s, k], the most likely
            the sth speaker of cth chunk belongs to kth cluster)
        centroids : (num_clusters, dimension), optional
            Clusters centroids if `embeddings` is provided, None otherwise.
        """

        num_chunks, num_frames, num_speakers = segmentations.data.shape
        window = segmentations.sliding_window

        oracle_segmentations = oracle_segmentation(file, window, frames=frames)
        #   shape: (num_chunks, num_frames, true_num_speakers)

        file["oracle_segmentations"] = oracle_segmentations

        _, oracle_num_frames, num_clusters = oracle_segmentations.data.shape

        segmentations = segmentations.data[:, : min(num_frames, oracle_num_frames)]
        oracle_segmentations = oracle_segmentations.data[
            :, : min(num_frames, oracle_num_frames)
        ]

        hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)
        soft_clusters = np.zeros((num_chunks, num_speakers, num_clusters))
        for c, (segmentation, oracle) in enumerate(
            zip(segmentations, oracle_segmentations)
        ):
            _, (permutation, *_) = permutate(oracle[np.newaxis], segmentation)
            for j, i in enumerate(permutation):
                if i is None:
                    continue
                hard_clusters[c, i] = j
                soft_clusters[c, i, j] = 1.0

        if embeddings is None:
            return hard_clusters, soft_clusters, None

        (
            train_embeddings,
            train_chunk_idx,
            train_speaker_idx,
        ) = self.filter_embeddings(
            embeddings,
            segmentations=segmentations,
        )

        train_clusters = hard_clusters[train_chunk_idx, train_speaker_idx]
        centroids = np.vstack(
            [
                np.mean(train_embeddings[train_clusters == k], axis=0)
                for k in range(num_clusters)
            ]
        )

        return hard_clusters, soft_clusters, centroids


class UMAPClustering(BaseClustering):
    """UMAP + HDBSCAN + PAHC clustering
    
    This implementation is based on WeSpeaker's clustering method which uses:
    1. UMAP for dimensionality reduction
    2. HDBSCAN for density-based clustering
    3. PAHC (Progressive Agglomerative Hierarchical Clustering) for post-processing
    
    Parameters
    ----------
    metric : {"cosine", "euclidean", ...}, optional
        Distance metric to use. Defaults to "cosine".
    max_num_embeddings : int, optional
        Maximum number of embeddings to use for clustering.
        Defaults to 1000.
    constrained_assignment : bool, optional
        Whether to use constrained assignment for clustering.
        Defaults to False.
        
    Hyper-parameters
    ----------------
    n_components : int
        Number of dimensions to reduce to with UMAP.
    n_neighbors : int
        Number of neighbors to consider in UMAP.
    min_dist : float
        Minimum distance between points in UMAP.
    hdbscan_min_cluster_size : int
        Minimum cluster size for HDBSCAN.
    pahc_merge_cutoff : float
        Similarity threshold for merging clusters in PAHC.
    pahc_min_cluster_size : int
        Minimum size of clusters after PAHC processing.
    pahc_absorb_cutoff : float
        Similarity threshold for absorbing small clusters in PAHC.
    """

    def __init__(
        self,
        metric: str = "cosine",
        max_num_embeddings: int = 1000,
        constrained_assignment: bool = False,
    ):
        super().__init__(
            metric=metric,
            max_num_embeddings=max_num_embeddings,
            constrained_assignment=constrained_assignment,
        )
        
        # UMAP parameters
        self.n_components = Integer(2, 64)
        self.n_neighbors = Integer(5, 50)
        self.min_dist = Uniform(0.0, 0.5)
        
        # HDBSCAN parameters
        self.hdbscan_min_cluster_size = Integer(2, 10)
        
        # PAHC parameters
        self.pahc_merge_cutoff = Uniform(0.0, 1.0)
        self.pahc_min_cluster_size = Integer(1, 10)
        self.pahc_absorb_cutoff = Uniform(-0.5, 0.5)

    def _pahc_fit_predict(self, labels, embeddings):
        """Progressive Agglomerative Hierarchical Clustering
        
        This method implements the PAHC algorithm from WeSpeaker for
        post-processing clustering results.
        
        Parameters
        ----------
        labels : np.ndarray
            Initial cluster labels from HDBSCAN
        embeddings : np.ndarray
            Original embeddings
            
        Returns
        -------
        labels : np.ndarray
            Refined cluster labels
        """
        # Step 1: Initialize structures
        label_map = {}
        cost_map = {}
        heap = []
        active_clusters = set()
        
        # Build label map (which embeddings belong to which cluster)
        for i, label in enumerate(labels):
            if label not in label_map:
                label_map[label] = []
            label_map[label].append(i)
        
        # Handle noise points (label -1) by assigning them to new clusters
        num_labeled = len(label_map)
        if -1 in label_map:
            num_labeled -= 1
            for i, j in zip(
                range(num_labeled, num_labeled + len(label_map[-1])),
                label_map[-1]
            ):
                label_map[i] = [j]
            del label_map[-1]
        
        # Initialize active clusters
        N = len(label_map)
        active_clusters = set(range(N))
        next_index = N
        
        # Compute costs between all cluster pairs
        for i in range(N):
            for j in range(i + 1, N):
                i_indexes, j_indexes = label_map[i], label_map[j]
                
                # Skip pairs of predefined clusters (both < num_labeled)
                if i < num_labeled and j < num_labeled:
                    cost_map[(i, j)] = -np.inf
                    continue
                
                # Compute similarity between clusters
                i_embedding = sum([
                    embeddings[idx] / np.linalg.norm(embeddings[idx]) 
                    for idx in i_indexes
                ])
                j_embedding = sum([
                    embeddings[idx] / np.linalg.norm(embeddings[idx])
                    for idx in j_indexes
                ])
                cost_map[(i, j)] = np.dot(i_embedding, j_embedding)
                
                # Add to heap if similarity is high enough
                factor = len(i_indexes) * len(j_indexes)
                normalized_cost = cost_map[(i, j)] / factor
                if normalized_cost >= self.pahc_merge_cutoff:
                    heap.append((-normalized_cost, (i, j)))
        
        # Convert to heap structure
        import heapq
        heapq.heapify(heap)
        
        # Step 2: Merge clusters progressively
        def eliminate(cluster_id):
            del label_map[cluster_id]
            active_clusters.remove(cluster_id)
            
        while heap:
            _, (i, j) = heapq.heappop(heap)
            if i in active_clusters and j in active_clusters:
                # Merge clusters i and j
                i_indexes, j_indexes = label_map[i], label_map[j]
                
                for k in label_map.keys():
                    if k == i or k == j:
                        continue
                    # Compute cost between new merged cluster and existing clusters
                    pair1 = (k, i) if k < i else (i, k)
                    pair2 = (k, j) if k < j else (j, k)
                    cost = cost_map.get(pair1, -np.inf) + cost_map.get(pair2, -np.inf)
                    cost_map[(k, next_index)] = cost
                    
                    factor = (len(i_indexes) + len(j_indexes)) * len(label_map[k])
                    normalized_cost = cost / factor
                    if normalized_cost >= self.pahc_merge_cutoff:
                        heapq.heappush(heap, (-normalized_cost, (k, next_index)))
                
                # Create new merged cluster
                label_map[next_index] = i_indexes + j_indexes
                active_clusters.add(next_index)
                eliminate(i)
                eliminate(j)
                next_index += 1
        
        # Step 3: Absorb small clusters into larger ones
        minor_clusters = set()
        major_clusters = set()
        
        for k, indexes in label_map.items():
            if len(indexes) < self.pahc_min_cluster_size:
                minor_clusters.add(k)
            else:
                major_clusters.add(k)
        
        if len(major_clusters) > 0:
            for i in minor_clusters:
                max_cost = -np.inf
                closest_cluster = -1
                
                for j in major_clusters:
                    pair = (i, j) if i < j else (j, i)
                    i_indexes, j_indexes = label_map[i], label_map[j]
                    
                    if pair in cost_map:
                        factor = len(i_indexes) * len(j_indexes)
                        normalized_cost = cost_map[pair] / factor
                        
                        if normalized_cost > max_cost:
                            max_cost = normalized_cost
                            closest_cluster = j
                
                if max_cost >= self.pahc_absorb_cutoff and closest_cluster != -1:
                    label_map[closest_cluster].extend(label_map[i])
                    eliminate(i)
        
        # Step 4: Relabel clusters
        new_labels = [-1] * len(labels)
        for label, indexes in label_map.items():
            for index in indexes:
                new_labels[index] = label
                
        # Remap labels to be consecutive integers
        i = 0
        label_to_label = {}
        for label in new_labels:
            if label not in label_to_label:
                label_to_label[label] = i
                i += 1
                
        for i in range(len(new_labels)):
            new_labels[i] = label_to_label[new_labels[i]]
            
        return np.array(new_labels)

    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        num_clusters: Optional[int] = None,
    ) -> np.ndarray:
        """Apply UMAP+HDBSCAN+PAHC clustering
        
        Parameters
        ----------
        embeddings : (num_embeddings, dimension) array
            Embeddings to cluster
        min_clusters : int
            Minimum number of clusters
        max_clusters : int
            Maximum number of clusters
        num_clusters : int, optional
            Target number of clusters (if known)
            
        Returns
        -------
        clusters : (num_embeddings, ) array
            0-indexed cluster indices
        """
        num_embeddings, dimension = embeddings.shape
        
        # For very small number of embeddings, use simple assignment
        if num_embeddings <= 2:
            return np.zeros(num_embeddings, dtype=np.int8)
        
        # Use min(32, num_embeddings-2) as the upper limit for n_components
        n_components = min(self.n_components, num_embeddings - 2)
        
        try:
            # UMAP dimensionality reduction
            umap_embeddings = umap.UMAP(
                n_components=n_components,
                metric=self.metric,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                random_state=1234,
                n_jobs=1
            ).fit_transform(embeddings)
            
            # HDBSCAN clustering
            labels = hdbscan.HDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                allow_single_cluster=True,
                approx_min_span_tree=False,
                core_dist_n_jobs=1
            ).fit_predict(umap_embeddings)
            
            # PAHC post-processing
            labels = self._pahc_fit_predict(labels, embeddings)
            
            # Count clusters
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels >= 0]
            num_detected_clusters = len(unique_labels)
            
            # If we detect fewer clusters than min_clusters, 
            # try to enforce the minimum number of clusters
            if num_detected_clusters < min_clusters and num_clusters is None:
                # Try to get at least min_clusters by using a different approach
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=min_clusters, random_state=1234, n_init=10)
                labels = kmeans.fit_predict(embeddings)
            
            # If we detect more clusters than max_clusters,
            # try to reduce to max_clusters
            elif num_detected_clusters > max_clusters:
                # Merge smallest clusters 
                from sklearn.cluster import AgglomerativeClustering
                agg = AgglomerativeClustering(
                    n_clusters=max_clusters, 
                    affinity='cosine', 
                    linkage='average'
                )
                labels = agg.fit_predict(embeddings)
                
            # If specific num_clusters is requested, enforce it
            elif num_clusters is not None and num_detected_clusters != num_clusters:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=num_clusters, random_state=1234, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                
            # Count clusters again after potential adjustments
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels >= 0]
            num_detected_clusters = len(unique_labels)
            
            # Fallback if everything else fails
            if num_detected_clusters < 1:
                return np.zeros(num_embeddings, dtype=np.int8)
                
            return labels
            
        except Exception as e:
            # Fallback to single cluster if any errors occur
            print(f"Error in UMAP+HDBSCAN+PAHC clustering: {e}")
            return np.zeros(num_embeddings, dtype=np.int8)


class Clustering(Enum):
    AgglomerativeClustering = AgglomerativeClustering
    OracleClustering = OracleClustering
    UMAPClustering = UMAPClustering
