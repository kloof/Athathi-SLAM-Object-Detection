"""
World-frame spatial index for object re-identification.

Maps (world position, class) -> canonical object ID so that BoT-SORT track
fragmentation (occlusion, leaving FOV) doesn't create duplicate 3D objects.
"""

import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree

# Classes that YOLO commonly confuses with each other.
# If class A is in the compatibility set for class B, a spatial match is allowed.
_COMPATIBLE_CLASSES = {
    "table": {"desk"},
    "desk": {"table"},
    "sofa": {"bed"},
    "bed": {"sofa"},
    "cabinet": {"wardrobe"},
    "wardrobe": {"cabinet"},
}

# KD-tree is rebuilt after this many new entries accumulate.
_KDTREE_REBUILD_INTERVAL = 10


class SpatialObjectMemory:
    """World-frame spatial index for object re-identification."""

    def __init__(self, match_radius=0.8, class_must_match=True):
        self.match_radius = match_radius
        self.class_must_match = class_must_match

        # canonical_id -> {center, class_votes, obs_count}
        self.entries = {}
        self._next_id = 1
        self._track_to_canonical = {}  # botsort_track_id -> canonical_id

        # KD-tree for spatial lookup
        self._tree = None
        self._tree_ids = []        # canonical IDs in tree order
        self._new_since_rebuild = 0

    def lookup_or_create(self, track_id, world_center, class_name, confidence):
        """
        Given a BoT-SORT track_id and its world position, return a canonical ID.

        1. If track_id already mapped -> return canonical, update position
        2. Else search spatially: nearest entry within match_radius with compatible class
           -> map track_id to that canonical, fuse
        3. Else create new canonical entry
        """
        world_center = np.asarray(world_center, dtype=np.float64)

        # Fast path: track_id already known
        if track_id in self._track_to_canonical:
            cid = self._track_to_canonical[track_id]
            self._fuse(cid, world_center, class_name, confidence)
            return cid

        # Spatial search for existing match
        cid = self._find_spatial_match(world_center, class_name)
        if cid is not None:
            self._track_to_canonical[track_id] = cid
            self._fuse(cid, world_center, class_name, confidence)
            return cid

        # New object
        cid = self._next_id
        self._next_id += 1
        self.entries[cid] = {
            "center": world_center.copy(),
            "class_votes": defaultdict(float, {class_name: confidence}),
            "obs_count": 1,
        }
        self._track_to_canonical[track_id] = cid
        self._new_since_rebuild += 1
        self._maybe_rebuild_tree()
        return cid

    def get_class(self, canonical_id):
        """Return the best class name (confidence-weighted majority vote)."""
        entry = self.entries.get(canonical_id)
        if entry is None:
            return "unknown"
        return max(entry["class_votes"], key=entry["class_votes"].get)

    def _fuse(self, cid, center, class_name, confidence):
        """Update an existing entry with a new observation."""
        entry = self.entries[cid]
        n = entry["obs_count"]
        # Running average of position
        entry["center"] = (entry["center"] * n + center) / (n + 1)
        entry["class_votes"][class_name] += confidence
        entry["obs_count"] = n + 1

    def _find_spatial_match(self, center, class_name):
        """Find nearest compatible entry within match_radius."""
        if not self.entries:
            return None

        # Use KD-tree if available and up-to-date
        if self._tree is not None and self._new_since_rebuild == 0:
            dists, idxs = self._tree.query(center.reshape(1, -1), k=min(5, len(self._tree_ids)))
            dists = np.atleast_1d(dists.squeeze())
            idxs = np.atleast_1d(idxs.squeeze())
            for dist, idx in zip(dists, idxs):
                if dist > self.match_radius:
                    break
                cid = self._tree_ids[int(idx)]
                if self._class_compatible(cid, class_name):
                    return cid
            return None

        # Brute force fallback (few entries or tree stale)
        best_cid = None
        best_dist = self.match_radius
        for cid, entry in self.entries.items():
            dist = np.linalg.norm(entry["center"] - center)
            if dist < best_dist and self._class_compatible(cid, class_name):
                best_dist = dist
                best_cid = cid
        return best_cid

    def _class_compatible(self, cid, class_name):
        """Check if class_name is compatible with the entry's best class."""
        if not self.class_must_match:
            return True
        entry_class = self.get_class(cid)
        if entry_class == class_name:
            return True
        compatible = _COMPATIBLE_CLASSES.get(class_name, set())
        return entry_class in compatible

    def _maybe_rebuild_tree(self):
        """Rebuild KD-tree if enough new entries have accumulated."""
        if self._new_since_rebuild < _KDTREE_REBUILD_INTERVAL:
            return
        self._rebuild_tree()

    def _rebuild_tree(self):
        """Build a KD-tree from all current entry centers."""
        self._tree_ids = list(self.entries.keys())
        centers = np.array([self.entries[cid]["center"] for cid in self._tree_ids])
        self._tree = cKDTree(centers)
        self._new_since_rebuild = 0
