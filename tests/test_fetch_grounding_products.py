from pathlib import Path

import h5py
import numpy as np

from pipelines.fetch_grounding_products import extract_is2gzant_h5


def test_extract_is2gzant_filters_to_map_roi(tmp_path: Path):
    path = tmp_path / "IS2GZANT_v01.h5"
    with h5py.File(path, "w") as h5:
        for name in ("Point_F", "Point_H", "Point_Ib"):
            group = h5.create_group(name)
            group["latitude"] = np.array([-75.0, -72.0])
            group["longitude"] = np.array([-105.0, -105.0])
            group["beam"] = np.array([b"l", b"r"])
            group["beam_pair"] = np.array([1, 2])
            group["nominal_error"] = np.array([80.0, 80.0])
            group["repeat_cycles_no"] = np.array([7, 7])
            group["track"] = np.array([10, 11])
            if name != "Point_Ib":
                group["tide_range"] = np.array([1.2, 1.3])

    frame = extract_is2gzant_h5(path, (-115.0, -77.5, -95.0, -73.0))

    assert len(frame) == 3
    assert set(frame["feature_type"]) == {"Point_F", "Point_H", "Point_Ib"}
    assert set(frame["beam"]) == {"l"}
    assert frame.loc[frame.feature_type == "Point_Ib", "tide_range"].isna().all()
