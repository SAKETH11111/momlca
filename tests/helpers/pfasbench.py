"""Shared PFASBench test data helpers."""

from __future__ import annotations

from pathlib import Path


def write_sample_pfasbench_dataset(root: Path) -> Path:
    """Create a tiny PFASBench-style dataset under ``root``."""
    raw_dir = root / "pfasbench" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "pfasbench.csv").write_text(
        "\n".join(
            [
                "smiles,name,logS,logP,pKa",
                "C(=O)(C(F)(F)F)O,TFA,-0.5,0.5,0.5",
                "C(=O)(C(C(F)(F)F)(F)F)O,PFPA,-1.0,1.0,0.6",
                "C(=O)(C(C(C(F)(F)F)(F)F)(F)F)O,PFBA,-1.5,1.5,0.7",
                "C(=O)(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)O,PFPeA,-2.0,2.0,0.8",
                "c1ccccc1,Benzene,-0.8,2.1,15.0",
                "c1ccc(cc1)O,Phenol,-0.5,1.5,9.9",
            ]
        )
        + "\n"
    )
    return root
