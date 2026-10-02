"""
Centralized layer name constants for the labeling tool.
All hardcoded layer name strings should reference these constants.
"""


class LAYER_NAMES:
    ACCEPTED = "accepted_labels"
    SEMANTIC = "semantic_polygons"
    SEMANTIC_RAW = "semantic_polygons_raw"
    SAM_REFINED = "sam_refined_polygons"
    CANDIDATES = "candidates"
    CLASS_POLYGONS = "class_polygons"
    FINAL_COMPOSITE = "final_composite"
    TOPOLOGY_ISSUES = "topology_issues"
