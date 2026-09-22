"""Stage configs + scale gates."""
STAGES = ["S1", "S2", "S3"]
GATES = {
    "S1_to_S2": 0.60,   # need 60% win on S1 before S2 counts
    "S2_to_S3": 0.70,   # need 70% win on S2 before S3 counts
}
