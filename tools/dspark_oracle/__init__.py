"""Development-only numerical oracle for DeepSeek V4 Flash DSpark.

This package deliberately depends on NumPy, not on the Hebrus runtime.  It is
used to generate and check small fixtures before the same equations are
implemented in the production Metal path.
"""

from .metadata import DSparkMetadata, MetadataError, validate_0731_metadata
from .reference import (
    ConfidenceSchedule,
    SpeculativeSample,
    conditional_confidence,
    markov_greedy_draft,
    markov_step_bias,
    post_layer_hc_mean,
    speculative_sample_exact,
)

__all__ = [
    "ConfidenceSchedule",
    "DSparkMetadata",
    "MetadataError",
    "SpeculativeSample",
    "conditional_confidence",
    "markov_greedy_draft",
    "markov_step_bias",
    "post_layer_hc_mean",
    "speculative_sample_exact",
    "validate_0731_metadata",
]
