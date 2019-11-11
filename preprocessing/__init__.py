"""
The :mod:`sklearn.preprocessing` module includes scaling, centering,
normalization, binarization methods.
"""

from .rna_seq import DESeq2RLEVST, EdgeRTMMLogCPM, LimmaRemoveBatchEffect


__all__ = ['DESeq2RLEVST',
           'EdgeRTMMLogCPM',
           'LimmaRemoveBatchEffect']
