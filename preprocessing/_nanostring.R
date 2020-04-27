# NanoString transformer functions

nanostringdiff_fit <- function(X, y, feature_meta, meta_col="Code.Class") {
    suppressPackageStartupMessages(library("NanoStringDiff"))
    counts <- t(X)
    endogenous <- counts[feature_meta[[meta_col]] == "Endogenous", , drop=FALSE]
    positive <- counts[feature_meta[[meta_col]] == "Positive", , drop=FALSE]
    negative <- counts[feature_meta[[meta_col]] == "Negative", , drop=FALSE]
    housekeeping <- counts[feature_meta[[meta_col]] %in% c(
        "Control", "Housekeeping", "housekeeping"
    ), , drop=FALSE]
    nsd <- createNanoStringSet(
        endogenous=endogenous, positiveControl=positive,
        negativeControl=negative, housekeepingControl=housekeeping,
        designs=data.frame(Class=factor(y))
    )
    nsd <- estNormalizationFactors(nsd)
    return(list(
        positiveFactor(nsd), negativeFactor(nsd), housekeepingFactor(nsd)
    ))
}

nanostringdiff_transform <- function(
    X, postive_factor, negative_factor, housekeeping_factor,
    background_threshold=TRUE
) {
    counts <- t(X)
    if (background_threshold) {
        norm_counts <-
            t(ifelse(t(counts) < negative_factor, negative_factor, t(counts)))
    } else {
        norm_counts <- counts - negative_factor
    }
    norm_counts <- round(norm_counts / (postive_factor * housekeeping_factor))
    norm_counts[norm_counts < 0] <- 0
    return(t(norm_counts))
}
