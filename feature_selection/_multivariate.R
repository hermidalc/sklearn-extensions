# Multivariate feature selection functions

source(paste(dirname(sys.frame(1)$ofile), "_fcbf.R", sep="/"))

cfs_feature_idxs <- function(X, y) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    feature_idxs <- FSelector::cfs(
        as.formula("Class ~ ."), cbind(X, "Class"=factor(y))
    )
    return(as.integer(feature_idxs) - 1)
}

fcbf_feature_idxs <- function(X, y, threshold=0) {
    results <- select.fast.filter(
        cbind(X, as.factor(y)), disc.method="MDL", threshold=threshold
    )
    results <- results[order(results$NumberFeature), , drop=FALSE]
    return(list(results$NumberFeature - 1, results$Information.Gain))
}

gain_ratio_feature_idxs <- function(X, y) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::gain.ratio(
        as.formula("Class ~ ."), cbind(X, "Class"=factor(y)), unit="log2"
    )
    results <- results[results$attr_importance > 0, , drop=FALSE]
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(list(as.integer(row.names(results)) - 1, results$attr_importance))
}

sym_uncert_feature_idxs <- function(X, y) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::symmetrical.uncertainty(
        as.formula("Class ~ ."), cbind(X, "Class"=factor(y)), unit="log2"
    )
    results <- results[results$attr_importance > 0, , drop=FALSE]
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(list(as.integer(row.names(results)) - 1, results$attr_importance))
}

relieff_feature_score <- function(X, y, num_neighbors=10, sample_size=5) {
    X <- as.data.frame(X)
    colnames(X) <- seq(1, ncol(X))
    results <- FSelector::relief(
        as.formula("Class ~ ."), cbind(X, "Class"=factor(y)),
        neighbours.count=num_neighbors, sample.size=sample_size
    )
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(results$attr_importance)
}
