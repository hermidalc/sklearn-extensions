# Batch effect correction transformer functions

# adapted from limma::removeBatchEffect source code
limma_remove_ba_fit <- function(X, sample_meta, preserve_design=TRUE) {
    suppressPackageStartupMessages(library("limma"))
    batch <- sample_meta$Batch
    sample_meta$Batch <- NULL
    if (preserve_design) {
        sample_meta$Class <- as.factor(sample_meta$Class)
        design <- model.matrix(~Class, data=sample_meta)
    } else {
        design <- matrix(1, ncol(t(X)), 1)
    }
    batch <- as.factor(batch)
    contrasts(batch) <- contr.sum(levels(batch))
    batch <- model.matrix(~batch)[, -1, drop=FALSE]
    fit <- lmFit(t(X), cbind(design, batch))
    beta <- fit$coefficients[, -seq_len(ncol(design)), drop=FALSE]
    beta[is.na(beta)] <- 0
    return(beta)
}

limma_remove_ba_transform <- function(X, sample_meta, beta) {
    batch <- sample_meta$Batch
    batch <- as.factor(batch)
    contrasts(batch) <- contr.sum(levels(batch))
    batch <- model.matrix(~batch)[, -1, drop=FALSE]
    return(t(t(X) - beta %*% t(batch)))
}
