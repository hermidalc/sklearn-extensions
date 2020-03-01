# RNA-seq transformer functions

deseq2_vst_fit <- function(
    X, y, sample_meta=NULL, blind=FALSE, fit_type="parametric",
    model_batch=FALSE, is_classif=TRUE
) {
    suppressPackageStartupMessages(library("DESeq2"))
    counts <- t(X)
    geo_means <- exp(rowMeans(log(counts)))
    if (
        !is.null(sample_meta) && length(unique(sample_meta$Batch)) > 1 &&
        model_batch
    ) {
        sample_meta$Batch <- factor(sample_meta$Batch)
        if (is_classif) {
            sample_meta$Class <- factor(sample_meta$Class)
            dds <- DESeqDataSetFromMatrix(
                counts, as.data.frame(sample_meta), ~Batch + Class
            )
        } else {
            dds <- DESeqDataSetFromMatrix(
                counts, as.data.frame(sample_meta), ~Batch
            )
        }
    } else if (is_classif) {
        dds <- DESeqDataSetFromMatrix(
            counts, data.frame(Class=factor(y)), ~Class
        )
    } else {
        dds <- DESeqDataSetFromMatrix(
            counts, data.frame(row.names=seq(1, ncol(counts))), ~1
        )
    }
    dds <- estimateSizeFactors(dds, quiet=TRUE)
    suppressMessages(
        dds <- estimateDispersions(dds, fitType=fit_type, quiet=TRUE)
    )
    vsd <- varianceStabilizingTransformation(dds, blind=blind)
    return(list(
        t(as.matrix(assay(vsd))), geo_means, dispersionFunction(dds)
    ))
}

deseq2_vst_transform <- function(X, geo_means, disp_func) {
    suppressPackageStartupMessages(library("DESeq2"))
    counts <- t(X)
    dds <- DESeqDataSetFromMatrix(
        counts, data.frame(row.names=seq(1, ncol(counts))), ~1
    )
    dds <- estimateSizeFactors(dds, geoMeans=geo_means, quiet=TRUE)
    suppressMessages(dispersionFunction(dds) <- disp_func)
    vsd <- varianceStabilizingTransformation(dds, blind=FALSE)
    return(t(as.matrix(assay(vsd))))
}

edger_logcpm_transform <- function(X, prior_count=1) {
    return(t(edgeR::cpm(t(X), log=TRUE, prior.count=prior_count)))
}

# adapted from edgeR::calcNormFactors source code
edger_tmm_ref_column <- function(counts, lib.size=colSums(counts), p=0.75) {
    y <- t(t(counts) / lib.size)
    f <- apply(y, 2, function(x) quantile(x, p=p))
    ref_column <- which.min(abs(f - mean(f)))
}

edger_tmm_logcpm_fit <- function(X, prior_count=1) {
    suppressPackageStartupMessages(library("edgeR"))
    counts <- t(X)
    dge <- DGEList(counts=counts)
    dge <- calcNormFactors(dge, method="TMM")
    log_cpm <- cpm(dge, log=TRUE, prior.count=prior_count)
    ref_sample <- counts[, edger_tmm_ref_column(counts)]
    return(list(t(log_cpm), ref_sample))
}

edger_tmm_logcpm_transform <- function(X, ref_sample, prior_count=1) {
    suppressPackageStartupMessages(library("edgeR"))
    counts <- t(X)
    counts <- cbind(counts, ref_sample)
    colnames(counts) <- NULL
    dge <- DGEList(counts=counts)
    dge <- calcNormFactors(dge, method="TMM", refColumn=ncol(dge))
    log_cpm <- cpm(dge, log=TRUE, prior.count=prior_count)
    log_cpm <- log_cpm[, -ncol(log_cpm)]
    return(t(log_cpm))
}
