# RNA-seq transformer functions

deseq2_vst_fit <- function(
    X, y, sample_meta=NULL, fit_type="parametric", model_batch=FALSE,
    is_classif=TRUE
) {
    suppressPackageStartupMessages(library("DESeq2"))
    counts <- t(X)
    if (
        model_batch && !is.null(sample_meta) &&
        length(unique(sample_meta$Batch)) > 1
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
    geo_means <- exp(rowMeans(log(counts)))
    return(list(geo_means, dispersionFunction(dds)))
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
    return(t(assay(vsd)))
}

# adapted from edgeR::calcNormFactors source code
edger_tmm_ref_column <- function(counts, lib.size=colSums(counts), p=0.75) {
    y <- t(t(counts) / lib.size)
    f <- apply(y, 2, function(x) quantile(x, p=p))
    ref_column <- which.min(abs(f - mean(f)))
}

edger_tmm_fit <- function(X) {
    suppressPackageStartupMessages(library("edgeR"))
    counts <- t(X)
    ref_sample <- counts[, edger_tmm_ref_column(counts)]
    return(ref_sample)
}

edger_tmm_logcpm_transform <- function(X, ref_sample, prior_count=2) {
    suppressPackageStartupMessages(library("edgeR"))
    counts <- t(X)
    ref_sample_mask <- apply(counts, 2, function(c) all(c == ref_sample))
    if (any(ref_sample_mask)) {
        dge <- DGEList(counts=counts)
        dge <- calcNormFactors(
            dge, method="TMM", refColumn=which.min(ref_sample_mask)
        )
        log_cpm <- cpm(dge, log=TRUE, prior.count=prior_count)
    } else {
        counts <- cbind(counts, ref_sample)
        colnames(counts) <- NULL
        dge <- DGEList(counts=counts)
        dge <- calcNormFactors(dge, method="TMM", refColumn=ncol(dge))
        log_cpm <- cpm(dge, log=TRUE, prior.count=prior_count)
        log_cpm <- log_cpm[, -ncol(log_cpm)]
    }
    return(t(log_cpm))
}

edger_tmm_tpm_transform <- function(
    X, feature_meta, ref_sample, meta_col="Length"
) {
    suppressPackageStartupMessages(library("edgeR"))
    counts <- t(X)
    ref_sample_mask <- apply(counts, 2, function(c) all(c == ref_sample))
    if (any(ref_sample_mask)) {
        dge <- DGEList(counts=counts, genes=feature_meta)
        dge <- calcNormFactors(
            dge, method="TMM", refColumn=which.min(ref_sample_mask)
        )
        rpkm <- rpkm(dge, gene.length=meta_col, log=FALSE)
        tpm <- t(t(rpkm) / colSums(rpkm)) * 1e6
    } else {
        counts <- cbind(counts, ref_sample)
        colnames(counts) <- NULL
        dge <- DGEList(counts=counts, genes=feature_meta)
        dge <- calcNormFactors(dge, method="TMM", refColumn=ncol(dge))
        rpkm <- rpkm(dge, gene.length=meta_col, log=FALSE)
        tpm <- t(t(rpkm) / colSums(rpkm)) * 1e6
        tpm <- tpm[, -ncol(tpm)]
    }
    return(t(tpm))
}

edger_logcpm_transform <- function(X, prior_count=2) {
    return(t(edgeR::cpm(t(X), log=TRUE, prior.count=prior_count)))
}
