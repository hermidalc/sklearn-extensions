# RNA-seq transformer functions

deseq2_norm_fit <- function(
    X, y=NULL, sample_meta=NULL, type="ratio", fit_type="parametric",
    is_classif=TRUE, model_batch=FALSE
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
        stopifnot(!is.null(y))
        dds <- DESeqDataSetFromMatrix(
            counts, data.frame(Class=factor(y)), ~Class
        )
    } else {
        dds <- DESeqDataSetFromMatrix(
            counts, data.frame(row.names=seq(1, ncol(counts))), ~1
        )
    }
    if (type == "poscounts") {
        locfunc <- genefilter::shorth
    } else {
        locfunc <- stats::median
    }
    dds <- estimateSizeFactors(dds, type=type, locfunc=locfunc, quiet=TRUE)
    suppressMessages(
        dds <- estimateDispersions(dds, fitType=fit_type, quiet=TRUE)
    )
    if (type == "ratio") {
        geo_means <- exp(rowMeans(log(counts)))
    } else if (type == "poscounts") {
        # adapted from DESeq2::estimateSizeFactors source code
        geoMeanNZ <- function(x) {
            if (all(x == 0)) { 0 }
            else { exp( sum(log(x[x > 0])) / length(x) ) }
        }
        geo_means <- apply(counts, 1, geoMeanNZ)
    }
    return(list(geo_means=geo_means, disp_func=dispersionFunction(dds)))
}

deseq2_norm_vst_transform <- function(X, geo_means, disp_func) {
    suppressPackageStartupMessages(library("DESeq2"))
    if (is.data.frame(X)) {
        rnames <- row.names(X)
        cnames <- colnames(X)
    }
    counts <- t(X)
    geo_means <- as.numeric(geo_means)
    dds <- DESeqDataSetFromMatrix(
        counts, data.frame(row.names=seq(1, ncol(counts))), ~1
    )
    dds <- estimateSizeFactors(dds, geoMeans=geo_means, quiet=TRUE)
    suppressMessages(dispersionFunction(dds) <- disp_func)
    vsd <- varianceStabilizingTransformation(dds, blind=FALSE)
    Xt <- t(assay(vsd))
    if (is.data.frame(X)) {
        Xt <- as.data.frame(Xt)
        row.names(Xt) <- rnames
        colnames(Xt) <- cnames
    }
    return(Xt)
}

# adapted from edgeR::calcNormFactors source code
edger_tmm_ref_column <- function(counts) {
    calcFactorQuantile <- function(data, lib.size, p=0.75) {
        f <- rep_len(1, ncol(data))
        for (j in seq_len(ncol(data))) f[j] <- quantile(data[, j], probs=p)
        if (min(f) == 0) warning("One or more quantiles are zero")
        f / lib.size
    }
    f75 <- suppressWarnings(
        calcFactorQuantile(data=counts, lib.size=colSums(counts), p=0.75)
    )
    if (median(f75) < 1e-20) {
        ref_column <- which.max(colSums(sqrt(counts)))
    } else {
        ref_column <- which.min(abs(f75 - mean(f75)))
    }
}

edger_tmm_fit <- function(X) {
    suppressPackageStartupMessages(library("edgeR"))
    counts <- t(X)
    ref_sample <- counts[, edger_tmm_ref_column(counts)]
    return(ref_sample)
}

edger_tmm_cpm_transform <- function(X, ref_sample, log=TRUE, prior_count=2) {
    suppressPackageStartupMessages(library("edgeR"))
    if (is.data.frame(X)) {
        rnames <- row.names(X)
        cnames <- colnames(X)
    }
    counts <- t(X)
    ref_sample <- as.numeric(ref_sample)
    ref_sample_mask <- apply(counts, 2, function(c) all(c == ref_sample))
    if (any(ref_sample_mask)) {
        dge <- DGEList(counts=counts)
        suppressWarnings(dge <- calcNormFactors(
            dge, method="TMM", refColumn=min(which(ref_sample_mask))
        ))
        cpms <- cpm(dge, log=log, prior.count=prior_count)
    } else {
        counts <- cbind(counts, ref_sample)
        colnames(counts) <- NULL
        dge <- DGEList(counts=counts)
        suppressWarnings(dge <- calcNormFactors(
            dge, method="TMM", refColumn=ncol(dge)
        ))
        cpms <- cpm(dge, log=log, prior.count=prior_count)
        cpms <- cpms[, -ncol(cpms)]
    }
    Xt <- t(cpms)
    if (is.data.frame(X)) {
        Xt <- as.data.frame(Xt)
        row.names(Xt) <- rnames
        colnames(Xt) <- cnames
    }
    return(Xt)
}

edger_tmm_tpm_transform <- function(
    X, feature_meta, ref_sample, log=TRUE, prior_count=2,
    gene_length_col="Length"
) {
    if (is.null(feature_meta)) stop("feature_meta cannot be NULL")
    suppressPackageStartupMessages(library("edgeR"))
    if (is.data.frame(X)) {
        rnames <- row.names(X)
        cnames <- colnames(X)
    }
    counts <- t(X)
    ref_sample_mask <- apply(counts, 2, function(c) all(c == ref_sample))
    if (any(ref_sample_mask)) {
        dge <- DGEList(counts=counts, genes=feature_meta)
        suppressWarnings(dge <- calcNormFactors(
            dge, method="TMM", refColumn=min(which(ref_sample_mask))
        ))
    } else {
        counts <- cbind(counts, ref_sample)
        colnames(counts) <- NULL
        dge <- DGEList(counts=counts, genes=feature_meta)
        suppressWarnings(dge <- calcNormFactors(
            dge, method="TMM", refColumn=ncol(dge)
        ))
    }
    if (log) {
        # XXX: edgeR doesn't have built-in support for logTPM w/ prior.count
        #      so do API internal logic manually
        # TODO: use effectiveLibSizes() in newer edgeR versions
        lib_size <- dge$samples$lib.size * dge$samples$norm.factors
        scaled_prior_count <- prior_count * lib_size / mean(lib_size)
        adj_lib_size <- lib_size + 2 * scaled_prior_count
        fpkms <- t(
            (t(dge$counts) + scaled_prior_count) / adj_lib_size
        ) * 1e6 / dge$genes[[gene_length_col]] * 1e3
        stopifnot(all.equal(
            log2(fpkms), rpkm(
                dge, gene.length=gene_length_col, log=log,
                prior.count=prior_count
            )
        ))
        tpms <- log2(t(t(fpkms) / colSums(fpkms)) * 1e6)
    } else {
        fpkms <- rpkm(
            dge, gene.length=gene_length_col, log=log, prior.count=prior_count
        )
        tpms <- t(t(fpkms) / colSums(fpkms)) * 1e6
    }
    if (!any(ref_sample_mask)) tpms <- tpms[, -ncol(tpms)]
    Xt <- t(tpms)
    if (is.data.frame(X)) {
        Xt <- as.data.frame(Xt)
        row.names(Xt) <- rnames
        colnames(Xt) <- cnames
    }
    return(Xt)
}

edger_cpm_transform <- function(X, log=TRUE, prior_count=2) {
    if (is.data.frame(X)) {
        rnames <- row.names(X)
        cnames <- colnames(X)
    }
    counts <- t(X)
    cpms <- edgeR::cpm(counts, log=log, prior.count=prior_count)
    Xt <- t(cpms)
    if (is.data.frame(X)) {
        Xt <- as.data.frame(Xt)
        row.names(Xt) <- rnames
        colnames(Xt) <- cnames
    }
    return(Xt)
}
