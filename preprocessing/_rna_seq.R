# Count data normalization and transformation functions

source(paste(
    dirname(sys.frame(1)$ofile), "_wrench.R", sep="/"
))

deseq2_norm_fit <- function(
    X, y=NULL, sample_meta=NULL, norm_type="ratio", fit_type="parametric",
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
            colData <- as.data.frame(sample_meta)
            design <- ~Batch + Class
        } else {
            colData <- as.data.frame(sample_meta)
            design <- ~Batch
        }
    } else if (is_classif) {
        stopifnot(!is.null(y))
        colData <- data.frame(Class=factor(y))
        design <- ~Class
    } else {
        colData <- data.frame(row.names=seq(1, ncol(counts)))
        design <- ~1
    }
    if (norm_type == "ratio") {
        geo_means <- exp(rowMeans(log(counts)))
    } else if (norm_type == "poscounts") {
        # adapted from DESeq2::estimateSizeFactors source code
        geoMeanNZ <- function(x) {
            if (all(x == 0)) { 0 }
            else { exp( sum(log(x[x > 0])) / length(x) ) }
        }
        geo_means <- apply(counts, 1, geoMeanNZ)
    }
    dds <- DESeqDataSetFromMatrix(counts, colData, design)
    if (norm_type == "poscounts") {
        locfunc <- genefilter::shorth
    } else {
        locfunc <- stats::median
    }
    suppressMessages({
        dds <- estimateSizeFactors(
            dds, type=norm_type, locfunc=locfunc, quiet=TRUE
        )
        dds <- estimateDispersions(dds, fitType=fit_type, quiet=TRUE)
    })
    return(list(geo_means=geo_means, disp_func=dispersionFunction(dds)))
}

deseq2_norm_transform <- function(
    X, geo_means, disp_func, norm_type="ratio", trans_type="vst"
) {
    suppressPackageStartupMessages(library("DESeq2"))
    if (is.data.frame(X)) {
        rnames <- row.names(X)
        cnames <- colnames(X)
    }
    counts <- t(X)
    geo_means <- as.vector(geo_means)
    dds <- DESeqDataSetFromMatrix(
        counts, data.frame(row.names=seq(1, ncol(counts))), ~1
    )
    if (norm_type == "poscounts") {
        locfunc <- genefilter::shorth
    } else {
        locfunc <- stats::median
    }
    suppressMessages({
        dds <- estimateSizeFactors(
            dds, geoMeans=geo_means, type=norm_type, locfunc=locfunc, quiet=TRUE
        )
        dispersionFunction(dds) <- disp_func
    })
    if (trans_type == "vst") {
        vsd <- varianceStabilizingTransformation(dds, blind=FALSE)
        tmat <- assay(vsd)
    }
    Xt <- t(tmat)
    if (is.data.frame(X)) {
        Xt <- as.data.frame(Xt)
        row.names(Xt) <- rnames
        colnames(Xt) <- cnames
    }
    return(Xt)
}

deseq2_wrench_fit <- function(
    X, sample_meta, est_type="w.marg.mean", ref_type="sw.means", z_adj=FALSE,
    fit_type="parametric"
) {
    suppressPackageStartupMessages(library("DESeq2"))
    counts <- t(X)
    if (is.null(colnames(counts))) {
        colnames(counts) <- paste0("X", seq_len(ncol(counts)))
    }
    # condition <- as.vector(sample_meta$Class)
    condition <- rep(1, ncol(counts))
    suppressWarnings(W <- wrench(
        counts, condition, etype=est_type, ref.est=ref_type, z.adj=z_adj
    ))
    size_factors <- rep(1, ncol(counts))
    names(size_factors) <- colnames(counts)
    size_factors[names(W$nf)] <- W$nf
    size_factors[is.na(size_factors)] <- 1
    sample_meta$Class <- factor(sample_meta$Class)
    colData <- as.data.frame(sample_meta)
    design <- ~Class
    dds <- DESeqDataSetFromMatrix(counts, colData, design)
    sizeFactors(dds) <- size_factors
    suppressMessages(
        dds <- estimateDispersions(dds, fitType=fit_type, quiet=TRUE)
    )
    return(list(
        nzrows=W$others$nzrows, qref=W$others$qref, s2=W$others$s2,
        s2thetag=W$others$s2thetag, thetag=W$others$thetag,
        pi0_fit=W$others$pi0.fit, disp_func=dispersionFunction(dds)
    ))
}

deseq2_wrench_transform <- function(
    X, sample_meta, nzrows, qref, s2, s2thetag, thetag, pi0_fit,
    disp_func, est_type="w.marg.mean", ref_type="sw.means", z_adj=FALSE,
    trans_type="vst"
) {
    suppressPackageStartupMessages(library("DESeq2"))
    if (is.data.frame(X)) {
        rnames <- row.names(X)
        cnames <- colnames(X)
    }
    counts <- t(X)
    nzrows <- as.vector(nzrows)
    qref <- as.vector(qref)
    s2 <- as.vector(s2)
    s2thetag <- as.vector(s2thetag)
    thetag <- as.vector(thetag)
    if (is.null(colnames(counts))) {
        colnames(counts) <- paste0("X", seq_len(ncol(counts)))
    }
    # condition <- as.vector(sample_meta$Class)
    condition <- rep(1, ncol(counts))
    suppressWarnings(W <- wrench(
        counts, condition,
        nzrows=nzrows, qref=qref, s2=s2, s2thetag=s2thetag, thetag=thetag,
        pi0.fit=pi0_fit, etype=est_type, ref.est=ref_type, z.adj=z_adj
    ))
    size_factors <- rep(1, ncol(counts))
    names(size_factors) <- colnames(counts)
    size_factors[names(W$nf)] <- W$nf
    size_factors[is.na(size_factors)] <- 1
    dds <- DESeqDataSetFromMatrix(
        counts, data.frame(row.names=seq(1, ncol(counts))), ~1
    )
    suppressMessages({
        sizeFactors(dds) <- size_factors
        dispersionFunction(dds) <- disp_func
    })
    if (trans_type == "vst") {
        vsd <- varianceStabilizingTransformation(dds, blind=FALSE)
        tmat <- assay(vsd)
    }
    Xt <- t(tmat)
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

edger_norm_fit <- function(X, norm_type="TMM") {
    suppressPackageStartupMessages(library("edgeR"))
    counts <- t(X)
    if (norm_type == "TMM") {
        ref_sample <- counts[, edger_tmm_ref_column(counts)]
    }
    return(list(ref_sample=ref_sample))
}

edger_norm_transform <- function(
    X, ref_sample, feature_meta=NULL, norm_type="TMM", trans_type="cpm",
    log=TRUE, prior_count=2, gene_length_col="Length"
) {
    suppressPackageStartupMessages(library("edgeR"))
    if (is.data.frame(X)) {
        rnames <- row.names(X)
        cnames <- colnames(X)
    }
    counts <- t(X)
    ref_sample <- as.vector(ref_sample)
    ref_sample_mask <- apply(counts, 2, function(c) all(c == ref_sample))
    if (any(ref_sample_mask)) {
        dge <- DGEList(counts=counts, genes=feature_meta)
        suppressWarnings(dge <- calcNormFactors(
            dge, method=norm_type, refColumn=min(which(ref_sample_mask))
        ))
    } else {
        counts <- cbind(counts, ref_sample)
        colnames(counts) <- NULL
        dge <- DGEList(counts=counts, genes=feature_meta)
        suppressWarnings(dge <- calcNormFactors(
            dge, method=norm_type, refColumn=ncol(dge)
        ))
    }
    if (trans_type == "cpm") {
        tmat <- cpm(dge, log=log, prior.count=prior_count)
    } else if (trans_type == "tpm") {
        stopifnot(!is.null(feature_meta))
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
            tmat <- log2(t(t(fpkms) / colSums(fpkms)) * 1e6)
        } else {
            fpkms <- rpkm(
                dge, gene.length=gene_length_col, log=log,
                prior.count=prior_count
            )
            tmat <- t(t(fpkms) / colSums(fpkms)) * 1e6
        }
    }
    if (!any(ref_sample_mask)) tmat <- tmat[, -ncol(tmat)]
    Xt <- t(tmat)
    if (is.data.frame(X)) {
        Xt <- as.data.frame(Xt)
        row.names(Xt) <- rnames
        colnames(Xt) <- cnames
    }
    return(Xt)
}

edger_wrench_fit <- function(
    X, sample_meta, est_type="w.marg.mean", ref_type="sw.means", z_adj=FALSE
) {
    counts <- t(X)
    # condition <- as.vector(sample_meta$Class)
    condition <- rep(1, ncol(counts))
    suppressWarnings(W <- wrench(
        counts, condition, etype=est_type, ref.est=ref_type, z.adj=z_adj
    ))
    return(list(
        nzrows=W$others$nzrows, qref=W$others$qref, s2=W$others$s2,
        s2thetag=W$others$s2thetag, thetag=W$others$thetag,
        pi0_fit=W$others$pi0.fit
    ))
}

edger_wrench_transform <- function(
    X, sample_meta, nzrows, qref, s2, s2thetag, thetag, pi0_fit,
    est_type="w.marg.mean", ref_type="sw.means", z_adj=FALSE, feature_meta=NULL,
    trans_type="cpm", log=TRUE, prior_count=1, gene_length_col="Length"
) {
    suppressPackageStartupMessages(library("edgeR"))
    if (is.data.frame(X)) {
        rnames <- row.names(X)
        cnames <- colnames(X)
    }
    counts <- t(X)
    nzrows <- as.vector(nzrows)
    qref <- as.vector(qref)
    s2 <- as.vector(s2)
    s2thetag <- as.vector(s2thetag)
    thetag <- as.vector(thetag)
    if (is.null(colnames(counts))) {
        colnames(counts) <- paste0("X", seq_len(ncol(counts)))
    }
    # condition <- as.vector(sample_meta$Class)
    condition <- rep(1, ncol(counts))
    suppressWarnings(W <- wrench(
        counts, condition,
        nzrows=nzrows, qref=qref, s2=s2, s2thetag=s2thetag, thetag=thetag,
        pi0.fit=pi0_fit, etype=est_type, ref.est=ref_type, z.adj=z_adj
    ))
    norm_factors <- rep(1, ncol(counts))
    names(norm_factors) <- colnames(counts)
    norm_factors[names(W$ccf)] <- W$ccf
    norm_factors[is.na(norm_factors)] <- 1
    dge <- DGEList(counts=counts, genes=feature_meta, norm.factors=norm_factors)
    if (trans_type == "cpm") {
        tmat <- cpm(dge, log=log, prior.count=prior_count)
    } else if (trans_type == "tpm") {
        stopifnot(!is.null(feature_meta))
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
            tmat <- log2(t(t(fpkms) / colSums(fpkms)) * 1e6)
        } else {
            fpkms <- rpkm(
                dge, gene.length=gene_length_col, log=log,
                prior.count=prior_count
            )
            tmat <- t(t(fpkms) / colSums(fpkms)) * 1e6
        }
    }
    Xt <- t(tmat)
    if (is.data.frame(X)) {
        Xt <- as.data.frame(Xt)
        row.names(Xt) <- rnames
        colnames(Xt) <- cnames
    }
    return(Xt)
}
