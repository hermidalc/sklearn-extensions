# Count data feature scoring and normalization fit functions

source(paste(
    dirname(sys.frame(1)$ofile), "../preprocessing/_wrench.R", sep="/"
))

deseq2_feature_score <- function(
    X, y, sample_meta=NULL, norm_type="ratio", fit_type="parametric",
    score_type="pv", lfc=0, lfc_shrink=TRUE, lfc_shrink_type="apeglm",
    svalue=FALSE, model_batch=FALSE, n_threads=1
) {
    suppressPackageStartupMessages(library("DESeq2"))
    if (
        model_batch && !is.null(sample_meta) &&
        length(unique(sample_meta$Batch)) > 1
    ) {
        sample_meta$Batch <- factor(sample_meta$Batch)
        sample_meta$Class <- factor(sample_meta$Class)
        colData <- as.data.frame(sample_meta)
        design <- ~Batch + Class
    } else {
        colData <- data.frame(Class=factor(y))
        design <- ~Class
    }
    counts <- t(X)
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
    if (n_threads > 1) {
        BPPARAM <- BiocParallel::MulticoreParam(workers=n_threads)
        parallel <- TRUE
    } else {
        BPPARAM <- BiocParallel::SerialParam()
        parallel <- FALSE
    }
    suppressMessages({
        dds <- estimateSizeFactors(
            dds, type=norm_type, locfunc=locfunc, quiet=TRUE
        )
        dds <- DESeq(
            dds, fitType=fit_type, sfType=norm_type, parallel=parallel,
            BPPARAM=BPPARAM, quiet=TRUE
        )
    })
    if (lfc_shrink) {
        suppressMessages(
            results <- as.data.frame(lfcShrink(
                dds, coef=length(resultsNames(dds)), type=lfc_shrink_type,
                lfcThreshold=lfc, svalue=svalue, parallel=parallel,
                BPPARAM=BPPARAM, quiet=TRUE
            ))
        )
        if (svalue) {
            results$svalue[is.na(results$svalue)] <- 1
            results$pvalue <- results$svalue
            results$padj <- results$svalue
        } else {
            results$padj[is.na(results$padj)] <- 1
        }
    } else {
        results <- as.data.frame(results(
            dds, name=resultsNames(dds)[length(resultsNames(dds))],
            lfcThreshold=lfc, altHypothesis="greaterAbs", cooksCutoff=FALSE,
            independentFiltering=FALSE, pAdjustMethod="BH"
        ))
        results$padj[is.na(results$padj)] <- 1
    }
    if (score_type == "lfc_pv") {
        scores <- abs(results$log2FoldChange) * -log10(results$pvalue)
    } else {
        scores <- results$pvalue
    }
    return(list(
        scores=scores, padj=results$padj, geo_means=geo_means,
        disp_func=dispersionFunction(dds)
    ))
}

deseq2_wrench_feature_score <- function(
    X, y, sample_meta=NULL, est_type="w.marg.mean", ref_type="sw.means",
    z_adj=FALSE, fit_type="parametric", score_type="pv", lfc=0,
    lfc_shrink=TRUE, lfc_shrink_type="apeglm", svalue=FALSE, n_threads=1
) {
    suppressPackageStartupMessages(library("DESeq2"))
    counts <- t(X)
    if (is.null(colnames(counts))) {
        colnames(counts) <- paste0("X", seq_len(ncol(counts)))
    }
    condition <- as.vector(y)
    suppressWarnings(W <- wrench(
        counts, condition, etype=est_type, ref.est=ref_type, z.adj=z_adj
    ))
    size_factors <- rep(1, ncol(counts))
    names(size_factors) <- colnames(counts)
    size_factors[names(W$nf)] <- W$nf
    size_factors[is.na(size_factors)] <- 1
    colData <- data.frame(Class=factor(y))
    design <- ~Class
    dds <- DESeqDataSetFromMatrix(counts, colData, design)
    sizeFactors(dds) <- size_factors
    if (n_threads > 1) {
        BPPARAM <- BiocParallel::MulticoreParam(workers=n_threads)
        parallel <- TRUE
    } else {
        BPPARAM <- BiocParallel::SerialParam()
        parallel <- FALSE
    }
    suppressMessages(dds <- DESeq(
        dds, fitType=fit_type, parallel=parallel, BPPARAM=BPPARAM, quiet=TRUE
    ))
    if (lfc_shrink) {
        suppressMessages(
            results <- as.data.frame(lfcShrink(
                dds, coef=length(resultsNames(dds)), type=lfc_shrink_type,
                lfcThreshold=lfc, svalue=svalue, parallel=parallel,
                BPPARAM=BPPARAM, quiet=TRUE
            ))
        )
        if (svalue) {
            results$svalue[is.na(results$svalue)] <- 1
            results$pvalue <- results$svalue
            results$padj <- results$svalue
        } else {
            results$padj[is.na(results$padj)] <- 1
        }
    } else {
        results <- as.data.frame(results(
            dds, name=resultsNames(dds)[length(resultsNames(dds))],
            lfcThreshold=lfc, altHypothesis="greaterAbs", cooksCutoff=FALSE,
            independentFiltering=FALSE, pAdjustMethod="BH"
        ))
        results$padj[is.na(results$padj)] <- 1
    }
    if (score_type == "lfc_pv") {
        scores <- abs(results$log2FoldChange) * -log10(results$pvalue)
    } else {
        scores <- results$pvalue
    }
    return(list(
        scores=scores, padj=results$padj, nzrows=W$others$nzrows,
        qref=W$others$qref, s2=W$others$s2, s2thetag=W$others$s2thetag,
        thetag=W$others$thetag, pi0_fit=W$others$pi0.fit,
        disp_func=dispersionFunction(dds)
    ))
}

deseq2_zinbwave_feature_score <- function(
    X, y, sample_meta=NULL, epsilon=1e12, norm_type="poscounts",
    fit_type="parametric", score_type="pv", lfc=0, lfc_shrink=TRUE,
    lfc_shrink_type="normal", svalue=FALSE, model_batch=FALSE, n_threads=1
) {
    suppressPackageStartupMessages({
        library("SummarizedExperiment")
        library("DESeq2")
    })
    if (
        model_batch && !is.null(sample_meta) &&
        length(unique(sample_meta$Batch)) > 1
    ) {
        sample_meta$Batch <- factor(sample_meta$Batch)
        sample_meta$Class <- factor(sample_meta$Class)
        colData <- as.data.frame(sample_meta)
        design <- ~Batch + Class
    } else {
        colData <- data.frame(Class=factor(y))
        design <- ~Class
    }
    counts <- t(X)
    # adapted from DESeq2::estimateSizeFactors source code
    geoMeanNZ <- function(x) {
        if (all(x == 0)) { 0 }
        else { exp( sum(log(x[x > 0])) / length(x) ) }
    }
    geo_means <- apply(counts, 1, geoMeanNZ)
    if (n_threads > 1) {
        BPPARAM <- BiocParallel::MulticoreParam(workers=n_threads)
        parallel <- TRUE
    } else {
        BPPARAM <- BiocParallel::SerialParam()
        parallel <- FALSE
    }
    suppressWarnings({
        zinb <- zinbwave::zinbwave(
            SummarizedExperiment(assays=list(counts=counts), colData=colData),
            X=design, K=0, epsilon=epsilon, zeroinflation=TRUE,
            observationalWeights=TRUE, BPPARAM=BPPARAM
        )
        dds <- DESeqDataSet(zinb, design)
        suppressMessages({
            dds <- estimateSizeFactors(
                dds, type=norm_type, locfunc=genefilter::shorth, quiet=TRUE
            )
            dds <- DESeq(
                dds, fitType=fit_type, sfType=norm_type, useT=TRUE,
                minmu=1e-6, parallel=parallel, BPPARAM=BPPARAM, quiet=TRUE
            )
        })
    })
    if (lfc_shrink) {
        suppressMessages(
            results <- as.data.frame(lfcShrink(
                dds, coef=length(resultsNames(dds)), type=lfc_shrink_type,
                lfcThreshold=lfc, svalue=svalue, parallel=parallel,
                BPPARAM=BPPARAM, quiet=TRUE
            ))
        )
        if (svalue) {
            results$svalue[is.na(results$svalue)] <- 1
            results$pvalue <- results$svalue
            results$padj <- results$svalue
        } else {
            results$padj[is.na(results$padj)] <- 1
        }
    } else {
        results <- as.data.frame(results(
            dds, name=resultsNames(dds)[length(resultsNames(dds))],
            lfcThreshold=lfc, altHypothesis="greaterAbs", cooksCutoff=FALSE,
            independentFiltering=FALSE, pAdjustMethod="BH"
        ))
        results$padj[is.na(results$padj)] <- 1
    }
    if (score_type == "lfc_pv") {
        scores <- abs(results$log2FoldChange) * -log10(results$pvalue)
    } else {
        scores <- results$pvalue
    }
    return(list(
        scores=scores, padj=results$padj, geo_means=geo_means,
        disp_func=dispersionFunction(dds)
    ))
}

deseq2_wrench_zinbwave_feature_score <- function(
    X, y, sample_meta=NULL, est_type="w.marg.mean", ref_type="sw.means",
    z_adj=FALSE, epsilon=1e12, fit_type="parametric", score_type="pv",
    lfc=0, lfc_shrink=TRUE, lfc_shrink_type="normal", svalue=FALSE, n_threads=1
) {
    suppressPackageStartupMessages({
        library("SummarizedExperiment")
        library("DESeq2")
    })
    counts <- t(X)
    if (is.null(colnames(counts))) {
        colnames(counts) <- paste0("X", seq_len(ncol(counts)))
    }
    condition <- as.vector(y)
    suppressWarnings(W <- wrench(
        counts, condition, etype=est_type, ref.est=ref_type, z.adj=z_adj
    ))
    size_factors <- rep(1, ncol(counts))
    names(size_factors) <- colnames(counts)
    size_factors[names(W$nf)] <- W$nf
    size_factors[is.na(size_factors)] <- 1
    colData <- data.frame(Class=factor(y))
    design <- ~Class
    if (n_threads > 1) {
        BPPARAM <- BiocParallel::MulticoreParam(workers=n_threads)
        parallel <- TRUE
    } else {
        BPPARAM <- BiocParallel::SerialParam()
        parallel <- FALSE
    }
    suppressWarnings({
        zinb <- zinbwave::zinbwave(
            SummarizedExperiment(assays=list(counts=counts), colData=colData),
            X=design, K=0, epsilon=epsilon, zeroinflation=TRUE,
            observationalWeights=TRUE, BPPARAM=BPPARAM
        )
        dds <- DESeqDataSet(zinb, design)
        # use poscounts normalization for DGE feature scoring here but Wrench
        # for downstream normalization
        # sizeFactors(dds) <- size_factors
        suppressMessages({
            dds <- estimateSizeFactors(
                dds, type="poscounts", locfunc=genefilter::shorth, quiet=TRUE
            )
            dds <- DESeq(
                dds, fitType=fit_type, sfType="poscounts", useT=TRUE,
                minmu=1e-6, parallel=parallel, BPPARAM=BPPARAM, quiet=TRUE
            )
        })
    })
    if (lfc_shrink) {
        suppressMessages(
            results <- as.data.frame(lfcShrink(
                dds, coef=length(resultsNames(dds)), type=lfc_shrink_type,
                lfcThreshold=lfc, svalue=svalue, parallel=parallel,
                BPPARAM=BPPARAM, quiet=TRUE
            ))
        )
        if (svalue) {
            results$svalue[is.na(results$svalue)] <- 1
            results$pvalue <- results$svalue
            results$padj <- results$svalue
        } else {
            results$padj[is.na(results$padj)] <- 1
        }
    } else {
        results <- as.data.frame(results(
            dds, name=resultsNames(dds)[length(resultsNames(dds))],
            lfcThreshold=lfc, altHypothesis="greaterAbs", cooksCutoff=FALSE,
            independentFiltering=FALSE, pAdjustMethod="BH"
        ))
        results$padj[is.na(results$padj)] <- 1
    }
    if (score_type == "lfc_pv") {
        scores <- abs(results$log2FoldChange) * -log10(results$pvalue)
    } else {
        scores <- results$pvalue
    }
    return(list(
        scores=scores, padj=results$padj, nzrows=W$others$nzrows,
        qref=W$others$qref, s2=W$others$s2, s2thetag=W$others$s2thetag,
        thetag=W$others$thetag, pi0_fit=W$others$pi0.fit,
        disp_func=dispersionFunction(dds)
    ))
}

edger_filterbyexpr_mask <- function(
    X, y=NULL, sample_meta=NULL, min_count=10, min_total_count=15, large_n=10,
    min_prop=0.7, is_classif=TRUE, model_batch=FALSE
) {
    suppressPackageStartupMessages(library("edgeR"))
    if (
        model_batch && !is.null(sample_meta) &&
        length(unique(sample_meta$Batch)) > 1
    ) {
        sample_meta$Batch <- factor(sample_meta$Batch)
        if (is_classif) {
            sample_meta$Class <- factor(sample_meta$Class)
            design <- model.matrix(~Batch + Class, data=sample_meta)
        } else {
            design <- model.matrix(~Batch, data=sample_meta)
        }
    } else if (is_classif) {
        stopifnot(!is.null(y))
        design <- model.matrix(~factor(y))
    } else {
        design <- NULL
    }
    counts <- t(X)
    dge <- DGEList(counts=counts)
    return(
        suppressMessages(filterByExpr(
            dge, design, min.count=min_count, min.total.count=min_total_count,
            large.n=large_n, min.prop=min_prop
        ))
    )
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

edger_feature_score <- function(
    X, y, sample_meta=NULL, norm_type="TMM", score_type="pv", lfc=0,
    robust=TRUE, model_batch=FALSE
) {
    suppressPackageStartupMessages(library("edgeR"))
    if (
        model_batch && !is.null(sample_meta) &&
        length(unique(sample_meta$Batch)) > 1
    ) {
        sample_meta$Batch <- factor(sample_meta$Batch)
        sample_meta$Class <- factor(sample_meta$Class)
        design <- model.matrix(~Batch + Class, data=sample_meta)
    } else {
        design <- model.matrix(~factor(y))
    }
    counts <- t(X)
    ref_sample <- counts[, edger_tmm_ref_column(counts)]
    dge <- DGEList(counts=counts)
    dge <- calcNormFactors(dge, method=norm_type)
    dge <- estimateDisp(dge, design, robust=robust)
    fit <- glmQLFit(dge, design, robust=robust)
    if (lfc == 0) {
        glt <- glmQLFTest(fit, coef=ncol(design))
    } else {
        glt <- glmTreat(fit, coef=ncol(design), lfc=lfc)
    }
    results <- as.data.frame(topTags(
        glt, n=Inf, adjust.method="BH", sort.by="none"
    ))
    results$PValue[is.na(results$PValue)] <- 1
    results$FDR[is.na(results$FDR)] <- 1
    if (score_type == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$PValue)
    } else {
        scores <- results$PValue
    }
    return(list(scores=scores, padj=results$FDR, ref_sample=ref_sample))
}

edger_wrench_feature_score <- function(
    X, y, sample_meta=NULL, est_type="w.marg.mean", ref_type="sw.means",
    z_adj=FALSE, score_type="pv", lfc=0, robust=TRUE
) {
    suppressPackageStartupMessages(library("edgeR"))
    counts <- t(X)
    if (is.null(colnames(counts))) {
        colnames(counts) <- paste0("X", seq_len(ncol(counts)))
    }
    condition <- as.vector(y)
    suppressWarnings(W <- wrench(
        counts, condition, etype=est_type, ref.est=ref_type, z.adj=z_adj
    ))
    norm_factors <- rep(1, ncol(counts))
    names(norm_factors) <- colnames(counts)
    norm_factors[names(W$ccf)] <- W$ccf
    norm_factors[is.na(norm_factors)] <- 1
    design <- model.matrix(~factor(y))
    dge <- DGEList(counts=counts, norm.factors=norm_factors)
    suppressWarnings({
        dge <- estimateDisp(dge, design, robust=robust)
        fit <- glmQLFit(dge, design, robust=robust)
        if (lfc == 0) {
            glt <- glmQLFTest(fit, coef=ncol(design))
        } else {
            glt <- glmTreat(fit, coef=ncol(design), lfc=lfc)
        }
    })
    results <- as.data.frame(topTags(
        glt, n=Inf, adjust.method="BH", sort.by="none"
    ))
    results$PValue[is.na(results$PValue)] <- 1
    results$FDR[is.na(results$FDR)] <- 1
    if (score_type == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$PValue)
    } else {
        scores <- results$PValue
    }
    return(list(
        scores=scores, padj=results$FDR, nzrows=W$others$nzrows,
        qref=W$others$qref, s2=W$others$s2, s2thetag=W$others$s2thetag,
        thetag=W$others$thetag, pi0_fit=W$others$pi0.fit
    ))
}

edger_zinbwave_feature_score <- function(
    X, y, sample_meta=NULL, epsilon=1e12, norm_type="TMM", score_type="pv",
    robust=TRUE, model_batch=FALSE, n_threads=1
) {
    suppressPackageStartupMessages({
        library("SummarizedExperiment")
        library("edgeR")
    })
    if (
        model_batch && !is.null(sample_meta) &&
        length(unique(sample_meta$Batch)) > 1
    ) {
        sample_meta$Batch <- factor(sample_meta$Batch)
        sample_meta$Class <- factor(sample_meta$Class)
        colData <- as.data.frame(sample_meta)
        design_formula <- ~Batch + Class
    } else {
        colData <- data.frame(Class=factor(y))
        design_formula <- ~Class
    }
    counts <- t(X)
    ref_sample <- counts[, edger_tmm_ref_column(counts)]
    if (n_threads > 1) {
        BPPARAM <- BiocParallel::MulticoreParam(workers=n_threads)
        parallel <- TRUE
    } else {
        BPPARAM <- BiocParallel::SerialParam()
        parallel <- FALSE
    }
    suppressWarnings({
        zinb <- zinbwave::zinbwave(
            SummarizedExperiment(assays=list(counts=counts), colData=colData),
            X=design_formula, K=0, epsilon=epsilon, zeroinflation=TRUE,
            observationalWeights=TRUE, BPPARAM=BPPARAM
        )
        dge <- DGEList(counts=assay(zinb, "counts"))
        dge <- calcNormFactors(dge, method=norm_type)
        dge$weights <- assay(zinb, "weights")
        design <- model.matrix(design_formula, data=colData)
        dge <- estimateDisp(dge, design, robust=robust)
        fit <- glmFit(dge, design)
        lrt <- zinbwave::glmWeightedF(fit, coef=ncol(design))
    })
    results <- as.data.frame(topTags(
        lrt, n=Inf, adjust.method="BH", sort.by="none"
    ))
    results$PValue[is.na(results$PValue)] <- 1
    results$padjFilter[is.na(results$padjFilter)] <- 1
    results$FDR[is.na(results$FDR)] <- 1
    if (score_type == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$PValue)
    } else {
        scores <- results$PValue
    }
    return(list(scores=scores, padj=results$FDR, ref_sample=ref_sample))
}

edger_wrench_zinbwave_feature_score <- function(
    X, y, sample_meta=NULL, est_type="w.marg.mean", ref_type="sw.means",
    z_adj=FALSE, epsilon=1e12, score_type="pv", robust=TRUE, n_threads=1
) {
    suppressPackageStartupMessages({
        library("SummarizedExperiment")
        library("edgeR")
    })
    counts <- t(X)
    if (is.null(colnames(counts))) {
        colnames(counts) <- paste0("X", seq_len(ncol(counts)))
    }
    condition <- as.vector(y)
    suppressWarnings(W <- wrench(
        counts, condition, etype=est_type, ref.est=ref_type, z.adj=z_adj
    ))
    norm_factors <- rep(1, ncol(counts))
    names(norm_factors) <- colnames(counts)
    norm_factors[names(W$ccf)] <- W$ccf
    norm_factors[is.na(norm_factors)] <- 1
    colData <- data.frame(Class=factor(y))
    design_formula <- ~Class
    if (n_threads > 1) {
        BPPARAM <- BiocParallel::MulticoreParam(workers=n_threads)
        parallel <- TRUE
    } else {
        BPPARAM <- BiocParallel::SerialParam()
        parallel <- FALSE
    }
    suppressWarnings({
        zinb <- zinbwave::zinbwave(
            SummarizedExperiment(assays=list(counts=counts), colData=colData),
            X=design_formula, K=0, epsilon=epsilon, zeroinflation=TRUE,
            observationalWeights=TRUE, BPPARAM=BPPARAM
        )
        # use TMM normalization for DGE feature scoring here but Wrench
        # for downstream normalization
        # dge <- DGEList(
        #     counts=assay(zinb, "counts"), norm.factors=norm_factors
        # )
        dge <- DGEList(counts=assay(zinb, "counts"))
        dge <- calcNormFactors(dge, method="TMM")
        dge$weights <- assay(zinb, "weights")
        design <- model.matrix(design_formula, data=colData)
        dge <- estimateDisp(dge, design, robust=robust)
        fit <- glmFit(dge, design)
        lrt <- zinbwave::glmWeightedF(fit, coef=ncol(design))
    })
    results <- as.data.frame(topTags(
        lrt, n=Inf, adjust.method="BH", sort.by="none"
    ))
    results$PValue[is.na(results$PValue)] <- 1
    results$padjFilter[is.na(results$padjFilter)] <- 1
    results$FDR[is.na(results$FDR)] <- 1
    if (score_type == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$PValue)
    } else {
        scores <- results$PValue
    }
    return(list(
        scores=scores, padj=results$FDR, nzrows=W$others$nzrows,
        qref=W$others$qref, s2=W$others$s2, s2thetag=W$others$s2thetag,
        thetag=W$others$thetag, pi0_fit=W$others$pi0.fit
    ))
}

limma_voom_feature_score <- function(
    X, y, sample_meta=NULL, norm_type="TMM", score_type="pv", lfc=0,
    robust=TRUE, model_batch=FALSE, model_dupcor=FALSE
) {
    suppressPackageStartupMessages({
        library("edgeR")
        library("limma")
    })
    counts <- t(X)
    dge <- DGEList(counts=counts)
    ref_sample <- counts[, edger_tmm_ref_column(counts)]
    dge <- calcNormFactors(dge, method=norm_type)
    if ((model_batch || model_dupcor) && !is.null(sample_meta)) {
        sample_meta$Class <- factor(sample_meta$Class)
        if (model_batch && length(unique(sample_meta$Batch)) > 1) {
            sample_meta$Batch <- factor(sample_meta$Batch)
            design_formula <- ~Batch + Class
        } else {
            design_formula <- ~Class
        }
        design <- model.matrix(design_formula, data=sample_meta)
        v <- voom(dge, design)
        if (model_dupcor) {
            sample_meta$Group <- factor(sample_meta$Group)
            suppressMessages(dupcor <- duplicateCorrelation(
                v, design, block=sample_meta$Group
            ))
            v <- voom(
                dge, design, block=sample_meta$Group,
                correlation=dupcor$consensus
            )
            suppressMessages(dupcor <- duplicateCorrelation(
                v, design, block=sample_meta$Group
            ))
            fit <- lmFit(
                v, design, block=sample_meta$Group,
                correlation=dupcor$consensus
            )
        } else {
            fit <- lmFit(v, design)
        }
    } else {
        design <- model.matrix(~factor(y))
        v <- voom(dge, design)
        fit <- lmFit(v, design)
    }
    fit <- treat(fit, lfc=lfc, robust=robust, trend=FALSE)
    results <- topTreat(
        fit, coef=ncol(design), number=Inf, adjust.method="BH", sort.by="none"
    )
    if (score_type == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$P.Value)
    } else {
        scores <- results$P.Value
    }
    return(list(scores=scores, padj=results$adj.P.Val, ref_sample=ref_sample))
}

limma_voom_wrench_feature_score <- function(
    X, y, sample_meta=NULL, est_type="w.marg.mean", ref_type="sw.means",
    z_adj=FALSE, score_type="pv", lfc=0, robust=TRUE
) {
    suppressPackageStartupMessages({
        library("edgeR")
        library("limma")
    })
    counts <- t(X)
    if (is.null(colnames(counts))) {
        colnames(counts) <- paste0("X", seq_len(ncol(counts)))
    }
    condition <- as.vector(y)
    suppressWarnings(W <- wrench(
        counts, condition, etype=est_type, ref.est=ref_type, z.adj=z_adj
    ))
    norm_factors <- rep(1, ncol(counts))
    names(norm_factors) <- colnames(counts)
    norm_factors[names(W$ccf)] <- W$ccf
    norm_factors[is.na(norm_factors)] <- 1
    design <- model.matrix(~factor(y))
    dge <- DGEList(counts=counts, norm.factors=norm_factors)
    suppressWarnings({
        v <- voom(dge, design)
        fit <- lmFit(v, design)
        fit <- treat(fit, lfc=lfc, robust=robust, trend=FALSE)
    })
    results <- topTreat(
        fit, coef=ncol(design), number=Inf, adjust.method="BH", sort.by="none"
    )
    if (score_type == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$P.Value)
    } else {
        scores <- results$P.Value
    }
    return(list(scores=scores, padj=results$adj.P.Val, nzrows=W$others$nzrows,
        qref=W$others$qref, s2=W$others$s2, s2thetag=W$others$s2thetag,
        thetag=W$others$thetag, pi0_fit=W$others$pi0.fit
    ))
}

dream_voom_feature_score <- function(
    X, y, sample_meta, norm_type="TMM", score_type="pv", lfc=0,
    model_batch=FALSE, n_threads=1
) {
    suppressPackageStartupMessages({
        library("edgeR")
        library("limma")
        library("variancePartition")
    })
    if (model_batch && length(unique(sample_meta$Batch)) > 1) {
        design_formula <- ~Batch + Class + (1|Group)
        sample_meta$Batch <- factor(sample_meta$Batch)
    } else {
        design_formula <- ~Class + (1|Group)
    }
    sample_meta$Class <- factor(sample_meta$Class)
    sample_meta$Group <- factor(sample_meta$Group)
    if (n_threads > 1) {
        BPPARAM <- BiocParallel::MulticoreParam(workers=n_threads)
    } else {
        BPPARAM <- BiocParallel::SerialParam()
    }
    counts <- t(X)
    ref_sample <- counts[, edger_tmm_ref_column(counts)]
    dge <- DGEList(counts=counts)
    dge <- calcNormFactors(dge, method=norm_type)
    invisible(capture.output(
        v <- voomWithDreamWeights(
            dge, design_formula, sample_meta, BPPARAM=BPPARAM
        )
    ))
    invisible(capture.output(
        fit <- dream(
            v, design_formula, sample_meta, BPPARAM=BPPARAM,
            suppressWarnings=TRUE
        )
    ))
    results <- topTable(
        fit, coef=ncol(design), lfc=lfc, number=Inf, adjust.method="BH",
        sort.by="none"
    )
    if (score_type == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$P.Value)
    } else {
        scores <- results$P.Value
    }
    return(list(scores=scores, padj=results$adj.P.Val, ref_sample=ref_sample))
}

limma_feature_score <- function(
    X, y, sample_meta=NULL, score_type="pv", lfc=0, robust=FALSE, trend=FALSE,
    model_batch=FALSE
) {
    suppressPackageStartupMessages(library("limma"))
    if (
        model_batch && !is.null(sample_meta) &&
        length(unique(sample_meta$Batch)) > 1
    ) {
        sample_meta$Batch <- factor(sample_meta$Batch)
        sample_meta$Class <- factor(sample_meta$Class)
        design <- model.matrix(~Batch + Class, data=sample_meta)
    } else {
        design <- model.matrix(~factor(y))
    }
    fit <- lmFit(t(X), design)
    fit <- treat(fit, lfc=lfc, robust=robust, trend=trend)
    results <- topTreat(
        fit, coef=ncol(design), number=Inf, adjust.method="BH", sort.by="none"
    )
    if (score_type == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$P.Value)
    } else {
        scores <- results$P.Value
    }
    return(list(scores=scores, padj=results$adj.P.Val))
}
