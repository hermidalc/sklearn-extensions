# RNA-seq feature selection and scoring functions

deseq2_feature_score <- function(
    X, y, sample_meta=NULL, lfc=0, fit_type="parametric", lfc_shrink=TRUE,
    blind=FALSE, model_batch=FALSE, n_threads=1
) {
    suppressPackageStartupMessages(library("DESeq2"))
    suppressPackageStartupMessages(library("BiocParallel"))
    if (n_threads > 1) {
        register(MulticoreParam(workers=n_threads))
        parallel <- TRUE
    } else {
        register(SerialParam())
        parallel <- FALSE
    }
    counts <- t(X)
    if (
        model_batch && !is.null(sample_meta) &&
        length(unique(sample_meta$Batch)) > 1
    ) {
        sample_meta$Batch <- factor(sample_meta$Batch)
        sample_meta$Class <- factor(sample_meta$Class)
        dds <- DESeqDataSetFromMatrix(
            counts, as.data.frame(sample_meta), ~Batch + Class
        )
    } else {
        dds <- DESeqDataSetFromMatrix(
            counts, data.frame(Class=factor(y)), ~Class
        )
    }
    suppressMessages(
        dds <- DESeq(dds, fitType=fit_type, parallel=parallel, quiet=TRUE)
    )
    if (lfc_shrink) {
        suppressMessages(results <- as.data.frame(lfcShrink(
            dds, coef=length(resultsNames(dds)), type="apeglm",
            lfcThreshold=lfc, svalue=TRUE, parallel=parallel, quiet=TRUE
        )))
        results$svalue[is.na(results$svalue)] <- 1
        results$pvalue <- results$svalue
        results$padj <- results$svalue
    } else {
        results <- as.data.frame(results(
            dds, name=resultsNames(dds)[length(resultsNames(dds))],
            lfcThreshold=lfc, altHypothesis="greaterAbs", pAdjustMethod="BH"
        ))
        results$padj[is.na(results$padj)] <- 1
    }
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    vsd <- varianceStabilizingTransformation(dds, blind=blind)
    geo_means <- exp(rowMeans(log(counts)))
    return(list(
        results$pvalue, results$padj, t(assay(vsd)), geo_means,
        dispersionFunction(dds)
    ))
}

# adapted from edgeR::calcNormFactors source code
edger_tmm_ref_column <- function(counts, lib.size=colSums(counts), p=0.75) {
    y <- t(t(counts) / lib.size)
    f <- apply(y, 2, function(x) quantile(x, p=p))
    ref_column <- which.min(abs(f - mean(f)))
}

edger_feature_score <- function(
    X, y, sample_meta=NULL, lfc=0, robust=TRUE, prior_count=1, model_batch=FALSE
) {
    suppressPackageStartupMessages(library("edgeR"))
    counts <- t(X)
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
    dge <- DGEList(counts=counts)
    dge <- calcNormFactors(dge, method="TMM")
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
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    log_cpm <- cpm(dge, log=TRUE, prior.count=prior_count)
    ref_sample <- counts[, edger_tmm_ref_column(counts)]
    return(list(results$PValue, results$FDR, t(log_cpm), ref_sample))
}

edger_filterbyexpr_mask <- function(
    X, y, sample_meta=NULL, model_batch=FALSE, is_classif=TRUE
) {
    suppressPackageStartupMessages(library("edgeR"))
    dge <- DGEList(counts=t(X))
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
        design <- model.matrix(~factor(y))
    } else {
        design <- NULL
    }
    return(filterByExpr(dge, design))
}

limma_voom_feature_score <- function(
    X, y, sample_meta=NULL, lfc=0, robust=TRUE, prior_count=1,
    model_batch=FALSE, model_dupcor=FALSE
) {
    suppressPackageStartupMessages(library("edgeR"))
    suppressPackageStartupMessages(library("limma"))
    counts <- t(X)
    dge <- DGEList(counts=counts)
    dge <- calcNormFactors(dge, method="TMM")
    if ((model_batch || model_dupcor) && !is.null(sample_meta)) {
        if (model_batch && length(unique(sample_meta$Batch)) > 1) {
            formula <- ~Batch + Class
            sample_meta$Batch <- factor(sample_meta$Batch)
        } else {
            formula <- ~Class
        }
        sample_meta$Class <- factor(sample_meta$Class)
        design <- model.matrix(formula, data=sample_meta)
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
    fit <- treat(fit, lfc=lfc, robust=robust)
    results <- topTreat(
        fit, coef=ncol(design), number=Inf, adjust.method="BH", sort.by="none"
    )
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    log_cpm <- cpm(dge, log=TRUE, prior.count=prior_count)
    ref_sample <- counts[, edger_tmm_ref_column(counts)]
    return(list(results$P.Value, results$adj.P.Val, t(log_cpm), ref_sample))
}

dream_voom_feature_score <- function(
    X, y, sample_meta, lfc=0, prior_count=1, model_batch=FALSE, n_threads=1
) {
    suppressPackageStartupMessages(library("edgeR"))
    suppressPackageStartupMessages(library("limma"))
    suppressPackageStartupMessages(library("variancePartition"))
    suppressPackageStartupMessages(library("BiocParallel"))
    if (n_threads > 1) {
        register(MulticoreParam(workers=n_threads))
    } else {
        register(SerialParam())
    }
    counts <- t(X)
    dge <- DGEList(counts=counts)
    dge <- calcNormFactors(dge, method="TMM")
    if (model_batch && length(unique(sample_meta$Batch)) > 1) {
        formula <- ~Batch + Class + (1|Group)
        sample_meta$Batch <- factor(sample_meta$Batch)
    } else {
        formula <- ~Class + (1|Group)
    }
    sample_meta$Class <- factor(sample_meta$Class)
    sample_meta$Group <- factor(sample_meta$Group)
    invisible(capture.output(
        v <- voomWithDreamWeights(dge, formula, sample_meta)
    ))
    invisible(capture.output(
        fit <- dream(v, formula, sample_meta, suppressWarnings=TRUE)
    ))
    results <- topTable(
        fit, coef=ncol(design), lfc=lfc, number=Inf, adjust.method="BH",
        sort.by="none"
    )
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    log_cpm <- cpm(dge, log=TRUE, prior.count=prior_count)
    ref_sample <- counts[, edger_tmm_ref_column(counts)]
    return(list(results$P.Value, results$adj.P.Val, t(log_cpm), ref_sample))
}

limma_feature_score <- function(
    X, y, sample_meta=NULL, lfc=0, robust=FALSE, trend=FALSE, model_batch=FALSE
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
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    return(list(results$P.Value, results$adj.P.Val))
}
