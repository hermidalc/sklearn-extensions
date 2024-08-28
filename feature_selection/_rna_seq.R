# RNA-seq feature selection and scoring functions

deseq2_feature_score <- function(
    X, y, sample_meta=NULL, lfc=0, scoring_meth="pv", fit_type="parametric",
    lfc_shrink=TRUE, model_batch=FALSE, n_threads=1
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
    if (scoring_meth == "lfc_pv") {
        scores <- abs(results$log2FoldChange) * -log10(results$pvalue)
    } else {
        scores <- results$pvalue
    }
    return(list(scores, results$padj))
}

deseq2_zinbwave_feature_score <- function(
    X, y, sample_meta=NULL, lfc=0, scoring_meth="pv", K=0, epsilon=1e12,
    fit_type="parametric", model_batch=FALSE, n_threads=1
) {
    suppressPackageStartupMessages(library("SummarizedExperiment"))
    suppressPackageStartupMessages(library("zinbwave"))
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
        zinb <- zinbwave(
            SummarizedExperiment(
                assays=list(counts=counts), colData=as.data.frame(sample_meta)
            ),
            K=K, epsilon=epsilon, observationalWeights=TRUE
        )
        design <- ~Batch + Class
    } else {
        zinb <- zinbwave(
            SummarizedExperiment(
                assays=list(counts=counts), colData=data.frame(Class=factor(y))
            ),
            K=K, epsilon=epsilon, observationalWeights=TRUE
        )
        design <- ~Class
    }
    dds <- DESeqDataSet(zinb, design)
    suppressMessages(dds <- DESeq(
        dds, fitType=fit_type, sfType="poscounts", useT=TRUE, minmu=1e-6,
        parallel=parallel, quiet=TRUE
    ))
    suppressMessages(results <- as.data.frame(lfcShrink(
        dds, coef=length(resultsNames(dds)), type="normal", lfcThreshold=lfc,
        parallel=parallel, quiet=TRUE
    )))
    results$padj[is.na(results$padj)] <- 1
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    if (scoring_meth == "lfc_pv") {
        scores <- abs(results$log2FoldChange) * -log10(results$pvalue)
    } else {
        scores <- results$pvalue
    }
    return(list(scores, results$padj))
}

edger_filterbyexpr_mask <- function(
    X, y=NULL, sample_meta=NULL, min_count=10, min_total_count=15, large_n=10,
    min_prop=0.7, is_classif=TRUE, model_batch=FALSE
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
        stopifnot(!is.null(y))
        design <- model.matrix(~factor(y))
    } else {
        design <- NULL
    }
    return(
        suppressMessages(filterByExpr(
            dge, design, min.count=min_count, min.total.count=min_total_count,
            large.n=large_n, min.prop=min_prop
        ))
    )
}

edger_feature_score <- function(
    X, y, sample_meta=NULL, lfc=0, scoring_meth="pv", robust=TRUE,
    model_batch=FALSE
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
    if (scoring_meth == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$PValue)
    } else {
        scores <- results$PValue
    }
    return(list(scores, results$FDR))
}

edger_zinbwave_feature_score <- function(
    X, y, sample_meta=NULL, scoring_meth="pv", K=0, epsilon=1e12, robust=TRUE,
    model_batch=FALSE, n_threads=1
) {
    suppressPackageStartupMessages(library("SummarizedExperiment"))
    suppressPackageStartupMessages(library("zinbwave"))
    suppressPackageStartupMessages(library("edgeR"))
    suppressPackageStartupMessages(library("BiocParallel"))
    if (n_threads > 1) {
        register(MulticoreParam(workers=n_threads))
    } else {
        register(SerialParam())
    }
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
    zinb <- zinbwave(
        SummarizedExperiment(assays=list(counts=counts), colData=sample_meta),
        K=K, epsilon=epsilon, observationalWeights=TRUE
    )
    dge <- DGEList(counts=assay(zinb))
    dge$weights <- assay(zinb, "weights")
    dge <- calcNormFactors(dge, method="TMM")
    dge <- estimateDisp(dge, design, robust=robust)
    fit <- glmFit(dge, design)
    lrt <- glmWeightedF(fit, coef=ncol(design))
    results <- as.data.frame(topTags(
        lrt, n=Inf, adjust.method="BH", sort.by="none"
    ))
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    if (scoring_meth == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$PValue)
    } else {
        scores <- results$PValue
    }
    return(list(scores, results$FDR))
}

limma_voom_feature_score <- function(
    X, y, sample_meta=NULL, lfc=0, scoring_meth="pv", robust=TRUE,
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
    if (scoring_meth == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$P.Value)
    } else {
        scores <- results$P.Value
    }
    return(list(scores, results$adj.P.Val))
}

dream_voom_feature_score <- function(
    X, y, sample_meta, lfc=0, scoring_meth="pv", model_batch=FALSE, n_threads=1
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
    if (scoring_meth == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$P.Value)
    } else {
        scores <- results$P.Value
    }
    return(list(scores, results$adj.P.Val))
}

limma_feature_score <- function(
    X, y, sample_meta=NULL, lfc=0, scoring_meth="pv", robust=FALSE, trend=FALSE,
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
    results <- results[order(as.integer(row.names(results))), , drop=FALSE]
    if (scoring_meth == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$P.Value)
    } else {
        scores <- results$P.Value
    }
    return(list(scores, results$adj.P.Val))
}
