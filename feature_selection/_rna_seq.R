# RNA-seq feature selection and scoring functions

deseq2_feature_score <- function(
    X, y, sample_meta=NULL, lfc=0, scoring_meth="pv", fit_type="parametric",
    lfc_shrink=TRUE, lfc_shrink_type="apeglm", svalue=FALSE, model_batch=FALSE,
    n_threads=1
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
        colData <- as.data.frame(sample_meta)
        design <- ~Batch + Class
    } else {
        colData <- data.frame(Class=factor(y))
        design <- ~Class
    }
    dds <- DESeqDataSetFromMatrix(counts, colData, design)
    suppressMessages(
        dds <- DESeq(dds, fitType=fit_type, parallel=parallel, quiet=TRUE)
    )
    if (lfc_shrink) {
        suppressMessages(results <- as.data.frame(lfcShrink(
            dds, coef=length(resultsNames(dds)), type=lfc_shrink_type,
            lfcThreshold=lfc, svalue=svalue, parallel=parallel, quiet=TRUE
        )))
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
            lfcThreshold=lfc, altHypothesis="greaterAbs", pAdjustMethod="BH"
        ))
        results$padj[is.na(results$padj)] <- 1
    }
    if (scoring_meth == "lfc_pv") {
        scores <- abs(results$log2FoldChange) * -log10(results$pvalue)
    } else {
        scores <- results$pvalue
    }
    return(data.frame(score=scores, padj=results$padj))
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
    if (is.null(row.names(counts))) {
        row.names(counts) <- seq_len(nrow(counts))
    }
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
    zinb <- zinbwave(
        SummarizedExperiment(
            assays=list(counts=counts[rowSums(counts) > 0, ]), colData=colData
        ),
        X=design, K=K, epsilon=epsilon, zeroinflation=TRUE,
        observationalWeights=TRUE
    )
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
    zero_rownames <- row.names(counts)[rowSums(counts) == 0]
    zero_rowlen <- length(zero_rownames)
    if (zero_rowlen > 0) {
        results <- rbind(
            results,
            data.frame(
                row.names=zero_rownames,
                baseMean=rep(0, zero_rowlen),
                log2FoldChange=rep(0, zero_rowlen),
                lfcSE=rep(0, zero_rowlen),
                stat=rep(0, zero_rowlen),
                pvalue=rep(1, zero_rowlen),
                padj=rep(1, zero_rowlen)
            )
        )
        results <- results[match(row.names(counts), row.names(results)), ]
    }
    if (scoring_meth == "lfc_pv") {
        scores <- abs(results$log2FoldChange) * -log10(results$pvalue)
    } else {
        scores <- results$pvalue
    }
    return(data.frame(score=scores, padj=results$padj))
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
    suppressWarnings(dge <- calcNormFactors(dge, method="TMM"))
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
    if (scoring_meth == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$PValue)
    } else {
        scores <- results$PValue
    }
    return(data.frame(score=scores, padj=results$FDR))
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
    if (is.null(row.names(counts))) {
        row.names(counts) <- seq_len(nrow(counts))
    }
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
    zinb <- zinbwave(
        SummarizedExperiment(
            assays=list(counts=counts[rowSums(counts) > 0, ]), colData=colData
        ),
        X=design_formula, K=K, epsilon=epsilon, zeroinflation=TRUE,
        observationalWeights=TRUE
    )
    design <- model.matrix(design_formula, data=colData)
    dge <- DGEList(counts=assay(zinb, "counts"))
    suppressWarnings(dge <- calcNormFactors(dge, method="TMM"))
    dge$weights <- assay(zinb, "weights")
    dge <- estimateDisp(dge, design, robust=robust)
    fit <- glmFit(dge, design)
    lrt <- glmWeightedF(fit, coef=ncol(design))
    results <- as.data.frame(topTags(
        lrt, n=Inf, adjust.method="BH", sort.by="none"
    ))
    results$PValue[is.na(results$PValue)] <- 1
    results$padjFilter[is.na(results$padjFilter)] <- 1
    results$FDR[is.na(results$FDR)] <- 1
    zero_rownames <- row.names(counts)[rowSums(counts) == 0]
    zero_rowlen <- length(zero_rownames)
    if (zero_rowlen > 0) {
        results <- rbind(
            results,
            data.frame(
                row.names=zero_rownames,
                logFC=rep(0, zero_rowlen),
                logCPM=rep(0, zero_rowlen),
                LR=rep(0, zero_rowlen),
                PValue=rep(1, zero_rowlen),
                padjFilter=rep(1, zero_rowlen),
                FDR=rep(1, zero_rowlen)
            )
        )
        results <- results[match(row.names(counts), row.names(results)), ]
    }
    if (scoring_meth == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$PValue)
    } else {
        scores <- results$PValue
    }
    return(data.frame(score=scores, padj=results$FDR))
}

limma_voom_feature_score <- function(
    X, y, sample_meta=NULL, lfc=0, scoring_meth="pv", robust=TRUE,
    model_batch=FALSE, model_dupcor=FALSE
) {
    suppressPackageStartupMessages(library("edgeR"))
    suppressPackageStartupMessages(library("limma"))
    counts <- t(X)
    if (is.null(row.names(counts))) {
        row.names(counts) <- seq_len(nrow(counts))
    }
    suppressMessages(dge <- DGEList(counts=counts, remove.zeros=TRUE))
    suppressWarnings(dge <- calcNormFactors(dge, method="TMM"))
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
    zero_rownames <- row.names(counts)[rowSums(counts) == 0]
    zero_rowlen <- length(zero_rownames)
    if (zero_rowlen > 0) {
        results <- rbind(
            results,
            data.frame(
                row.names=zero_rownames,
                logFC=rep(0, zero_rowlen),
                AveExpr=rep(0, zero_rowlen),
                t=rep(0, zero_rowlen),
                P.Value=rep(1, zero_rowlen),
                adj.P.Val=rep(1, zero_rowlen)
            )
        )
        results <- results[match(row.names(counts), row.names(results)), ]
    }
    if (scoring_meth == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$P.Value)
    } else {
        scores <- results$P.Value
    }
    return(data.frame(score=scores, padj=results$adj.P.Val))
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
    suppressWarnings(dge <- calcNormFactors(dge, method="TMM"))
    if (model_batch && length(unique(sample_meta$Batch)) > 1) {
        design_formula <- ~Batch + Class + (1|Group)
        sample_meta$Batch <- factor(sample_meta$Batch)
    } else {
        design_formula <- ~Class + (1|Group)
    }
    sample_meta$Class <- factor(sample_meta$Class)
    sample_meta$Group <- factor(sample_meta$Group)
    invisible(capture.output(
        v <- voomWithDreamWeights(dge, design_formula, sample_meta)
    ))
    invisible(capture.output(
        fit <- dream(v, design_formula, sample_meta, suppressWarnings=TRUE)
    ))
    results <- topTable(
        fit, coef=ncol(design), lfc=lfc, number=Inf, adjust.method="BH",
        sort.by="none"
    )
    if (scoring_meth == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$P.Value)
    } else {
        scores <- results$P.Value
    }
    return(data.frame(score=scores, padj=results$adj.P.Val))
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
    if (scoring_meth == "lfc_pv") {
        scores <- abs(results$logFC) * -log10(results$P.Value)
    } else {
        scores <- results$P.Value
    }
    return(data.frame(score=scores, padj=results$adj.P.Val))
}
