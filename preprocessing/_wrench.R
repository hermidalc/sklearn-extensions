# Adapted from https://github.com/HCBravoLab/Wrench/blob/master/R/wrenchSource.R
# Normalization for sparse, under-sampled count data

.getThetag <- function(mat, group, qref) {
    p <- nrow(mat)
    # group-wise ratios
    Yg <- vapply(unique(group), function(g) {
        g_idx <- which(group == g)
        ng <- sum(group == g)
        if (ng > 1) {
            rowSums(mat[, g_idx])
        } else {
            mat[, g_idx]
        }
    }, FUN.VALUE = numeric(p))
    qg <- sweep(Yg, 2, colSums(Yg), "/") # weighted estimator
    rg <- qg / qref
    lrg <- log(rg)
    lrg[!is.finite(lrg)] <- NA
    s2thetag <- matrixStats::colVars(lrg, na.rm = TRUE)
    thetag <- colMeans(rg)
    return(list(s2thetag = s2thetag, thetag = thetag, rg = rg))
}

wrench <- function(
    mat, condition, qref = NULL, s2 = NULL, s2thetag = NULL, thetag = NULL,
    etype = "w.marg.mean", ebcf = TRUE,
    z.adj = FALSE, phi.adj = TRUE, detrend = FALSE, ...
) {
    # trim
    # nzrows <- rowSums(mat) > 0
    # mat <- mat[nzrows, ]
    # nzcols <- colSums(mat) > 0
    # mat <- mat[, nzcols]
    # condition <- condition[nzcols]

    # stopifnot(all(rowSums(mat) > 0))
    stopifnot(all(colSums(mat) > 0))
    stopifnot(ncol(mat) == length(condition))
    stopifnot((nrow(mat) == length(qref)) && (nrow(mat) == length(s2)))

    # feature-wise parameters: hurdle, variance, reference and raw ratios
    n <- ncol(mat)
    p <- nrow(mat)
    tots <- colSums(mat)

    compute.pi0 <- !((etype %in% c("mean", "median", "s2.w.mean")) & !z.adj)
    if (compute.pi0) {
        pi0 <- Wrench:::.getHurdle(mat, ...)$pi0
    }

    group <- as.character(condition)
    if (length(unique(group)) == 1) {
        design <- model.matrix(mat[1, ] ~ 1)
    } else {
        design <- model.matrix(~ -1 + group)
    }

    # variances
    if (is.null(s2)) {
        s2 <- Wrench:::.gets2(mat, design, ...)
    }

    # reference
    if (is.null(qref)) {
        qref <- Wrench:::.getReference(mat, ...)
    }

    # sample-wise ratios
    qmat <- sweep(mat, 2, colSums(mat), "/")
    r <- qmat / qref

    if (ebcf) {
        if (is.null(s2thetag) && is.null(thetag)) {
            tgres <- .getThetag(mat, group, qref)
            s2thetag <- tgres$s2thetag
            thetag <- tgres$thetag
            # rg <- tgres$rg
        }
        s2thetag_rep <- design %*% s2thetag
        thetag_rep <- c(design %*% thetag)

        # regularized estimation of positive means.
        r <- sweep(r, 2, thetag_rep, "/")
        thetagj <- exp(vapply(seq(n), function(j) {
            x <- log(r[, j])
            x[!is.finite(x)] <- NA
            stats::weighted.mean(
                x,
                w = 1 / (s2 + s2thetag_rep[j]), na.rm = TRUE
            )
        }, FUN.VALUE = numeric(1)))

        thetagi <- t(vapply(seq(p), function(i) {
            exp((s2thetag_rep / (s2[i] + s2thetag_rep)) *
                (log(r[i, ]) - log(thetagj)))
        }, FUN.VALUE = numeric(n)))

        r <- sweep(thetagi, 2, thetagj * thetag_rep, "*")
    }

    # adjustments for marginal, and truncated means
    phi2 <- exp(s2)
    radj <- r
    if (z.adj) {
        radj <- radj / (1 - pi0)
    }
    if (phi.adj) {
        radj <- sweep(radj, 1, sqrt(phi2), "/")
    }

    # return result structure
    res <- list()
    res$others <- list()
    if (ebcf) {
        res$others <- list(
            # "rg" = rg,
            "thetag" = thetag,
            "thetagi" = thetagi,
            "s2thetag" = s2thetag
        )
    }
    res$others <- c(
        res$others,
        list(
            "qref" = qref,
            "design" = design,
            "s2" = s2,
            "r" = r,
            "radj" = radj
        )
    )
    if (compute.pi0) {
        res$others <- c(res$others, list("pi0" = pi0))
    }

    res$ccf <- Wrench:::.estimSummary(
        res,
        estim.type = etype, z.adj = z.adj, ...
    )
    res$ccf <- with(res, ccf / exp(mean(log(ccf))))

    if (detrend) {
        res$others$ccf0 <- res$ccf
        detrended <- Wrench:::.detrend.ccf(res$ccf, tots, condition)
        res$others$ccf.detr.un <- detrended$ccf.detr.un
        res$ccf <- detrended$ccf.detr
    }

    tjs <- colSums(mat) / exp(mean(log(colSums(mat))))
    names(res$ccf) <- names(tjs)
    res$nf <- res$ccf * tjs

    res
}

wrench_fit <- function(X, sample_meta, ref_type = "sw.means") {
    counts <- t(X)
    nzrows <- rowSums(counts) > 0
    counts <- counts[nzrows, ]
    nzcols <- colSums(counts) > 0
    counts <- counts[, nzcols]
    group <- as.character(sample_meta$Class)
    group <- group[nzcols]
    if (length(unique(group)) == 1) {
        design <- model.matrix(counts[1, ] ~ 1)
    } else {
        design <- model.matrix(~ -1 + group)
    }
    suppressWarnings(s2 <- Wrench:::.gets2(counts, design))
    qref <- Wrench:::.getReference(counts, ref.est = ref_type)
    tgres <- .getThetag(counts, group, qref)
    s2thetag <- tgres$s2thetag
    thetag <- tgres$thetag
    return(list(
        nzrows = nzrows, qref = qref, s2 = s2, s2thetag = s2thetag,
        thetag = thetag
    ))
}

wrench_cpm_transform <- function(
    X, sample_meta, nzrows, qref, s2, s2thetag, thetag,
    est_type = "w.marg.mean", log = TRUE, prior_count = 1
) {
    suppressPackageStartupMessages(library("edgeR"))
    if (is.data.frame(X)) {
        rnames <- row.names(X)
        cnames <- colnames(X)
    }
    counts <- t(X)
    nzrows <- as.logical(nzrows)
    qref <- as.numeric(qref)
    s2 <- as.numeric(s2)
    s2thetag <- as.numeric(s2thetag)
    thetag <- as.numeric(thetag)
    if (is.null(colnames(counts))) {
        colnames(counts) <- paste0("X", seq_len(ncol(counts)))
    }
    unf_counts <- counts
    unf_ccf <- rep(1, ncol(counts))
    names(unf_ccf) <- colnames(counts)
    counts <- counts[nzrows, ]
    nzcols <- colSums(counts) > 0
    counts <- counts[, nzcols]
    group <- as.character(sample_meta$Class)
    group <- group[nzcols]
    suppressWarnings(W <- wrench(
        counts,
        condition = group, qref = qref, s2 = s2, s2thetag = s2thetag,
        thetag = thetag, etype = est_type
    ))
    unf_ccf[names(W$ccf)] <- W$ccf
    dge <- DGEList(counts = unf_counts, norm.factors = unf_ccf)
    cpms <- cpm(dge, log = log, prior.count = prior_count)
    Xt <- t(cpms)
    if (is.data.frame(X)) {
        Xt <- as.data.frame(Xt)
        row.names(Xt) <- rnames
        colnames(Xt) <- cnames
    }
    return(Xt)
}
