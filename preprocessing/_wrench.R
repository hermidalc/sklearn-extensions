# Adapted from https://github.com/HCBravoLab/Wrench/blob/master/R/wrenchSource.R
# ML-compatible Wrench normalization for sparse, under-sampled, zero-inflated
# count data

.getHurdleFit <- function(mat, pres.abs.mod = TRUE, ...) {
    tau <- colSums(mat)
    design <- model.matrix(~ -1 + log(tau))
    if (pres.abs.mod) {
        pi0.fit <- apply(mat, 1, function(x) {
            fastglm(design, c(1 * (x == 0)), family = binomial())
        })
    } else {
        pi0.fit <- apply(mat, 1, function(x) {
            fastglm(design, cbind(tau - x, x), family = binomial())
        })
    }
    pi0.fit
}

.getHurdle <- function(
    mat, pi0.fit, pres.abs.mod = TRUE, thresh = FALSE, thresh.val = 1e-8, ...
) {
    n <- ncol(mat)
    tau <- colSums(mat)
    newdata <- as.matrix(data.frame(LogTau = log(tau)))
    if (pres.abs.mod) {
        pi0 <- t(vapply(
            pi0.fit, function(x) {
                predict(x, newdata = newdata, type = "response")
            },
            FUN.VALUE = numeric(n)
        ))
    } else {
        pi0 <- t(vapply(
            pi0.fit, function(x) {
                exp(tau * log(predict(x, newdata = newdata, type = "response")))
            },
            FUN.VALUE = numeric(n)
        ))
    }
    if (thresh) {
        pi0[pi0 > 1 - thresh.val] <- 1 - thresh.val
        pi0[pi0 < thresh.val] <- thresh.val
    }
    pi0
}

.gets2 <- function(
    mat, design = model.matrix(mat[1, ] ~ 1), ebs2 = TRUE, smoothed = FALSE, ...
) {
    p <- nrow(mat)
    nzrows <- rowSums(mat) > 0
    mat <- mat[nzrows, ]

    tjs <- colSums(mat)
    tjs <- tjs / exp(mean(log(tjs)))

    design <- cbind(design, log(tjs))
    mat <- log(mat)
    mat[!is.finite(mat)] <- NA
    fit <- lmFit(mat, design)
    s <- fit$sigma
    s[s == 0] <- NA

    mu <- rowMeans(mat, na.rm = TRUE)
    sqrt.s <- sqrt(s)

    k <- tryCatch(
        {
            l <- locfit(sqrt.s ~ mu, family = "gamma")
            predict(l, mu)^4
        },
        error = function(e) {
            NULL
        }
    )

    if (smoothed && !is.null(k)) {
        s2 <- k
    } else {
        s2 <- s^2
        if (!is.null(k)) {
            s2[is.na(s2)] <- k[is.na(s2)]
        } else {
            s2[is.na(s2)] <- 0
        }
        if (ebs2) {
            s2 <- limma::squeezeVar(
                s2,
                df = max(c(1, fit$df.residual), na.rm = TRUE)
            )$var.post
        }
    }
    s2.tmp <- c(matrix(NA, nrow = p, ncol = 1))
    s2.tmp[nzrows] <- s2
    s2.tmp
}

.getReference <- function(mat, ref.est = "sw.means", ...) {
    tau <- colSums(mat)
    if (ref.est == "logistic") {
        design <- model.matrix(~ 1, data=data.frame(tau))
        qref <- 1 - plogis(
            apply(mat, 1, function(x) {
                fastglm(
                    design, cbind(tau - x, x),
                    family = binomial()
                )$coefficients
            })
        )
    } else if (ref.est == "sw.means") {
        qmat <- sweep(mat, 2, tau, "/")
        qref <- rowMeans(qmat)
    } else {
        stop("Unknown reference type.")
    }
    qref
}

wrench <- function(
    mat, condition, nzrows = NULL, qref = NULL, s2 = NULL, s2thetag = NULL,
    thetag = NULL, pi0.fit = NULL, etype = "w.marg.mean", ref.est = "sw.means",
    ebcf = TRUE, z.adj = FALSE, phi.adj = TRUE, detrend = FALSE, ...
) {
    suppressPackageStartupMessages({
        library(fastglm)
        library(limma)
        library(locfit)
        library(stats)
        library(matrixStats)
    })
    # trim
    if (is.null(nzrows)) {
        nzrows <- rowSums(mat) > 0
    }
    mat <- mat[nzrows, ]
    nzcols <- colSums(mat) > 0
    mat <- mat[, nzcols]
    condition <- condition[nzcols]

    # stopifnot(all(rowSums(mat) > 0))
    stopifnot(all(colSums(mat) > 0))
    stopifnot(ncol(mat) == length(condition))
    if (!is.null(qref)) stopifnot(nrow(mat) == length(qref))
    if (!is.null(s2)) stopifnot(nrow(mat) == length(s2))

    n <- ncol(mat)
    p <- nrow(mat)
    tau <- colSums(mat)

    # feature-wise parameters: hurdle, variance, reference and raw ratios
    if (!((etype %in% c("mean", "median", "s2.w.mean")) && !z.adj)) {
        if (is.null(pi0.fit)) {
            suppressWarnings(pi0.fit <- .getHurdleFit(mat, ...))
        }
        pi0 <- .getHurdle(mat, pi0.fit, ...)
    } else {
        pi0.fit <- NULL
        pi0 <- NULL
    }

    group <- as.character(condition)
    if (length(unique(group)) == 1) {
        design <- model.matrix(mat[1, ] ~ 1)
    } else {
        design <- model.matrix(~ -1 + group)
    }

    # variances
    if (is.null(s2)) {
        s2 <- .gets2(mat, design, ...)
    }

    # reference
    if (is.null(qref)) {
        qref <- .getReference(mat, ref.est = ref.est, ...)
    }

    # sample-wise ratios
    qmat <- sweep(mat, 2, tau, "/")
    r <- qmat / qref

    if (ebcf) {
        if (is.null(s2thetag) && is.null(thetag)) {
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
            # weighted estimator
            qg <- sweep(Yg, 2, colSums(Yg), "/")
            rg <- qg / qref
            lrg <- log(rg)
            lrg[!is.finite(lrg)] <- NA
            s2thetag <- colVars(lrg, na.rm = TRUE)
            thetag <- colMeans(rg)
        }
        s2thetag_rep <- design %*% s2thetag
        thetag_rep <- c(design %*% thetag)

        # regularized estimation of positive means.
        r <- sweep(r, 2, thetag_rep, "/")
        thetagj <- exp(vapply(seq(n), function(j) {
            x <- log(r[, j])
            x[!is.finite(x)] <- NA
            weighted.mean(
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

    # adjustments for marginal and truncated means
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
            "nzrows" = nzrows,
            "pi0.fit" = pi0.fit,
            "pi0" = pi0,
            "qref" = qref,
            "design" = design,
            "s2" = s2,
            "r" = r,
            "radj" = radj
        )
    )

    res$ccf <- Wrench:::.estimSummary(
        res,
        estim.type = etype, z.adj = z.adj, ...
    )
    res$ccf <- with(res, ccf / exp(mean(log(ccf))))

    if (detrend) {
        res$others$ccf0 <- res$ccf
        detrended <- Wrench:::.detrend.ccf(res$ccf, tau, condition)
        res$others$ccf.detr.un <- detrended$ccf.detr.un
        res$ccf <- detrended$ccf.detr
    }

    tjs <- colSums(mat) / exp(mean(log(colSums(mat))))
    names(res$ccf) <- names(tjs)
    res$nf <- res$ccf * tjs

    res
}
