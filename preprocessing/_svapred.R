paramssvaba <- function(X, batch, mod, mod0, algorithm="fast", controls=NULL) {
    if (any(is.na(X)))
        stop("Data contains missing values.")
    if (!is.matrix(X))
        stop("'X' has to be of class 'matrix'.")
    numsv <- sva::num.sv(t(X), mod, method="be", seed=777)
    if (numsv != 0) {
        if (!is.null(controls)) {
          svobj <- sva::sva(
              t(X), mod, mod0, method="supervised", n.sv=numsv,
              controls=controls
          )
        } else {
          svobj <- sva::sva(t(X), mod, mod0, n.sv=numsv)
        }
        # sva doesn't print newline after progress output
        cat("\n")
        xadj <- svabaxadj(t(X), mod, svobj)
        # nmod <- dim(mod)[2]
        # mod <- cbind(mod, svobj$sv)
        # gammahat <- (
        #     t(X) %*% mod %*% solve(t(mod) %*% mod)
        # )[, (nmod + 1):(nmod + numsv)]
        # db = t(X) - gammahat %*% t(svobj$sv)
        # xadj <- t(db)
        params <- list(
            xadj=xadj, xtrain=X, mod=mod, svobj=svobj, algorithm=algorithm
        )
    }
    else {
        warning("Estimated number of factors was zero.")
        params <- list(
            xadj=X, xtrain=X, mod=mod, svobj=NULL, algorithm=algorithm
        )
    }
    params$nbatches <- length(unique(batch))
    params$batch <- batch
    class(params) <- "svatrain"
    return(params)
}

# regress surrogate vars out of exprs to get batch corrected exprs
svabaxadj <- function(exprs, mod, svobj) {
    X <- cbind(mod, svobj$sv)
    Hat <- solve(t(X) %*% X) %*% t(X)
    beta <- (Hat %*% t(exprs))
    P <- ncol(mod)
    exprsadj <- exprs - t(as.matrix(X[, -c(1:P)]) %*% beta[-c(1:P), ])
    return(t(exprsadj))
}

svabaaddon <- function(params, X) {
    if (any(is.na(X)))
        stop("Data contains missing values.")
    if (!is.matrix(X))
        stop("'X' has to be of class 'matrix'.")
    if (class(params) != "svatrain")
        stop("Input parameter 'params' has to be of class 'svatrain'.")
    if (ncol(params$xtrain) != ncol(X))
        stop(
            "Number of variables in test data matrix different to that of ",
            "training data matrix."
        )
    if (!is.null(params$svobj)) {
        fsvaobj <- sva::fsva(
            t(params$xtrain), params$mod, params$svobj, newdat=t(X),
            method=params$algorithm
        )
        xadj <- t(fsvaobj$new)
    }
    else {
        xadj <- X
    }
    return(xadj)
}
