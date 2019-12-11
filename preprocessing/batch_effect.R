# Batch effect correction transformer functions

source(paste(dirname(sys.frame(1)$ofile), "stICA.R", sep="/"))
source(paste(dirname(sys.frame(1)$ofile), "svapred.R", sep="/"))

# adapted from limma::removeBatchEffect source code
limma_removeba_fit <- function(X, sample_meta, preserve_design=TRUE) {
    suppressPackageStartupMessages(library("limma"))
    batch <- sample_meta$Batch
    sample_meta$Batch <- NULL
    if (preserve_design) {
        sample_meta$Class <- as.factor(sample_meta$Class)
        design <- model.matrix(~Class, data=sample_meta)
    } else {
        design <- matrix(1, ncol(t(X)), 1)
    }
    batch <- as.factor(batch)
    contrasts(batch) <- contr.sum(levels(batch))
    batch <- model.matrix(~batch)[, -1, drop=FALSE]
    fit <- lmFit(t(X), cbind(design, batch))
    beta <- fit$coefficients[, -seq_len(ncol(design)), drop=FALSE]
    beta[is.na(beta)] <- 0
    return(beta)
}

limma_removeba_transform <- function(X, sample_meta, beta) {
    batch <- sample_meta$Batch
    batch <- as.factor(batch)
    contrasts(batch) <- contr.sum(levels(batch))
    batch <- model.matrix(~batch)[, -1, drop=FALSE]
    return(t(t(X) - beta %*% t(batch)))
}

stica_removeba_fit <- function(
    X, sample_meta, method=c("stICA", "SVD"), k=20, alpha=0.5
) {
    if (method == "stICA") {
        params <- normFact(
            method, t(X), sample_meta$Batch, "categorical", k=k, alpha=alpha,
            ref2=sample_meta$Class, refType2="categorical"
        )
    } else {
        params <- normFact(
            method, t(X), sample_meta$Batch, "categorical", k=k,
            ref2=sample_meta$Class, refType2="categorical"
        )
    }
    return(list(t(params$Xn), params[names(params) != "Xn"]))
}

stica_removeba_transform <- function(X, params) {
    # Renard et al stICA IEEE 2017 paper code add-on batch effect correction
    # Vte = dot(dot(Xte.T,U),np.linalg.inv(dot(U.T,U)))
    # Xte_n = dot(U,Vte.T)
    return(
        t(params$U %*% t(X %*% params$U %*% solve(t(params$U) %*% params$U)))
    )
}

bapred_removeba_fit <- function(
    X, sample_meta, method=c(
        "combat", "meancenter", "fabatch", "qunorm", "ratioa", "ratiog",
        "standardize", "sva"
    )
) {
    suppressPackageStartupMessages(library("bapred"))
    y <- as.factor(sample_meta$Class + 1)
    batch <- sample_meta$Batch
    unique_batch <- sort(unique(batch))
    for (j in seq_len(unique_batch)) {
        if (j != unique_batch[j]) {
            batch <- replace(batch, batch == unique_batch[j], j)
        }
    }
    batch <- as.factor(batch)
    if (method == "combat") {
        params <- combatba(X, batch)
    }
    else if (method == "meancenter") {
        params <- meancenter(X, batch)
    }
    else if (method == "fabatch") {
        params <- fabatch(X, y, batch)
    }
    else if (method == "qunorm") {
        params <- qunormtrain(X)
    }
    else if (method == "ratioa") {
        params <- ratioa(X, batch)
    }
    else if (method == "ratiog") {
        params <- ratiog(X, batch)
    }
    else if (method == "standardize") {
        params <- standardize(X, batch)
    }
    else if (method == "sva") {
        sample_meta$Class <- as.factor(sample_meta$Class)
        mod <- model.matrix(~Class, data=sample_meta)
        mod0 <- model.matrix(~1, data=sample_meta)
        # ctrls <- as.numeric(grepl("^AFFX", rownames(t(X))))
        params <- svaba(X, batch, mod, mod0, algorithm="fast")
    }
    if (method == "qunorm") {
        xadj_key <- "xnorm"
    } else {
        xadj_key <- "xadj"
    }
    return(list(params[[xadj_key]], params[names(params) != xadj_key]))
}

bapred_removeba_transform <- function(
    X, sample_meta, method=c(
        "combat", "meancenter", "fabatch", "qunorm", "ratioa", "ratiog",
        "standardize", "sva"
    ), params
) {
    suppressPackageStartupMessages(library("bapred"))
    batch <- sample_meta$Batch
    unique_batch <- sort(unique(batch))
    for (j in seq_len(unique_batch)) {
        if (j != unique_batch[j]) {
            batch <- replace(batch, batch == unique_batch[j], j)
        }
    }
    batch <- as.factor(batch)
    if (method == "combat") {
        return(combatbaaddon(params, X, batch))
    }
    else if (method == "meancenter") {
        return(meancenteraddon(params, X, batch))
    }
    else if (method == "fabatch") {
        return(fabatchaddon(params, X, batch))
    }
    else if (method == "qunorm") {
        return(qunormaddon(params, X))
    }
    else if (method == "ratioa") {
        return(ratioaaddon(params, X, batch))
    }
    else if (method == "ratiog") {
        return(ratiogaddon(params, X, batch))
    }
    else if (method == "standardize") {
        return(standardizeaddon(params, X, batch))
    }
    else if (method == "sva") {
        return(svabaaddon(params, X))
    }
}
