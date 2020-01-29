# Batch effect correction transformer functions

source(paste(dirname(sys.frame(1)$ofile), "_stICA.R", sep="/"))
source(paste(dirname(sys.frame(1)$ofile), "_svapred.R", sep="/"))

# adapted from limma::removeBatchEffect source code
limma_removeba_fit <- function(X, sample_meta, preserve_design=TRUE) {
    suppressPackageStartupMessages(library("limma"))
    batch <- factor(sample_meta$Batch)
    if (length(levels(batch)) > 1) {
        contrasts <- contr.sum(levels(batch))
        contrasts(batch) <- contrasts
        batch_design <- model.matrix(~batch)[, -1, drop=FALSE]
        if (preserve_design) {
            sample_meta$Class <- factor(sample_meta$Class)
            design <- model.matrix(~Class, data=sample_meta)
        } else {
            design <- matrix(1, nrow=nrow(X), ncol=1)
        }
        fit <- lmFit(t(X), cbind(design, batch_design))
        beta <- fit$coefficients[, -seq_len(ncol(design)), drop=FALSE]
        beta[is.na(beta)] <- 0
        batch_adj <- beta %*% t(contrasts)
    } else {
        batch_adj <- matrix(0, nrow=ncol(X), ncol=1)
        colnames(batch_adj) <- levels(batch)
    }
    return(as.data.frame(batch_adj))
}

limma_removeba_transform <- function(X, sample_meta, batch_adj) {
    batch <- as.character(sample_meta$Batch)
    batch_adj[, setdiff(batch, colnames(batch_adj))] <- 0
    return(t(t(X) - as.matrix(batch_adj)[, batch]))
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
    y <- factor(sample_meta$Class + 1)
    batch <- as.character(sample_meta$Batch)
    unique_batch <- sort(unique(batch))
    for (n in seq_len(length(unique_batch))) {
        if (n != unique_batch[n]) {
            batch <- replace(batch, batch == unique_batch[n], n)
        }
    }
    batch <- factor(batch)
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
        sample_meta$Class <- factor(sample_meta$Class)
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
    batch <- as.character(sample_meta$Batch)
    unique_batch <- sort(unique(batch))
    for (n in seq_len(length(unique_batch))) {
        if (n != unique_batch[n]) {
            batch <- replace(batch, batch == unique_batch[n], n)
        }
    }
    batch <- factor(batch)
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
