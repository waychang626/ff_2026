#!/usr/bin/env Rscript
# One-shot setup: install ffanalytics and its dependencies, then verify.
#
# Usage:  Rscript R/setup.R
#
# Only installs what is missing, so re-running after a partial failure is cheap.
# Ends with the smoke test from the project brief - the open item was whether
# ffanalytics installs cleanly in your environment, and this answers it.

cat("=== ffdraft R setup ===\n\n")

# --- 1. R version ------------------------------------------------------------
version_ok <- getRversion() >= "4.1.0"
cat(sprintf("R version: %s %s\n", getRversion(), if (version_ok) "OK" else "TOO OLD"))
if (!version_ok) {
  stop("ffanalytics needs R >= 4.1. Upgrade R, then re-run this script.", call. = FALSE)
}

# --- 2. CRAN dependencies ----------------------------------------------------
deps <- c("remotes", "data.table", "readxl", "httr2", "rvest",
          "purrr", "tidyr", "dplyr", "rrapply", "readr")
missing <- deps[!vapply(deps, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing)) {
  cat(sprintf("\ninstalling %d missing package(s): %s\n",
              length(missing), paste(missing, collapse = ", ")))
  cat("this compiles from source and can take 10-20 minutes\n\n")
  install.packages(missing, repos = "https://cloud.r-project.org")
} else {
  cat("CRAN dependencies: all present\n")
}

still_missing <- deps[!vapply(deps, requireNamespace, logical(1), quietly = TRUE)]
if (length(still_missing)) {
  stop(sprintf("these failed to install: %s\nRe-run the script; if one keeps failing, paste its error.",
               paste(still_missing, collapse = ", ")), call. = FALSE)
}

# --- 3. ffanalytics ----------------------------------------------------------
if (!requireNamespace("ffanalytics", quietly = TRUE)) {
  cat("\ninstalling ffanalytics from GitHub ...\n")
  # A rate-limited GitHub can fail here. If it does, set GITHUB_PAT and re-run:
  #   Sys.setenv(GITHUB_PAT = "ghp_...")
  remotes::install_github("FantasyFootballAnalytics/ffanalytics", upgrade = "never")
} else {
  cat("\nffanalytics: already installed\n")
}

if (!requireNamespace("ffanalytics", quietly = TRUE)) {
  stop("ffanalytics still not installed. Paste the error above.", call. = FALSE)
}
cat(sprintf("ffanalytics version: %s\n", utils::packageVersion("ffanalytics")))

# --- 4. Smoke test -----------------------------------------------------------
# The brief's open item: does a real scrape work in this environment?
cat("\n=== smoke test: scraping RB projections from FantasyPros ===\n")
suppressPackageStartupMessages(library(ffanalytics))

result <- tryCatch(
  scrape_data(src = "FantasyPros", pos = "RB", season = 2026, week = 0),
  error = function(e) e
)

if (inherits(result, "error")) {
  cat(sprintf("\nSMOKE TEST FAILED: %s\n", conditionMessage(result)))
  cat("The package installed but the scrape did not work. Paste this error.\n")
  quit(status = 1)
}

n <- if (!is.null(result$RB)) nrow(result$RB) else 0
cat(sprintf("\nrows returned: %d\n", n))

if (n > 0) {
  cat(sprintf("columns: %s\n",
              paste(utils::head(names(result$RB), 12), collapse = ", ")))
  cat("\nPASS - ffanalytics works. Next: Rscript R/pull_projections.R --season 2026\n")
} else {
  cat("\nFAIL - the scrape returned zero rows.\n")
  cat("Usually means the source changed its page layout, or 2026 projections\n")
  cat("are not published yet at this source. Try another src, e.g. \"CBS\".\n")
  quit(status = 1)
}
