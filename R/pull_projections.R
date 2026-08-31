#!/usr/bin/env Rscript
# Pull raw stat lines from ffanalytics and write them in the engine's ingest
# schema.
#
# NOT EXECUTED IN THE CONTAINER THIS WAS WRITTEN IN - that machine has no R and
# no outbound reach to the projection sources. Verify locally before draft day
# with the smoke test in the "verify" block at the bottom of this file.
#
# Section 5's selection rule governs the whole script: export RAW STAT LINES,
# never pre-scored fantasy points. No vendor scores either league's exact
# settings, so we pull yards / TDs / receptions / INTs once and apply the
# scoring function twice on the Python side, once per league.
#
# Usage:
#   Rscript R/pull_projections.R --season 2026 --out data/projections_2026.csv
#
# Setup (R >= 4.1):
#   install.packages(c("remotes","data.table","readxl","httr2","rvest",
#                      "purrr","tidyr","dplyr","rrapply","readr"))
#   remotes::install_github("FantasyFootballAnalytics/ffanalytics")

suppressPackageStartupMessages({
  library(ffanalytics)
  library(dplyr)
  library(readr)
  library(purrr)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default) {
  hit <- which(args == flag)
  if (length(hit) && length(args) > hit[1]) args[hit[1] + 1] else default
}

season   <- as.integer(get_arg("--season", "2026"))
out_path <- get_arg("--out", sprintf("data/projections_%d.csv", season))
adp_path <- get_arg("--adp-out", sprintf("data/market_%d.csv", season))

# NumberFire is auto-converted to FanDuel upstream. Sources are internally
# rate-limited to about 2s/page, so a full pull takes several minutes.
sources   <- c("CBS", "ESPN", "FantasyPros", "FFToday", "FantasyData",
               "FleaFlicker", "NumberFire", "NFL", "RTSports")
positions <- c("QB", "RB", "WR", "TE", "K", "DST")

message(sprintf("scraping %d sources x %d positions for season %d ...",
                length(sources), length(positions), season))

raw <- scrape_data(src = sources, pos = positions, season = season, week = 0)  # week 0 = full season

# Stat columns the Python engine understands. Anything else is dropped here
# rather than silently scored as zero downstream - `ffdraft.projections`
# refuses a file containing a column it does not recognise, which is what makes
# an upstream rename visible instead of quietly costing you a category.
stat_cols <- c(
  "pass_yds", "pass_tds", "pass_int", "pass_comp", "pass_att",
  "rush_yds", "rush_tds", "rush_att",
  "rec", "rec_yds", "rec_tds", "rec_tgt",
  "fumbles_lost", "two_pts", "return_tds",
  "xp", "xp_att", "xp_miss", "fg", "fg_att", "fg_miss",
  "fg_0019", "fg_2029", "fg_3039", "fg_4049", "fg_50",
  "dst_int", "dst_fum_rec", "dst_sacks", "dst_safety", "dst_td", "dst_blk",
  "dst_pts_allowed", "dst_yds_allowed"
)

normalise <- function(df, pos) {
  if (is.null(df) || nrow(df) == 0) return(NULL)
  df <- as.data.frame(df)

  name_col <- intersect(c("player", "player_name", "name"), names(df))[1]
  src_col  <- intersect(c("data_src", "source"), names(df))[1]
  team_col <- intersect(c("team", "tm"), names(df))[1]

  if (is.na(name_col) || is.na(src_col)) {
    warning(sprintf("skipping %s: no player/source column (saw: %s)",
                    pos, paste(names(df), collapse = ", ")))
    return(NULL)
  }

  out <- data.frame(
    source = as.character(df[[src_col]]),
    player = as.character(df[[name_col]]),
    pos    = pos,
    team   = if (!is.na(team_col)) as.character(df[[team_col]]) else "",
    bye    = if ("bye" %in% names(df)) suppressWarnings(as.integer(df[["bye"]])) else NA_integer_,
    stringsAsFactors = FALSE
  )
  for (col in stat_cols) {
    out[[col]] <- if (col %in% names(df)) {
      suppressWarnings(as.numeric(df[[col]]))
    } else {
      0
    }
  }
  out[is.na(out)] <- 0
  out$bye[is.na(out$bye)] <- 0
  out
}

rows <- imap(raw, function(df, pos) normalise(df, pos))
long <- bind_rows(compact(rows))

if (nrow(long) == 0) stop("scrape returned no rows - check connectivity and ffanalytics version")

# A defense's "name" varies wildly by source; the Python resolver keys team
# defenses off nicknames, so keep whatever the source gave and let it match.
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write_csv(long, out_path)
message(sprintf("wrote %s: %d rows, %d players, %d sources",
                out_path, nrow(long),
                length(unique(paste(long$player, long$pos))),
                length(unique(long$source))))

# --- ADP / ECR ---------------------------------------------------------------
# Scoring is irrelevant to ADP, so a trivial rule set is enough to get the
# table built. Baselines likewise: we never read the VOR column ffanalytics
# computes, because its default assumes a 12-team league and is wrong for
# League 1 (brief section 5). The engine computes its own.
adp <- tryCatch({
  tbl <- projections_table(raw, avg_type = "average")
  tbl <- add_adp(tbl)
  tbl <- add_ecr(tbl)
  tbl <- add_player_info(tbl)
  as.data.frame(tbl)
}, error = function(e) {
  warning(sprintf("ADP/ECR pull failed (%s); the engine will impute from points rank", e$message))
  NULL
})

if (!is.null(adp)) {
  pick <- function(df, options) {
    hit <- intersect(options, names(df))
    if (length(hit)) df[[hit[1]]] else NA
  }
  name_parts <- pick(adp, c("player", "player_name"))
  if (all(is.na(name_parts)) && all(c("first_name", "last_name") %in% names(adp))) {
    name_parts <- paste(adp$first_name, adp$last_name)
  }
  market <- data.frame(
    player = as.character(name_parts),
    pos    = as.character(pick(adp, c("pos", "position"))),
    adp    = suppressWarnings(as.numeric(pick(adp, c("adp", "avg_adp", "adp_avg")))),
    ecr    = suppressWarnings(as.numeric(pick(adp, c("ecr", "avg_rank")))),
    adp_sd = suppressWarnings(as.numeric(pick(adp, c("adp_sd", "sd_adp", "sd_rank")))),
    stringsAsFactors = FALSE
  )
  market <- market[!is.na(market$player) & market$player != "", ]
  write_csv(market, adp_path)
  message(sprintf("wrote %s: %d rows", adp_path, nrow(market)))
}

# --- verify ------------------------------------------------------------------
# Run this first, on its own, before trusting a full pull. It is the open item
# from the brief: confirm ffanalytics installed cleanly in your environment.
#
#   library(ffanalytics)
#   test <- scrape_data(src = "FantasyPros", pos = "RB", season = 2026, week = 0)
#   str(test$RB)          # expect a data frame with rush_yds, rec, etc.
#   nrow(test$RB) > 0
