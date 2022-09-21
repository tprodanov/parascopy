#!/usr/bin/env Rscript
pdf(NULL)

suppressMessages(library(argparse))
suppressMessages(library(tidyverse))
suppressMessages(library(ggplot2))

process_args <- function(args) {
  input <<- normalizePath(args$input)
  outp <- normalizePath(args$output)
  highlight_samples <<- args$samples

  if (is.null(args$locus)) {
    locus <<- basename(input)
  } else {
    locus <<- args$locus
  }
  output <<- sprintf('%s%s%s.', outp, ifelse(dir.exists(outp), '/', ''), locus)

  if (!file.exists(file.path(input, 'depth.csv'))) {
    input <<- file.path(input, 'extra')
  }

  n_highlight <<- length(highlight_samples)
  has_highlight <<- n_highlight > 0
}

load <- function(name, comment.char='#', ...) {
  filename <- sprintf('%s/%s%s', input, name, ifelse(grepl('\\.', name), '', '.csv'))
  if (!file.exists(filename)) {
    cat(sprintf('Cannot load csv file "%s"\n', filename))
    stop(1)
  }
  read.csv(filename, sep='\t', comment.char=comment.char, ...)
}

fmt_pos <- function(x) {
  sprintf('%s kb', format(round(x / 1e3), big.mark=',', scientific=F))
}

# ------ Aggregation ------

non_overl_aggr <- function(vec, fun, n) {
  l <- length(vec)
  k <- ceiling(l / n)
  sapply((1:k) * n, function(i) fun(vec[(i - n + 1) : min(i, l)]))
}

overl_aggr <- function(vec, fun, n) {
  zoo::rollapply(vec, width=min(n, length(vec)), FUN=fun, align="right")
}

aggregate.average <- function(df, windows, columns, n=10, rolling=F) {
  if (!('sample' %in% colnames(df))) {
    df$sample <- 'NULL'
  }
  n_windows <- nrow(windows)
  n_samples <- length(unique(df$sample))

  if (n_windows * n_samples != nrow(df)) {
    cat(sprintf('n_windows = %d\n', n_windows))
    cat(sprintf('n_samples = %d\n', n_samples))
    cat(sprintf('n_windows * n_samples = %d\n', n_windows * n_samples))
    cat(sprintf('Dataframe nrow        = %d\n', nrow(df)))
    stop(1)
  }

  const_ends <- cumsum(rle(windows$copy_num)$lengths)
  const_ranges <- data.frame(start=c(0, head(const_ends, -1)) + 1, end=const_ends)

  aggr <- if (rolling) { overl_aggr } else { non_overl_aggr }
  if (nrow(const_ranges) > 1) {
    regions.aggr <- function(values, fun, n) {
      do.call(c, lapply(1:nrow(const_ranges),
          function(i) aggr(values[const_ranges[i, 1]:const_ranges[i, 2]], fun, n)))
    }
  } else {
    regions.aggr <- aggr
  }

  window_starts <- regions.aggr(windows$window_ix, min, n)
  window_ends <- regions.aggr(windows$window_ix, max, n)
  res <- data.frame(aggr_ix=rep(1:length(window_starts), n_samples),
                    start_window=rep(window_starts, n_samples),
                    end_window=rep(window_ends, n_samples))
  res$start <- windows[match(res$start_window, windows$window_ix),]$start
  res$end <- windows[match(res$end_window, windows$window_ix),]$end
  # res$middle <- (res$start + res$end) / 2
  res$middle <- regions.aggr(windows$middle, mean, n)

  for (col in columns) {
    if (grepl(':', col, fixed=T)) {
      tmp <- strsplit(col, ':')[[1]]
      col <- tmp[1]
      fun <- get(tmp[2])
    } else {
      fun <- mean
    }

    if (any(is.na(df[col]))) {
      cat(sprintf('Column "%s" contains NAs\n', col))
      stop(1)
    }
    tmp <- aggregate(as.formula(sprintf('%s ~ sample', col)), df,
                     regions.aggr, fun, n)
    tmp <- cbind(data.frame(sample=tmp$sample), as.data.frame(tmp[[col]]))
    tmp <- pivot_longer(tmp, starts_with('V') | starts_with('tmp'), names_prefix='V',
                        names_to='window', values_to=col)


    if (is.null(res$sample)) {
      res$sample <- tmp$sample
    }
    res <- cbind(res, tmp[col])
  }
  res
}

# ------ Argument parser ------

parser <- ArgumentParser(description='Draw aggregate copy number.',
      usage='%(prog)s -i <dir> -o <dir>|<prefix> [arguments]')
io_args <- parser$add_argument_group('Input/output files')
io_args$add_argument('-i', '--input', type='character', required=T, metavar='<path>',
    help='Input directory for a specific locus.')
io_args$add_argument('-o', '--output', type='character', required=T, metavar='<path>',
    help='Output directory or prefix.')

opt_args <- parser$add_argument_group('Optional arguments')
opt_args$add_argument('-l', '--locus', type='character', metavar='<name>', required=F,
    help='Locus name [default: basename(input)].')
opt_args$add_argument('-s', '--samples', type='character', nargs='*', metavar='<sample>',
    help=paste('Highlight provided samples. If there are at most 5 samples, ',
               'all samples are highlighted by default.'))
opt_args$add_argument('-w', '--window', type='integer', metavar='<int>', default=20,
    help='Aggregate over <int> windows [default: %(default)s].')
opt_args$add_argument('-r', '--rolling', action='store_true',
    help='Use rolling windows to aggregate read depth.')
opt_args$add_argument('--no-title', action='store_true',
    help='Do not draw plot titles.')
opt_args$add_argument('-p', '--positions', type='double', metavar='<int>', nargs=2,
    help='Only draw windows between provided positions.')
opt_args$add_argument('-a', '--alpha', type='double', metavar='<float>', default=0.1,
    help='Line transparency [default: %(default)s].')
args <- parser$parse_args()

process_args(args)

cat(sprintf('[%s] Drawing aggregate copy number to "%s*.png".%s\n', locus, output,
    ifelse(has_highlight, sprintf(' Highlight %d samples.', n_highlight), '')))

# ------ Additional modifications ------

alpha <- args$alpha
color_breaks <- waiver()
colors <- suppressWarnings(RColorBrewer::brewer.pal(Inf, 'Dark2'))
locus_annotation <- list()

if (args$no_title) {
  title <- function(...) geom_blank()
} else {
  title <- ggtitle
}

# ------ Draw raw normalized read depth (all windows) ------

windows <- load('windows.bed', comment.char='') |> rename(chrom = X.chrom)
if ('in_viterbi' %in% colnames(windows)) {
  windows <- rename(windows, in_hmm = in_viterbi)
}

if (is.null(args$positions)) {
  min_window <- 0
  max_window <- max(windows$window_ix)
} else {
  min_window <- min(filter(windows, start >= args$positions[1])$window)
  max_window <- max(filter(windows, end <= args$positions[2])$window)
}
windows <- filter(windows, window_ix >= min_window & window_ix <= max_window)
windows$middle <- (windows$start + windows$end) / 2

depth <- load('depth') |> select(window_ix, sample, depth1, norm_cn1)
depth <- filter(depth, window_ix >= min_window & window_ix <= max_window)
depth <- left_join(depth,
                   select(windows, window_ix, region_ix, region_group, in_hmm),
                   by='window_ix')

MAX_HIGHLIGHTED <- 5
if (!has_highlight && length(unique(depth$sample)) <= MAX_HIGHLIGHTED) {
  highlight_samples <- unique(depth$sample)
  n_highlight <- length(highlight_samples)
  has_highlight <- T
  alpha <- 1
}

x_label <- sprintf('Position (%s)', windows[1,]$chrom)
depth.av <- aggregate.average(depth, windows, 'norm_cn1',
                              n=args$window, rolling=args$rolling)
windows.av <- aggregate.average(windows, windows, 'copy_num',
                              n=args$window, rolling=args$rolling)

curr_geom <- if (nrow(windows.av) > 1) { geom_line } else { geom_point }
ggplot(depth.av) +
  curr_geom(aes(middle, copy_num), color='gray30', size=2, data=windows.av) +
  curr_geom(aes(middle, norm_cn1, group=sample), alpha=alpha, color='blue') +
  { if (n_highlight > 0) {
    list(curr_geom(aes(middle, norm_cn1, color=sample), size=1,
              data=filter(depth.av, sample %in% highlight_samples)),
         scale_color_discrete('Sample'))
    } else { list() } } +
  locus_annotation +
  scale_x_continuous(x_label, labels=fmt_pos, expand=expansion(mult=0.01)) +
  scale_y_continuous('Normalized read depth', breaks=0:40, minor_breaks=NULL,
      expand=expansion(mult=0.01)) +
  title(sprintf('%s: all windows, no multipliers', locus)) +
  theme_minimal()
ggsave(sprintf('%sa_norm_depth_all.png', output), width=8, height=5, bg='white')

# ------ Loading HMM results ------

hmm_states <- load('hmm_states')
hmm_states <- filter(hmm_states, window_ix >= min_window & window_ix <= max_window)
hmm_states$middle <- windows[match(hmm_states$window_ix, windows$window_ix),]$middle
local({
  unique_pos <- unique(hmm_states$middle)
  hmm_states$ix <<- match(hmm_states$middle, unique_pos)
})
hmm_states <- pivot_longer(hmm_states,
    !c('region_group', 'iteration', 'window_ix', 'ix', 'middle'),
    names_to='sample', values_to='pred_cn')
hmm_states_first <- group_by(hmm_states, window_ix, sample) |>
  slice_head(n = 1) |> ungroup()
hmm_states_last <- group_by(hmm_states, window_ix, sample) |>
  slice_tail(n = 1) |> ungroup()

hmm_params <- load('hmm_params')
hmm_params <- group_by(hmm_params, window_ix) |> slice_tail(n = 1) |> ungroup()

# ------ Draw normalized read depth for good windows ------

depth_v <- filter(depth, in_hmm)
depth_v$corr_cn1 <- depth_v$norm_cn1 /
  hmm_params[match(depth_v$window_ix, hmm_params$window_ix),]$multiplier
depth_v <- left_join(depth_v, select(hmm_states_last, 'sample', 'window_ix', 'pred_cn'),
                     by=c('sample', 'window_ix'))
windows_v <- filter(windows, in_hmm)

depth_v.av <- aggregate.average(depth_v, windows_v, c('norm_cn1', 'corr_cn1', 'pred_cn'),
                                n=args$window, rolling=args$rolling)
windows_v.av <- aggregate.average(windows_v, windows_v, 'copy_num',
                                n=args$window, rolling=args$rolling)
if (n_highlight > 0) {
  highl_depth_v.av <- filter(depth_v.av, sample %in% highlight_samples)
}

curr_geom <- if (nrow(windows_v.av) > 1) { geom_line } else { geom_point }
ggplot(depth_v.av) +
  curr_geom(aes(middle, copy_num), color='gray30', size=2, data=windows_v.av) +
  curr_geom(aes(middle, norm_cn1, group=sample), alpha=alpha, color='blue') +
  { if (n_highlight > 0) {
    list(
      curr_geom(aes(middle, norm_cn1, color=sample), size=1, data=highl_depth_v.av),
      scale_color_discrete('Sample'))
  } else { list() } } +
  locus_annotation +
  scale_x_continuous(x_label, labels=fmt_pos, expand=expansion(mult=0.01)) +
  scale_y_continuous('Normalized read depth', breaks=0:40, minor_breaks=NULL,
      expand=expansion(mult=0.01)) +
  title(sprintf('%s: good windows, no multipliers', locus)) +
  theme_minimal()
ggsave(sprintf('%sb_norm_depth_good.png', output), width=8, height=5, bg='white')

ggplot(depth_v.av) +
  curr_geom(aes(middle, copy_num), color='gray30', size=2, data=windows_v.av) +
  curr_geom(aes(middle, corr_cn1, group=sample), alpha=alpha, color='blue') +
  { if (n_highlight > 0) {
    list(
      curr_geom(aes(middle, corr_cn1, color=sample), size=1, data=highl_depth_v.av),
      scale_color_discrete('Sample'))
  } else { list() } } +
  locus_annotation +
  scale_x_continuous(x_label, labels=fmt_pos, expand=expansion(mult=0.01)) +
  scale_y_continuous('Corrected normalized read depth', breaks=0:40, minor_breaks=NULL,
      expand=expansion(mult=0.01)) +
  title(sprintf('%s: good windows, use multipliers', locus)) +
  theme_minimal()
ggsave(sprintf('%sc_corr_depth.png', output), width=8, height=5, bg='white')

n_colors <- length(unique(round(depth_v.av$pred_cn)))
colors <- if (n_colors <= 8) {
  RColorBrewer::brewer.pal(8, 'Dark2')
} else {
  viridis::viridis(n_colors)
}

ggplot(depth_v.av) +
  curr_geom(aes(middle, copy_num), color='gray30', size=2, data=windows_v.av) +
  curr_geom(aes(middle, corr_cn1, group=sample,
                color=factor(round(pred_cn))), alpha=min(1, alpha * 2)) +
  locus_annotation +
  scale_x_continuous(x_label, labels=fmt_pos, expand=expansion(mult=0.01)) +
  scale_y_continuous('Normalized read depth', breaks=0:40, minor_breaks=NULL,
      expand=expansion(mult=0.01)) +
  scale_color_manual('CN estimate',
                     values=colors, breaks=color_breaks) +
  title(sprintf('%s: good windows, use multipliers', locus)) +
  guides(color = guide_legend(override.aes = list(alpha=1, size=2.5))) +
  theme_minimal()
ggsave(sprintf('%sd_corr_depth_color.png', output), width=8, height=5, bg='white')

# ------ Draw HMM paths ------

get_text_pos <- function(hmm_states_last) {
  max_ix <- max(hmm_states$ix)
  transitions <- unlist(aggregate(pred_cn ~ sample,
        hmm_states_last, function(x) which(diff(x) != 0))$pred_cn)
  n_samples <- length(unique(hmm_states$sample))
  transitions <- table(transitions)
  transitions <- as.numeric(names(transitions)[transitions >= 0.1 * n_samples])
  common_transitions <<- transitions
  transitions <- c(1, transitions, max_ix)
  transitions <- pmin(pmax(unique(round(transitions / 5) * 5), 1), max_ix)
  unique(pmin(pmax((head(transitions, -1) + tail(transitions, -1)) %/% 2, 1), max_ix))
}
text_pos <- get_text_pos(hmm_states_last)

if (any(hmm_states_first$iteration != 'v')) {
  hmm_states_first$pred_cn_round <- round(hmm_states_first$pred_cn)
  sample_counts <- aggregate(sample ~ middle + ix + pred_cn_round,
    filter(hmm_states_first, ix %in% text_pos), length)
  ggplot(hmm_states_first) +
    locus_annotation +
    geom_line(aes(middle, copy_num), data=windows_v, size=4, color='gray70') +
    geom_line(aes(middle, pred_cn, group=sample), color='blue', alpha=alpha) +
    { if (n_highlight > 0) {
      list(
        geom_line(aes(middle, pred_cn, color=sample), size=1,
                data=filter(hmm_states_first, sample %in% highlight_samples)),
        scale_color_discrete('Sample'))
    } else { list() } } +
    geom_text(aes(middle, y = pred_cn_round + 0.3, label = sample),
              data=sample_counts, size=3) +
    scale_x_continuous(x_label, labels=fmt_pos, expand=expansion(mult=0.01)) +
    scale_y_continuous('Copy number estimate', breaks=0:40, minor_breaks=NULL,
        expand=expansion(mult=0.01)) +
    theme_minimal()
  ggsave(sprintf('%se_hmm_first.png', output), width=8, height=5, bg='white')
}

sample_counts <- aggregate(sample ~ middle + ix + pred_cn,
    filter(hmm_states_last, ix %in% text_pos), length)
ggplot(hmm_states_last) +
  locus_annotation +
  geom_line(aes(middle, copy_num), data=windows_v, size=4, color='gray70') +
  geom_line(aes(middle, pred_cn, group=sample), color='blue', alpha=alpha) +
  { if (n_highlight > 0) {
    list(geom_line(aes(middle, pred_cn, color=sample), size=1,
              data=filter(hmm_states_last, sample %in% highlight_samples)),
         scale_color_discrete('Sample'))
  } else { list() } } +
  geom_text(aes(middle, y = pred_cn + 0.3, label = sample), data=sample_counts, size=3) +
  scale_x_continuous(x_label, labels=fmt_pos, expand=expansion(mult=0.01)) +
  scale_y_continuous('Copy number estimate', breaks=0:40, minor_breaks=NULL,
      expand=expansion(mult=0.01)) +
  theme_minimal()
ggsave(sprintf('%sf_hmm_last.png', output), width=8, height=5, bg='white')

if (0 < n_highlight && n_highlight < 40) {
  highl_depth_v <- aggregate.average(
      filter(depth_v, sample %in% highlight_samples), windows_v,
      c('corr_cn1', 'pred_cn'), n=1)
  highl_depth_v2 <- aggregate.average(
      filter(depth, sample %in% highlight_samples & !in_hmm),
      filter(windows, !in_hmm),
      'norm_cn1', n=1)

  transition_lines <- windows_v[common_transitions,]$middle
  # if (length(transition_lines) > 20) {
  #   transition_lines <- c()
  # }
  ggplot(highl_depth_v) +
    geom_vline(xintercept=transition_lines, color='gray80', linetype='dashed') +
    geom_line(aes(middle, copy_num), data=windows_v, size=2, color='gray80') +
    geom_point(aes(middle, corr_cn1, group=sample,
                  color = factor(pred_cn))) +
    geom_point(aes(middle, norm_cn1, group=sample), data = highl_depth_v2,
        shape = 21) +
    facet_wrap(~ sample, ncol=3, scales='free_y') +
    scale_x_continuous(x_label, labels=fmt_pos, expand=expansion(mult=0.01)) +
    scale_y_continuous('Normalized read depth', breaks=0:40, minor_breaks=NULL,
        expand=expansion(mult=0.01)) +
    scale_color_manual('CN estimate',
                       values=colors, breaks=color_breaks) +
    guides(color = guide_legend(override.aes = list(alpha=1, size=2))) +
    theme_minimal() +
    theme(panel.grid.major.y=element_line(color='gray80'))

  width <- 5 + nrow(windows_v) / 100 * 1.5 * min(n_highlight, 3)
  height <- max(5, ceiling(n_highlight / 3) * 2)
  ggsave(sprintf('%sg_highlight_samples.png', output),
         width=width, height=height, limitsize=F, bg='white')
}
