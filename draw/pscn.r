#!/usr/bin/env Rscript
pdf(NULL)

suppressMessages(library(argparse))
suppressMessages(library(tidyverse))
suppressMessages(library(ComplexHeatmap))
suppressMessages(library(viridis))
suppressMessages(library(circlize))
library(RColorBrewer)
library(ggthemes)
ht_opt$message <- F

process_args <- function(args) {
    input <<- normalizePath(args$input)
    outp <- normalizePath(args$output)

    if (is.null(args$locus)) {
        locus <<- basename(input)
    } else {
        locus <<- args$locus
    }
    output <<- sprintf('%s%s%s.', outp, ifelse(dir.exists(outp), '/', ''), locus)

    semirel_threshold <<- args$reliable[1]
    reliable_threshold <<- args$reliable[2]
    no_title <<- args$no_title

    highlight_samples <<- args$samples
}

load <- function(name, comment.char='#', ...) {
  filename <- sprintf('%s/%s%s', input, name, ifelse(grepl('\\.', name), '', '.csv'))
  if (!file.exists(filename)) {
    cat(sprintf('Cannot load csv file "%s"\n', filename))
    stop(1)
  }
  read.csv(filename, sep='\t', comment.char=comment.char, ...)
}

# ------ Color functions and constant color scales ------

colors_to_ramp <- function(colors, min_v=NULL, max_v=NULL, data=NULL) {
    if (is.null(min_v)) {
        min_v <- min(data)
    }
    if (is.null(max_v)) {
        max_v <- max(data)
    }
    n <- length(colors)
    colorRamp2(seq(min_v, max_v, length = n), colors)
}

yellow_colors <- colors_to_ramp(brewer.pal(9, 'YlOrRd'), 0.0, 1.0)
white_colors <- colorRamp2(c(0, 1), c('white', 'white'))

# ------ Argument parser ------

parser <- ArgumentParser(description='Draw paralog-specific copy number.',
      usage='%(prog)s -i <dir> -o <dir>|<prefix> [arguments]')
io_args <- parser$add_argument_group('Input/output files')
io_args$add_argument('-i', '--input', type='character', required=T, metavar='<path>',
    help='Input directory for a specific locus.')
io_args$add_argument('-o', '--output', type='character', required=T, metavar='<path>',
    help='Output directory or prefix.')

opt_args <- parser$add_argument_group('Optional arguments')
opt_args$add_argument('-l', '--locus', type='character', metavar='<name>', required=F,
    help='Locus name [default: basename(input)].')
opt_args$add_argument('-s', '--samples', type='character', nargs='+', metavar='<sample>',
    help=paste('Highlight provided samples. If there are at most 5 samples, ',
               'all samples are highlighted by default.'))
opt_args$add_argument('--no-title', action='store_true',
    help='Do not draw plot titles.')
opt_args$add_argument('--reliable', type='double', metavar='<float>', nargs=2, default=c(0.80, 0.95),
    help='Semi-reliable and Reliable PVS thresholds [default: %(default)s].')
args <- parser$parse_args()
process_args(args)

# ------ Load PSVs ------

ref_frac_matrix <- local({
    use_matrix <- load('extra/em_use_psv_sample') |> column_to_rownames('psv')
    use_matrix <- use_matrix[ , 3:ncol(use_matrix), drop = F] |> as.matrix() == '++'
    use_matrix <- use_matrix[rowSums(use_matrix) > 0, , drop = F]

    psvs_vcf <- suppressMessages(read_delim(file.path(input, 'psvs.vcf.gz'), '\t', comment = '##'))
    names(psvs_vcf)[1:6] <- c('chrom', 'pos', 'ID', 'ref', 'alt', 'qual')
    vcf_samples <- colnames(psvs_vcf)[10:ncol(psvs_vcf)]
    psvs_vcf <- filter(psvs_vcf, grepl('\\bAD\\b', FORMAT)) |>
        mutate(ad_ix = sapply(strsplit(psvs_vcf$FORMAT, ':'), function(x) which(x == 'AD')),
               psv = sprintf('%s:%s', chrom, pos))
    psv_list <- intersect(rownames(use_matrix), psvs_vcf$psv)
    if (length(psv_list) == 0) {
      cat(sprintf('[%s] No PSVs present.\n', locus))
      return(0)
    }

    use_matrix <- use_matrix[psv_list, , drop = F]
    psvs_vcf <- psvs_vcf[match(psv_list, psvs_vcf$psv), , drop = F]

    ref_frac_matrix <- sapply(vcf_samples, function(sample) {
        split_col <- strsplit(psvs_vcf[[sample]], ':', fixed = T)
        ad_col <- sapply(seq_along(split_col), function(i) split_col[[i]][psvs_vcf$ad_ix[i]])
        frac_col <- sapply(strsplit(ad_col, ',', fixed = T), function(x) { x <- as.numeric(x); x[1] / sum(x) })
        ifelse(use_matrix[, sample], frac_col, NA)
    }) |> as.matrix()
    colnames(ref_frac_matrix) <- vcf_samples
    rownames(ref_frac_matrix) <- psvs_vcf$psv
    ref_frac_matrix[, colSums(!is.na(ref_frac_matrix)) > 0, drop = F]
})

# ------ Load other files ------

all_sample_gts <- load('extra/em_sample_gts')
all_likelihoods <- load('extra/em_likelihoods')
all_f_values <- load('extra/em_f_values')

# ------ Draw PSV matrices ------

draw_matrix <- function(sample_gts, sample_gt_probs, ref_fracs, f_matrix,
                        title, out_filename) {
    n_psvs <- dim(ref_fracs)[1]
    if (n_psvs == 0) {
        return()
    }

    # ------ Subset samples ------
    ref_fracs <- ref_fracs[, colSums(!is.na(ref_fracs)) > 0, drop = F]
    samples <- intersect(names(sample_gts), colnames(ref_fracs))
    n_samples <- length(samples)
    if (n_samples == 0) {
        return()
    }

    ref_fracs <- ref_fracs[, samples, drop = F]
    sample_gts <- sample_gts[samples]
    sample_gt_probs <- sample_gt_probs[samples]

    # ------ Cluster samples ------
    sample_clust <- if (n_samples > 1) {
        sample_dist <- dist(t(ref_fracs))
        sample_dist[is.na(sample_dist)] <- 1.1 * max(sample_dist, na.rm = T)
        hclust(sample_dist)
    } else {
      NULL
    }

    # ------ Sample genotype colors ------
    obs_gts <- sort(unique(sample_gts))
    n_gts <- length(obs_gts)
    gt_colors <- if (n_gts > 20) {
        viridis(n_gts, option = 'B')
    } else if (n_gts > 10) {
        tableau_color_pal('Tableau 20')(n_gts)
    } else {
        tableau_color_pal('Tableau 10')(n_gts)
    }
    gt_colors <- structure(gt_colors, names = obs_gts)

    # ------ Rounding ref. fractions ------
    n_copies <- length(strsplit(obs_gts[1], ',', fixed=T)[[1]])
    ref_fracs_round <- apply(ref_fracs * 2 * n_copies, 2,
                            function(x) sprintf('%.0f / %.0f', x, 2 * n_copies)) |>
        matrix(ncol = n_samples)
    if (n_psvs <= 200) {
        rownames(ref_fracs_round) <- rownames(ref_fracs)
    }
    colnames(ref_fracs_round) <- colnames(ref_fracs)
    ref_fracs_round[grepl('^NA', ref_fracs_round, ignore.case = T)] <- NA
    ref_frac_values <- sprintf('%.0f / %.0f', seq(0, 2 * n_copies, 1), 2 * n_copies)
    ref_fracs_colors <- structure(viridis(length(ref_frac_values)), names = ref_frac_values)

    # ------ Bottom annotation (samples) ------
    sample_annot <- HeatmapAnnotation(
        Genotype = sample_gts,
        Weight = sample_gt_probs,
        col = list(Genotype = gt_colors, Weight = yellow_colors),
        which = 'column'
        )

    # ------ Left annotation (PSVs) ------
    min_fval <- apply(f_matrix, 1, min) |> replace_na(0)
    psv_annot <- HeatmapAnnotation(
        Reliable = anno_simple(rep(0, n_psvs),
            pch = ifelse(min_fval >= reliable_threshold, '*', NA), col = white_colors),
        Weight = f_matrix,
        col = list('Weight' = yellow_colors),
        which = 'row',
        show_annotation_name = T
        )

    # ------ Draw the matrix ------
    sample_colors <- ifelse(colnames(ref_fracs) %in% highlight_samples, 'red', 'black')
    {
    scale <- 1.8
    png(out_filename, width=2000 * scale, height=1000 * scale, res=100 * scale)
    h = Heatmap(ref_fracs_round,
        name = 'Allele 1\nfraction',
        col = ref_fracs_colors,
        use_raster = TRUE,
        column_names_gp = grid::gpar(fontsize = 5.5, col = sample_colors),
        row_names_gp = grid::gpar(fontsize = case_when(
            n_psvs > 100 ~ 4,
            n_psvs > 50 ~ 7.5,
            T ~ 9)),
        left_annotation = psv_annot,
        bottom_annotation = sample_annot,
        cluster_columns = sample_clust,
        cluster_rows = F,
        column_title = if (args$no_title) { NULL } else { title }
        )
    draw(h, merge_legend = T)
    ignore <- invisible(dev.off())
    }
}

draw_region_group <- function(group) {
    start_time <- proc.time()['elapsed']
    cat(sprintf('[%s: %s] Start\n', locus, group))

    # ------ Subset data for the current region group ------
    possible_sample_gts <- filter(all_sample_gts, region_group == group)
    likelihoods <- filter(all_likelihoods, region_group == group)
    psv_names <- intersect(filter(all_f_values, region_group == group)$psv,
                        rownames(ref_frac_matrix))
    n_psvs <- length(psv_names)
    if (n_psvs == 0) {
        cat(sprintf('[%s: %s] No PSVs present.\n', locus, group))
        return()
    }
    f_values <- all_f_values[match(psv_names, all_f_values$psv), , drop = F]

    # ------ Select best cluster and load its likelihood ------
    last_iter_lik <- likelihoods |> group_by(cluster) |> slice_tail(n = 1) |> ungroup()
    best_cluster <- last_iter_lik[which.max(last_iter_lik$likelihood),]$cluster[1]
    likelihoods <- filter(likelihoods, cluster == best_cluster)
    last_iteration <- tail(likelihoods, 1)$iteration
    last_likelihood <- tail(likelihoods, 1)$likelihood

    # ------ Extract best sample genotypes and their probabilities ------
    possible_sample_gts <- filter(possible_sample_gts, cluster == best_cluster & iteration == last_iteration)
    possible_sample_gts <- possible_sample_gts[, !is.na(possible_sample_gts[1,])]
    samples <- colnames(possible_sample_gts)[6:ncol(possible_sample_gts)]
    n_samples <- length(samples)
    if (n_samples == 0) {
        cat(sprintf('[%s: %s] No samples present.\n', locus, group))
        return()
    }
    all_gts <- possible_sample_gts$genotype
    sample_gts <- all_gts[apply(possible_sample_gts[samples], 2, which.max)] |> setNames(samples)
    sample_gt_probs <- (10 ^ apply(possible_sample_gts[samples], 2, max)) |> setNames(samples)

    # ------ Reformat f-values matrix ------
    stopifnot(psv_names == f_values$psv)
    f_values <- f_values[, apply(f_values, 2, function(x) sum(is.na(x))) < n_psvs, drop = F]
    rownames(f_values) <- NULL
    f_matrix <- column_to_rownames(f_values, 'psv') |> select(starts_with('copy')) |> as.matrix()
    colnames(f_matrix) <- paste('Copy', 1:ncol(f_matrix))

    # ------ Set title ------
    min_fval <- apply(f_matrix, 1, min) |> replace_na(0)
    title <- sprintf('%s: region group %s. %%s.\nCluster %d,  likelihood %.3f,  %s iterations,  %d/%d reliable PSVs.',
        locus, group, best_cluster, last_likelihood, last_iteration, sum(min_fval >= reliable_threshold), n_psvs)

    # ------ Draw matrices ------
    ref_fracs <- ref_frac_matrix[psv_names, samples, drop = F]

    draw_matrix(sample_gts, sample_gt_probs, ref_fracs, f_matrix,
        title = sprintf(title, 'All PSVs'),
        out_filename = sprintf('%s%s.1_all.png', output, group))

    psv_ixs <- min_fval >= semirel_threshold
    draw_matrix(sample_gts, sample_gt_probs,
                ref_fracs[psv_ixs, , drop = F], f_matrix[psv_ixs, , drop = F],
                title = sprintf(title, 'Semi-reliable PSVs'),
                out_filename = sprintf('%s%s.2_semirel.png', output, group))

    psv_ixs <- min_fval >= reliable_threshold
    draw_matrix(sample_gts, sample_gt_probs,
                ref_fracs[psv_ixs, , drop = F], f_matrix[psv_ixs, , drop = F],
                title = sprintf(title, 'Reliable PSVs'),
                out_filename = sprintf('%s%s.3_reliable.png', output, group))

    cat(sprintf('[%s: %s] Success (%.1f seconds)\n',
                locus, group, proc.time()['elapsed'] - start_time))
}

# ------ Iterate over region groups ------

region_groups <- unique(all_likelihoods$region_group)
for (group in region_groups) {
  tryCatch(draw_region_group(group),
           error = function(e) {
             cat(sprintf('------\n[%s: %s] Could not finish:\n%s\n------\n',
                         locus, group, e))
           })
}
