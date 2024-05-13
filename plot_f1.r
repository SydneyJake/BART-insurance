library(reshape2)
library(tidyverse)

theme_set(theme_dark(base_size = 15))

# Read the F1 scores into a dataframe
read_F1 <- function(scores){
  df <- melt(scores, value.name = "F1", varnames = c("ntree", "nfeature", "thresh"))
  df$F1[which(is.na(df$F1))] <- 0
  df$ntree  <- factor(df$ntree, labels = paste(c("ntree"), seq(10, 100, by = 10)))
  df$thresh <- factor(df$thresh, labels = as.character(seq(0.05, 0.5, by=0.05)))
  return(df)
}

# Plot the F1 scores
plot_F1 <-function(df){
  p <- ggplot(data=df, aes(x=nfeature, y=F1, group=thresh, col=thresh, fill=thresh)) +
          geom_point(alpha=0.33, shape=21, col="black", stroke=0.2) +
          geom_smooth(se = F, method = "lm", alpha=0.33, formula = "y~poly(x,2)") +
          facet_wrap(facets = vars(ntree), ncol = 5) +
          scale_color_brewer(palette = "Set3") +
          scale_fill_brewer(palette = "Set3") +
          coord_cartesian(ylim = c(0, 0.25)) +
          ylab("F1 score")
  return(p)
}

df_xgb  <- read_F1(readRDS("XGB_F1_scores_cv.rds"))
df_bart <- read_F1(readRDS("bart_F1_scores.rds"))

F1_xgb  <- plot_F1(df_xgb)
ggsave(plot = F1_xgb, filename = "xgb_f1.png", units = "in", dpi = 300, width=10)

F1_bart <- plot_F1(df_bart)
ggsave(plot = F1_bart, filename = "bart_f1.png", units = "in", dpi = 300, width=10)
