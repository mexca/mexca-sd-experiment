#-----------------------------------------------------------------------------------#
#----- results of the speech-brain and pyannote speaker diarization pipelines ------#
#-----------------------------------------------------------------------------------#
#### import libraries ####

library(ggpubr)
library(tidyverse)
library(ggstatsplot)
library(egg)

#### set paths ####
# set local directory
localdir <- "/Users/evaviviani/Library/CloudStorage/OneDrive-NetherlandseScienceCenter/mexca/eva_analysis/"
# define output folder
outputFolder <- "/Users/evaviviani/Library/CloudStorage/OneDrive-NetherlandseScienceCenter/mexca/eva_analysis/output/"

#### load results of the speaker diarization pipelines ####
read.csv(paste0(outputFolder,"results_pipelines.csv"), stringsAsFactors = T)-> sb_results
read.csv(paste0(outputFolder,"pyannote_results.csv"), stringsAsFactors = T)-> pa_results

#### speaker diarization pipelines results plot ####
# bind datasets together
pa_results$cluster <- as.factor('none') #we don't have different clusters methods for pyannote
final_results <- bind_rows(sb_results, pa_results)

final_results = final_results%>%
  # we select only one cluster methods for the other pipelines as they were not leading different results 
  # this gives us the same number of datapoints across pipelines (i.e., n=16)
  filter(cluster %in% c("ac","none"))%>%droplevels() 

# build the plot
p1<-ggbetweenstats(data = final_results%>%filter(metric == "der"),
                   x = method,
                   y = value,
                   # color = cluster,
                   plot.type = "boxviolin",
                   results.subtitle = FALSE,
                   pairwise.comparisons = FALSE,
                   title = "Diarization Error Rate w/ overlapping segments",
                   ggplot.component =
                     list(ggplot2::scale_y_continuous(
                       breaks = seq(0, 1, .2),
                       limits = (c(0, 1)))))
p1
p2<-ggbetweenstats(data = final_results%>%filter(metric == "der_skip"),
                   x = method,
                   y = value,
                   color = cluster,
                   plot.type = "boxviolin",
                   results.subtitle = FALSE,
                   pairwise.comparisons = FALSE,
                   title = "Diarization Error Rate w/o overlapping segments",
                   ggplot.component =
                     list(ggplot2::scale_y_continuous(
                       breaks = seq(0, 1, .2),
                       limits = (c(0, 1)))))
p2
p3<-ggbetweenstats(data = final_results%>%filter(metric == "coverage"),
                   x = method,
                   y = value,
                   color = cluster,
                   plot.type = "boxviolin",
                   results.subtitle = FALSE,
                   pairwise.comparisons = FALSE,
                   title = "Coverage",
                   ggplot.component =
                     list(ggplot2::scale_y_continuous(
                       breaks = seq(0, 1, .2),
                       limits = (c(0, 1)))))

p4<-ggbetweenstats(data = final_results%>%filter(metric == "purity"),
                   x = method,
                   y = value,
                   color = cluster,
                   plot.type = "boxviolin",
                   results.subtitle = FALSE,
                   pairwise.comparisons = FALSE,
                   title = "Purity",
                   ggplot.component =
                     list(ggplot2::scale_y_continuous(
                       breaks = seq(0, 1, .2),
                       limits = (c(0, 1)))))

# combine them
combine_plots(
  list(p1, p2, p3, p4),
  plotgrid.args = list(nrow = 2),
  annotation.args = list(
    title = "Comparison of performance metrics across methods",
    caption = "Dataset: AMI Meeting Corpus"
  )
)
#save plot
ggsave(paste0(localdir,"compare_sd_pipelines.png"), width = 9)
