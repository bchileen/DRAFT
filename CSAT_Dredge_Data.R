library(readr)
library(dplyr)
library(stringr)
library(purrr)
library(ggplot2)
library(lubridate)
library(tidyverse)
library(zoo)
options(scipen = 100, digits = 2)
dredge_data<-read_csv("DredgeData.csv", show_col_types = FALSE)
# Set folder path
folder_path <- "C:/workspace/CSAT/CSAT_distribution/output/CEMVR/19990101__to__20241231"

reach_data<-read_csv("C:/workspace/DRAFT/ReachData.csv")

# List matching files (ends with dredged_date_vol.csv)
file_list <- list.files(path = folder_path,
                        pattern = "_dredged_date_vol\\.csv$",
                        full.names = TRUE)

# Function to extract metadata and read file
read_and_annotate <- function(file_path) {
  # Extract file name only
  file_name <- basename(file_path)
  
  # Use regex to extract River, Pool, and Reach
  # Pattern: CEMVR_(River)_(Pool)_MVR_(Reach)_dredged_date_vol
  matches <- str_match(file_name, "^CEMVR_([A-Z]{2})_([A-Z0-9]+)_MVR_([0-9]+)_dredged_date_vol\\.csv$")
  
  if (is.na(matches[1,1])) {
    warning(paste("Filename does not match expected pattern:", file_name))
    return(NULL)
  }

  
  reachname <- 
  river <- matches[1,2]
  pool <- matches[1,3]
  reach <- matches[1,4]
  reachname <-paste0("CEMVR_",river,"_",pool,"_MVR_",reach) 
  
  # Read data
  df <- read_csv(file_path, show_col_types = FALSE)
  
  # Add extracted info
  df <- df %>%
    mutate(channelreachidpk = reachname,
           River = river,
           Pool = pool,
           Reach = reach)
  
  return(df)
}

# Apply to all files and combine
csat_dredge_data <- file_list %>%
  lapply(read_and_annotate) %>%
  bind_rows()

csat_dredged<-rename(csat_dredge_data, VolumeChange = `Volume Change (cy)`)

glimpse(csat_dredged)

data_cleaned <- csat_dredged|>
  filter(VolumeChange>500)


CSAT_dredge_data <- data_cleaned|>
  left_join(reach_data, join_by(channelreachidpk))|>
  select("SurveyDate","VolumeChange","channelreachidpk","River","Pool", "Reach",
         "mileagestart","mileageend","depthmaintained", "reportedlength")

CSAT_dredge_date <- CSAT_dredge_data |>
  mutate("Date" = ymd(SurveyDate))

dredge_date <- dredge_data|>
  mutate("Date" = mdy_hm(DATE_START))|>
  mutate("VolumeChange" = VOLUMEDREDGED)

CSAT_dredge_data <- CSAT_dredge_date |>
  mutate(dataset = "CSAT Dredge")

dredge_data <- dredge_date |>
  mutate(dataset = "Dredge Data")

# Combine the two datasets
combined_data <- bind_rows(CSAT_dredge_data, dredge_data)

overlap_data <- CSAT_dredge_data |>
  inner_join(dredge_data, by = "Date", suffix = c("_csat", "_dredge"), relationship = "many-to-many")|>
  select(SurveyDate, River,Pool,VolumeChange_csat,VolumeChange_dredge)|>
  mutate("VolDiff" = VolumeChange_csat - VolumeChange_dredge)

# View the overlapping data
head(overlap_data)

# Create the plot
ggplot(combined_data, aes(x = Date, y = VolumeChange, color = dataset)) +
  geom_point() +  # or geom_point() depending on your preference
  labs(x = "Date", y = "Volume", title = "Volume over Time by Dataset") +
  theme_minimal() +
  scale_color_manual(values = c("CSAT Dredge" = "blue", "Dredge Data" = "red")) 

write_csv(CSAT_dredge_data, "C:/workspace/DRAFT/CSAT_Data/Dredged_Date_Vol.csv")

ggplot(CSAT_dredge_data, aes(x = SurveyDate, y = VolumeChange, colour = River)) +
  geom_point(size = 2)  # Scatter plot

Miss<-dredge_data |>
  filter(RIVER == "Mississippi_River")

IL<-dredge_data |>
  filter(RIVER == "Illinois_Waterway")
  mean(IL$VOLUMEDREDGED)
  mean(Miss$VOLUMEDREDGED)

