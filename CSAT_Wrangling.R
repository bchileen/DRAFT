library(readr)
library(dplyr)
library(stringr)
library(purrr)
library(ggplot2)
library(lubridate)
library(tidyverse)
library(zoo)
options(scipen = 100, digits = 2)

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
combined_data <- file_list %>%
  lapply(read_and_annotate) %>%
  bind_rows()

combined_data<-rename(combined_data, VolumeChange = `Volume Change (cy)`)

data_cleaned <- combined_data|>
  filter(VolumeChange>500)


CSAT_dredge_data <- data_cleaned|>
  left_join(reach_data, join_by(channelreachidpk))|>
  select("SurveyDate","VolumeChange","channelreachidpk","River","Pool", "Reach",
         "mileagestart","mileageend","depthmaintained", "reportedlength")

write_csv(CSAT_dredge_data, "C:/workspace/DRAFT/CSAT_Data/Dredged_Date_Vol.csv")

ggplot(CSAT_dredge_data, aes(x = SurveyDate, y = VolumeChange, colour = River)) +
  geom_point(size = 2)  # Scatter plot


### Shoaling Rates

files <-list.files(path = folder_path,
                                 pattern = "_SurveyPairVolumeDifference\\.csv$",
                                 full.names = TRUE)

read_and_label2 <- function(file_path) {
  # Extract filename
  file_name <- basename(file_path)
  
  # Remove extension
  file_name_no_ext <- tools::file_path_sans_ext(file_name)
  
  # Split by underscore
  parts <- str_split(file_name_no_ext, "_", simplify = TRUE)
  
  # Extract relevant parts
  river <- parts[1]
  pool <- parts[2]
  reach <- parts[4]  # MVR is at [3], 14 or 22 is at [4]
  
  # Read the data
  data <- read_csv(file_path)
  
  # Add new columns
  data <- data %>%
    mutate(
      river = river,
      pool = pool,
      reach = reach,
      source_file = file_name
    )
  
  return(data)
}

# Apply the function to all files and bind into one big dataframe
combined_data <- map_dfr(files, read_and_label2)

# Check it out
glimpse(combined_data)

combined_data_clean<-combined_data|>
  mutate(Before_date = ymd(SurveyDateBefore),
         After_date = ymd(SurveyDateAfter))|>
  mutate(DateDiff = as.numeric(After_date - Before_date))|>
  mutate(DailyShoalingRate_ftperday = AnnualShoalingRate_ftperyr/DateDiff,
         DailyShoalingVolume_Cyperday = AnnualShoalingVolume_CYperyr/DateDiff)
#  filter(DailyShoalingVolume_Cyperday < 12500000 & DailyShoalingRate_ftperday < 900)


ggplot(combined_data_clean, aes(x = After_date, y = DailyShoalingVolume_Cyperday, colour = river)) +
  geom_point() + 
  geom_line()+
  facet_wrap(~ river)

ggplot(combined_data_clean, aes(x = After_date, y = DailyShoalingRate_ftperday, colour = river)) +
  geom_point() + 
  geom_line()+
  facet_wrap(~ river)


daily_expanded <- combined_data_clean |>
  # Create a sequence of dates for each row
  rowwise() |>
  mutate(DateSeq = list(seq(Before_date, After_date, by = "day"))) |>
  ungroup() |>
  # Expand to one row per day
  unnest(cols = c(DateSeq)) |>
  rename(Date = DateSeq) |>
  # Keep only the relevant daily rate values
  select(Date, DailyShoalingVolume_Cyperday,DailyShoalingRate_ftperday, river, pool, reach)

  glimpse(daily_expanded)

# daily_series <- daily_expanded |>
#   group_by(Date, river, pool, reach) |>
#   summarise(DailyVolume = sum(DailyShoalingVolume_Cyperday, na.rm = TRUE), 
#             DailyRate = sum(DailyShoalingRate_ftperday, na.rm = TRUE),
#             .groups = "drop") |>
#   filter(pool == "LA")|>
#   arrange(river, pool, reach, Date) |>
#   group_by(river, pool, reach) |>
#   summarise(mutate(RollingAvgVol_7d = rollmean(DailyVolume, k = 7, fill = NA, align = "right"),
#          RollingAvgRate_7d = rollmean(DailyRate, k = 7, fill = NA, align = "right")))  |>
#   ungroup()


daily_avg <- daily_expanded %>%
  group_by(Date, river, pool, reach) %>%
  summarise(
    avg_volume = mean(DailyShoalingVolume_Cyperday, na.rm = TRUE),
    count_volume = sum(!is.na(DailyShoalingVolume_Cyperday)),
    avg_rate = mean(DailyShoalingRate_ftperday, na.rm = TRUE),
    count_rate = sum(!is.na(DailyShoalingRate_ftperday)),
    .groups = "drop"
  )|>
  mutate(site = paste(river, pool, reach, sep = "_"))

daily_avg_rolled <- daily_avg %>%
  arrange(site, Date) %>%
  group_by(site) %>%
  mutate(
    avg_volume_7day = rollmean(avg_volume, k = 7, fill = NA, align = "right"),
    avg_rate_7day = rollmean(avg_rate, k = 7, fill = NA, align = "right")
  ) %>%
  ungroup()

write_csv(daily_avg_rolled, "C:/workspace/DRAFT/CSAT_Data/SurveyPairVolumeDifference.csv")

# Loop through each unique value in the `pool` column
for (pool_value in unique(daily_avg_rolled$pool)) {
  
  # Create a new dataframe for each pool value, named dynamically
  assign(paste0("P_", pool_value), daily_avg_rolled %>% filter(pool == pool_value))
}



daily_avg_rolled |>
  filter(pool == "20")|>
  filter(year(Date) >= 2015 & year(Date) <= 2020)|>
  ggplot(aes(x = Date, y = count_rate, colour = river)) +
  geom_point() + 
  geom_line()+
  facet_wrap(~reach)


#, scales = "free_y"
  

# Then: pivot to wide format
daily_avg_wide <- daily_avg %>%
  select(Date, site, avg_volume) %>%
  pivot_wider(
    names_from = site,
    values_from = avg_volume
  )


filtered_daily_series <- daily_series |>
  filter(month(Date) != 1 & month(Date) != 2) |>
  filter(year(Date) >= 2014 & year(Date) <= 2024)|>
  filter(pool = "LA")

# Plot
ggplot(filtered_daily_series, aes(x = Date, y = RollingAvgVol_7d)) +
  geom_point() + 
  geom_line() +
  labs(title = "7-Day Rolling Average of Daily Sediment Volume by Pool",
       x = "Date",
       y = "7-Day Rolling Average (Cy/day)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
