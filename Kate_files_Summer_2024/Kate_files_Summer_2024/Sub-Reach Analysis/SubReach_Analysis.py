# -*- coding: utf-8 -*-
"""
initial code started 6/10/2024
goal: analyze at sub-reach level

6/14/2024: steps changed to include a way to determine which reaches are intersected using NCF feature service
6/26/2024: step 1 completed
7/12/2024: step 2 completed
7/29/2024: step 3 completed
8/5/2024: script validated
"""
import arcpy
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import os
import shutil
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from dateutil.relativedelta import relativedelta
from datetime import datetime

# Inputs that change per user
PolygonFile = Path('..\\data\\shp\\MVR_AOI.shp')  # Path to the constraining shapefile
ParentDir = Path('..\\output\\CEMVR\\19990101_to_20241231_omittedtable')  # Path to the CSAT outputs
# The CSAT outputs should be located in the output folder, right? Should this be default?

# Number of years for analysis
NY = 27

# Name of AOI defined as the name of the shapefile
aoi_name = PolygonFile.stem

# Output directories: the paths below bank on this script being placed in the "source code" folder within CSAT
BaseOutputDir = Path('..\\output\\Sub-Reach Analysis')
OutputDir = os.path.join(BaseOutputDir, aoi_name)  # Folder for the outputs of this code
MovingDir = os.path.join(OutputDir, "WorkingFiles")  # Created path for the intersecting reaches

if os.path.exists(OutputDir):
    response = input(f"Directory for {aoi_name} exists. Overwrite? (y/n): ")
    if response.lower() == 'y':
        shutil.rmtree(OutputDir)
    else:
        print("Operation cancelled.")
        exit()
os.makedirs(MovingDir)
# Do we want to keep the above lines in the code?



# variables from feature service and feature service url for channel reaches
feature_service_url = "https://services7.arcgis.com/n1YM8pTrFmm7L4hs/ArcGIS/rest/services/National_Channel_Framework/FeatureServer/2/query"
fields = "channelreachidpk,depthauthorized,channelareaidfk,sdsfeaturename"
arcpy.env.overwriteOutput = True



"Step 1: Access feature service and determine which reaches intersect with the AOI. These reaches are then pulled from " \
        "the CSAT outputs and copied to a working directory"

def feature_service(url, where="1=1", out_fields=fields, chunk_size=2000):
    """
    This function utilizes the feature service to pull of the fields noted above

    Parameters
    ----------
    url --> feature service url
    out_fields --> fields needed from the feature service
                    (If all fields are needed, put '*' for the field variable above)

    Returns
    -------
    a GeoDataFrame with all the data from the out-fields listed above
    """
    all_features = []
    offset = 0
    while True:
        query_params = {
            'where': where,
            'outFields': out_fields,
            'f': 'geojson',
            'resultOffset': offset,
            'resultRecordCount': chunk_size
        }
        response = requests.get(url, params=query_params)
        response.raise_for_status()  # Ensure we raise an error if the request fails
        data = response.json()

        # Append features
        if 'features' in data:
            all_features.extend(data['features'])
        else:
            break

        # Check if we have received fewer features than the chunk size
        if len(data['features']) < chunk_size:
            break

        offset += chunk_size

    # Convert the list of features to a GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(all_features)
    return gdf

def remove_district_prefix(intersect):
    """
    This function removes the three-letter district prefix from each channel ID in the list.

    Parameter
    ----------
    intersect --> list of reach IDs with district prefixes

    Returns
    ------
    list of reach IDs without the district prefixes
    """
    return [channelreachidpk[6:] for channelreachidpk in intersect]

def copy_files(file_names, source_dir, dest_dir, extensions):
    """
    This function copies the intersecting reach files from the Shoaling Rates and Last Surveys folders from the CSAT output files

    Parameters
    ----------
    file_names --> list of intersecting channel reach IDs
    source_dir --> path to the CSAT output files
    dest_dir --> path to the working directory where the files are being copied to
    extensions --> list of file extensions to be copied
    """
    # Iterate over the directory tree
    for root, dirs, files in os.walk(source_dir):
        for file_name in file_names:
            for ext in extensions:
                file_with_ext = f"{file_name}{ext}"
                if file_with_ext in files:
                    # Find all occurrences of the file with the same name and extension
                    file_paths = [os.path.join(root, file_with_ext) for root, dirs, files in os.walk(source_dir) if
                                  file_with_ext in files]

                    for source_file in file_paths:
                        # Determine relative path from source directory
                        relative_path = os.path.relpath(os.path.dirname(source_file), source_dir)

                        # Create corresponding destination directory
                        destination_dir = os.path.join(dest_dir, relative_path)
                        if not os.path.exists(destination_dir):
                            os.makedirs(destination_dir)

                        destination_file = os.path.join(destination_dir, file_with_ext)

                        # Copy the file to the destination directory
                        shutil.copy2(source_file, destination_file)
                        print(f"Copied: {file_with_ext} to {destination_dir}")


"Step 2: Clip rasters based on shapefile and merge together into one TIFF file"

def clip_raster(raster_path, shp, output_path):
    """
    This function takes the raster and clips it based on the shapefile provided

    Parameters
    ----------
    raster_path --> path of the raster file(s) to clip
    shp --> shapefile to clip the raster files
    output_path --> where the clipped files will be located
    """
    # Ensure the shapefile is in the same CRS as the raster
    with rasterio.open(raster_path) as src:
        shp = shp.to_crs(src.crs)

    # Get the geometry that will be used to clip
    geometry = [mapping(geom) for geom in shp.geometry]

    # Perform the clipping
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, geometry, crop=True)
        out_meta = src.meta.copy()

        # Update the metadata
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Write the clipped raster
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

def process_raster(input_dir, output_dir, polygon, aoi):
    """
    This function processes the rasters when there is more than one reach intersecting the AOI.

    Parameters
    ----------
    input_dir --> where the files are to clip
    output_dir --> where the clipped files will be located
    polygon --> constraining shapefile
    """
    for root, dirs, files in os.walk(input_dir):
        if root == input_dir:
            continue

        subdir_name = os.path.basename(root)
        print(f"Processing subdirectory: {subdir_name}")

        new_dir = os.path.join(output_dir, subdir_name)
        os.makedirs(new_dir, exist_ok=True)

        # Clip each raster file
        for file in files:
            if file.endswith(('.tif', '.tiff')):
                input_raster = os.path.join(root, file)
                output_raster = os.path.join(new_dir, f"clipped_{aoi}_{file}")

                try:
                    clip_raster(input_raster, polygon, output_raster)
                    print(f"Clipped raster saved to: {output_raster}")
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

def process_raster_onereach(input_dir, output_dir, polygon, aoi_name):
    """
    This function processes the rasters when there is only one reach intersecting the AOI.

    Parameters
    ----------
    input_dir --> where the files are to clip
    output_dir --> where the clipped files will be located
    polygon --> constraining shapefile
    aoi_name --> name of the AOI to add to the file name
    """
    for root, dirs, files in os.walk(input_dir):
        if root == input_dir:
            continue

        subdir_name = os.path.basename(root)
        print(f"Processing subdirectory: {subdir_name}")

        new_dir = os.path.join(output_dir, subdir_name)
        os.makedirs(new_dir, exist_ok=True)

        # Clip each raster file
        for file in files:
            if file.endswith(('.tif', '.tiff')):
                input_raster = os.path.join(root, file)
                output_raster = os.path.join(new_dir, f"{subdir_name}_{aoi_name}.tif")

                try:
                    clip_raster(input_raster, polygon, output_raster)
                    print(f"Clipped raster saved to: {output_raster}")
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

def merge_rasters_in_subdirs(root_dir, aoi):
    """
    This function is for when there is more than one reach that intersects the shapefile. It takes the clipped files
    and merges them together into one raster file.

    Parameters
    ----------
    root_dir --> directory where the clipped files are located
    aoi --> name of the AOI to add to file name
    """
    arcpy.env.overwriteOutput = True

    for subdir, dirs, files in os.walk(root_dir):
        if subdir == root_dir:
            continue  # Skip the root directory itself

        # Filter for raster files
        raster_files = [os.path.join(subdir, f) for f in files if f.startswith(f"clipped_{aoi}")]

        if not raster_files:
            continue  # Skip directories with no raster files

        # Create the output filename
        subdir_name = os.path.basename(subdir)
        out_path = os.path.join(root_dir, subdir_name)
        out_file = os.path.join(out_path, f"{subdir_name}_{aoi}.tif")

        # Merge rasters
        arcpy.management.MosaicToNewRaster(
            input_rasters=raster_files,
            output_location=subdir,
            raster_dataset_name_with_extension=f"{subdir_name}_{aoi}.tif",
            coordinate_system_for_the_raster="#",  # Use the coordinate system of the first raster
            pixel_type="32_BIT_FLOAT",
            cellsize="#",  # Use the cell size of the first raster
            number_of_bands="1",
            mosaic_method="LAST",
            mosaic_colormap_mode="FIRST"
        )

        print(f"Merged raster created: {out_file}")


def delete_working_files(root_dir):
    """
    This files deletes the temporary working directory that held the copies of the CSAT outputs.

    Parameter
    ----------
    root_directory --> temporary working directory

    Note: This is a separate function from "delete_unwanted_folders" so that the CSAT outputs can be removed before the
    clipped files are processed.
    """
    # Delete the root directory
    try:
        shutil.rmtree(root_dir)
        print(f"Root directory '{root_dir}' has been deleted.")
    except Exception as e:
        print(f"Error deleting root directory '{root_dir}': {e}")

def delete_unwanted_folders(directory, keep_folders):   # Change this to delete empty folders
    """
    Delete folders in the given directory that are not in the keep_folders list.

    The "clip" function creates a folder for each subdirectory, even though they are unnecessary. This function deletes
    folder except for the "Last Surveys" and "Shoaling Rates" folders.

    Parameters
    ----------
    directory --> output directory
    keep_folders --> the names of the folders we want to keep
    """
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and item not in keep_folders:
            try:
                shutil.rmtree(item_path)
                print(f"Deleted folder: {item_path}")
            except Exception as e:
                print(f"Error deleting folder {item_path}: {e}")


"Step 3: Generate CSAT output tables for the Area of Interest. At this point, we are generating volume and SPQ tables."

# The two following functions are from the CSAT code; they might need changes
def dredge_vol_calculator(last_bathy=None, auth_elev=None, time_til_dredge=None, shoal_multi=None,
                          cell_size=10, unit_conv=27):
    """

    Computes the volume of sediment shallower than target dredge cut for predicted shoaled bathymetry

    Parameters
    ----------
    last_bathy : np.array
        one dimensional array of the "Now" survey conditions (elevation is positive)
    auth_elev : float
        volume shallower than this target cut depth will be calculated
    time_til_dredge : float
        time since "Now" conditions to compute shoaled bathy, should be provided as decimal year
        (e.g. 3 months = 0.25, 6 months = 0.5, and 24 months = 2.0)
    shoal_multi : np.array
        annual shoaling rates at each point in the reach
    cell_size : float, default = 10.0
        dimensions of raster cell size.
    unit_conv : float, default = 27
        conversion factor used to convert raster cell size to units of cubic yards.
        Default is 27 in order to convert dZ (ft) * cell_size (ft) * cell_size (ft) = volume_ft3 (ft^3)
        to volume_yd -> volume_yd = volume_ft3/unit_cont. If cell_size is not in units of feet, unit_conv needs to
        provide the appropriate conversion from (units of cell_size)^3 to yds^3

    Returns
    -------
    volume : float
        volume of sediment shallower than auth_elev
    dredged_bathy : np.array
        the predicted shoaled bathymetry as a 1D array
    """

    dredged_bathy = last_bathy + time_til_dredge * shoal_multi

    volholder = dredged_bathy - auth_elev

    #   Modified by Jay to use logical indexing 2015 May 12
    volholder[volholder < 0] = 0

    volume = np.nansum(volholder) * (cell_size * cell_size) / unit_conv

    return volume, dredged_bathy

def find_csv(root_dir, target_file):
    """
    Locates the necessary file (can be used for any files, not just csv)

    Parameters
    ----------
    root_dir --> directory where the desired file is located
    target_file --> name of the desired file

    Return
    ------
    desired file(s)
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if target_file in filenames:
            return os.path.join(dirpath, target_file)
    return None

def find_date(text):
    """
    Finds the dates within the survey names and extracts them

    Parameter
    ----------
    text --> survey names with dates embedded

    Return
    ------
    date that was embedded in survey title
    """
    for i in range(len(text) - 7):
        potential_date = text[i:i + 8]
        if potential_date.isdigit() and len(potential_date) == 8:
            year = int(potential_date[:4])
            month = int(potential_date[4:6])
            day = int(potential_date[6:])
            if 1 <= month <= 12 and 1 <= day <= 31:
                return potential_date
    return None


# Generate Volume and SPQ tables
def generate_tables_one(output_directory, nYears, aoi, aoi_dir):
    """
    This function generates the volumes and SPQ tables for the intersecting reaches, only the portions that are located within
    the Area of Interest. Two tables are returned for each AOI

    Parameters
    ----------
    output_directory --> OutputDir
    nYears --> Number of years for analysis
    aoi --> name of the area of interest being processed
    aoi_dir --> base name of the area of interest to use for the directory

    Returns
    -------
    volumes and spq dataframes
    exported csvs of volumes and spq

    Note: This function is used when there is only one reach that intersects with the area(s) of interest.
    """
    unit_conv = 27
    cell_size = 10

    # last bathymetry survey (CSAT output from Step 2)
    last_bathy_survey = Path('..\\output\\Sub-Reach Analysis\\' + aoi_dir + '\\Last Surveys\\Last Surveys_' + aoi + '.tif')
    with rasterio.open(last_bathy_survey) as src:
        raster_array = src.read(1)
        no_data = src.nodata  # Get the no-data value

        # Replace no-data with numpy NaN
        raster_array = np.where(raster_array == no_data, np.nan, raster_array)

        # Flatten to 1D
        last_bathy = raster_array.flatten()

    # requested shoaling rates (CSAT output from Step 2)
    shoaling_rates = Path('..\\output\\Sub-Reach Analysis\\' + aoi_dir + '\\Shoaling Rates\\Shoaling Rates_' + aoi + '.tif')
    with rasterio.open(shoaling_rates) as src:
        raster_array = src.read(1)
        no_data = src.nodata  # Get the no-data value

        # Replace no-data with numpy NaN
        raster_array = np.where(raster_array == no_data, np.nan, raster_array)

        # Flatten to 1D
        requested_shoaling_rate_results = raster_array.flatten()

        # Enforce extreme shoaling threshold
        requested_shoaling_rate_results[(requested_shoaling_rate_results == -9999)] = 0  # Allowing scour
        shoaling_threshold = 10
        requested_shoaling_rate_results[requested_shoaling_rate_results > shoaling_threshold] = shoaling_threshold

    # Feature Service Dataframe
    fs_df = pd.DataFrame(clipped_gdf)

    for reach in intersect:
        # Find index value of reach in the Feature Service df
        intersecting_reach = reach
        # intersecting_reach = intersecting_reaches[0]
        reach_index_long = fs_df.index[fs_df['channelreachidpk'] == intersecting_reach].tolist()
        reach_index = reach_index_long[0]
        reach_auth_depth = fs_df._get_value(index=reach_index, col='depthauthorized')

    authorized_depth = reach_auth_depth

    start_elev = np.ceil(authorized_depth + authorized_depth * 2.0) * -1  # deepest allowable cut depth (in elev)

    cutoff_elev = np.ceil(authorized_depth - authorized_depth * 1.0) * -1  # shallowest allowable cut depth (in elev)

    # Limits for SPQ table generation
    subtractLimit = 16
    plusLimit = 16

    # ensure that depths 16 ft deeper than AD are reported (a check for shallow draft channels)
    start_elev = min(start_elev, - np.ceil(authorized_depth + subtractLimit))

    if start_elev == 0:
        start_elev = -subtractLimit

    OD = 0  # Over-Draft allowance
    AM = 0  # Advanced Maintenance allowance
    inc = 1  # dredging increment

    cut = np.arange(start_elev - OD - AM, cutoff_elev + 1, inc)

    # TODO: Allow user to specify forecast interval (e.g. 1-month, 6-month, annual)
    shoal_forecast_interval = 6  # months
    assert (shoal_forecast_interval > 0)

    time_keeper = np.zeros(2 * nYears + 1, dtype=np.int_)

    nContours = len(cut)
    volume = np.zeros((nContours, time_keeper.size))

    for k in range(nContours):
        dredge_elev = cut[k]
        # looping over 6-month increments for nYears worth of dredging predictions
        for n in range(2 * nYears + 1):
            time_til_dredge = (n + 1) * 0.5 - 0.5

            time_keeper[n] = time_til_dredge * 12

            volume[k, n] = dredge_vol_calculator(last_bathy, dredge_elev, time_til_dredge,
                                                 requested_shoaling_rate_results, cell_size, unit_conv)[0]

    # combining depth contours and dredging volume predictions to speed up output
    output_df = pd.DataFrame(volume,
                             columns=['Now', '6_months', '12_months', '18_months', '24_months', '30_months',
                                      '36_months', '42_months', '48_months', '54_months', '60_months'])
    output_df.insert(loc=0, column='dredge_cut_ft', value=cut.T)

    # Go through output_df column-by-column and round to the nearest one hundred, skipping the columns we don't want to round.
    df_column_names = list(output_df.columns)
    for name in df_column_names:
        if name != 'dredge_cut_ft':
            this_column = list(output_df[name])
            floored_column = np.multiply(np.floor(np.divide(this_column, 100)), 100)
            # divide by 100 to shift decimal two places left, then floor values, then multiply by 100 to move the decimal back. (no need to convert to int, the float format '%.0f' in pandas.read_csv takes care of that)
            output_df[name] = floored_column

    filename_volumes = os.path.join(output_directory, aoi + '_volumes.csv')  # Changed from reach_name
    output_df.to_csv(filename_volumes, float_format='%.0f', index=False)
    print(f"Volumes table created and exported successfully.")


        ### SPQ ###
    # Survey Planning Quantity
    spq_df = pd.DataFrame()

    # Define reach attributes
    sheet_name = fs_df._get_value(index=reach_index, col='sdsfeaturename')
    aoi_area = fs_df._get_value(index=reach_index, col='channelareaidfk')

    # Remove district label from reach and channel area
    reach_id = reach[6:]
    aoi_area = aoi_area[6:]

    # Access the avg max_min table from the CSAT outputs
    csv_name = reach_id + "_avg_max_min.csv"
    csv_path = find_csv(ParentDir, csv_name)
    if csv_path is not None:
        csv_df = pd.read_csv(csv_path)

        # Grab the last survey date for the reach from the max_min table
        long_survey_date = csv_df.LastZ_SurvID_ft
        survey_dates = []
        for date in long_survey_date:
            survey_date = find_date(date)
            survey_dates.append(survey_date)

        date_array = np.array(survey_dates)
        if np.all(date_array == date_array[0]):
            survey_date_last = date_array[0]
        else:
            date_array_int = date_array.astype(int)
            survey_date_max = np.max(date_array_int)
            survey_date_last = survey_date_max.astype(str)

        last_survey_date = datetime.strptime(survey_date_last, "%Y%m%d")
    else:
        last_survey_date = "Unable to obtain last survey date."

    reach_depth_int = np.ceil(authorized_depth)
    consideredDepths = np.arange(start=reach_depth_int - subtractLimit, stop=reach_depth_int + plusLimit + 1,
                                 dtype=np.int_)

    consideredDepths[consideredDepths < 0] = 0

    VAs = np.char.add("VA_s", np.arange(1, plusLimit + 1).astype(str))
    VAp = np.char.add("VA_p", np.arange(subtractLimit, -1, -1).astype(str))
    # VBs = np.char.replace(VAs, "VA", "VB")
    # VBp = np.char.replace(VAp, "VA", "VB")

    spq_idx = np.in1d(-cut, consideredDepths).nonzero()[0]
    # numpy in1d returns only unique values, in some cases there are multiple 0's
    # copying the last zero position to keep the indexing same across
    while len(spq_idx) != len(consideredDepths):
        spq_idx = np.append(spq_idx, spq_idx[-1])

    spq_output = volume[spq_idx]

    VAp0top2 = np.arange(reach_depth_int, reach_depth_int + 3)

    VA_idx = np.in1d(-cut, VAp0top2, ).nonzero()[0]

    spq_p0top2_output = volume[VA_idx]

    next_dredge_time_string = []

    for iMonth in range(len(time_keeper)):
        next_dredge_time_string.append(last_survey_date + relativedelta(months=time_keeper[iMonth]))

    spq_temp_df = pd.DataFrame(spq_output.T, columns=np.concatenate((VAp, VAs)))

    spq_temp_df.insert(loc=0, column='TimeToDredge', value=time_keeper)
    spq_temp_df.insert(loc=1, column='Sheet_Name', value=sheet_name)
    spq_temp_df.insert(loc=2, column='Channel_Area_ID_FK', value=aoi_area)
    spq_temp_df.insert(loc=3, column='R_Q_NAME', value=reach_id)  # replace with reach_name or something like that
    spq_temp_df.insert(loc=4, column='Depth', value=authorized_depth)
    spq_temp_df.insert(loc=5, column='PotentialDredgeDate', value=next_dredge_time_string)
    # spq_temp_df['PotentialDredgeDate'] = next_dredge_time_string
    spq_temp_p0_df = pd.DataFrame(np.flip(spq_p0top2_output.T, 1), columns=['VA_P0', 'VA_P1', 'VA_P2'])
    spq_temp_df = pd.concat([spq_temp_df, spq_temp_p0_df], axis=1)
    spq_df = pd.concat([spq_df, spq_temp_df], axis=0)


    spq_filename = os.path.join(output_directory, 'SPQ_' + aoi + '.csv')
    spq_df.to_csv(spq_filename, float_format='%.0f', date_format='%Y-%m-%d', index=False)
    print(f"SPQ table created and exported successfully.")

    return output_df, spq_df

def generate_volumes_mult(output_directory, nYears, aoi, aoi_dir):
    """
    This function generates the volumes table for the intersecting reaches, only the portions that are located within
    the Area of Interest. One table is returned for each AOI

    Parameters
    ----------
    output_directory --> OutputDir
    nYears --> Number of years for analysis
    aoi --> name of the area of interest being processed
    aoi_dir --> base name of the area of interest to use for the directory

    Returns
    -------
    volumes dataframe
    exported csv of volumes

    Note: This function is used when there is more than one reach that intersects with the area(s) of interest.
    """
    unit_conv = 27
    cell_size = 10

    # last bathymetry survey (CSAT output from Step 2)
    last_bathy_survey = Path(
        '..\\output\\Sub-Reach Analysis\\' + aoi_dir + '\\Last Surveys\\Last Surveys_' + aoi + '.tif')
    with rasterio.open(last_bathy_survey) as src:
        raster_array = src.read(1)
        no_data = src.nodata  # Get the no-data value

        # Replace no-data with numpy NaN
        raster_array = np.where(raster_array == no_data, np.nan, raster_array)

        # Flatten to 1D
        last_bathy = raster_array.flatten()

    # requested shoaling rates (CSAT output from Step 2)
    shoaling_rates = Path(
        '..\\output\\Sub-Reach Analysis\\' + aoi_dir + '\\Shoaling Rates\\Shoaling Rates_' + aoi + '.tif')
    with rasterio.open(shoaling_rates) as src:
        raster_array = src.read(1)
        no_data = src.nodata  # Get the no-data value

        # Replace no-data with numpy NaN
        raster_array = np.where(raster_array == no_data, np.nan, raster_array)

        # Flatten to 1D
        requested_shoaling_rate_results = raster_array.flatten()

        # Enforce extreme shoaling threshold
        requested_shoaling_rate_results[(requested_shoaling_rate_results == -9999)] = 0  # Allowing scour
        shoaling_threshold = 10
        requested_shoaling_rate_results[requested_shoaling_rate_results > shoaling_threshold] = shoaling_threshold

    # Feature Service Dataframe
    fs_df = pd.DataFrame(clipped_gdf)

    total_reach_auth_depth = []

    for reach in intersect:
        # Find index value of reach in the Feature Service df
        intersecting_reach = reach
        # intersecting_reach = intersecting_reaches[0]
        reach_index_long = fs_df.index[fs_df['channelreachidpk'] == intersecting_reach].tolist()
        reach_index = reach_index_long[0]

        # Find authorized depths and add to array
        reach_auth_depths = fs_df._get_value(index=reach_index, col='depthauthorized')
        total_reach_auth_depth.append(reach_auth_depths)

    depth_array = np.array(total_reach_auth_depth)
    if np.all(depth_array == depth_array[0]):
        authorized_depth = depth_array[0]
    else:
        authorized_depth = np.max(depth_array)  # if depths are different, takes the deepest

    start_elev = np.ceil(authorized_depth + authorized_depth * 2.0) * -1  # deepest allowable cut depth (in elev)

    cutoff_elev = np.ceil(
        authorized_depth - authorized_depth * 1.0) * -1  # shallowest allowable cut depth (in elev)

    # Limits for SPQ table generation
    subtractLimit = 16

    # ensure that depths 16 ft deeper than AD are reported (a check for shallow draft channels)
    start_elev = min(start_elev, - np.ceil(authorized_depth + subtractLimit))

    OD = 0  # Over-Draft allowance
    AM = 0  # Advanced Maintenance allowance
    inc = 1  # dredging increment

    cut = np.arange(start_elev - OD - AM, cutoff_elev + 1, inc)

    time_keeper = np.zeros(2 * nYears + 1, dtype=np.int_)

    nContours = len(cut)
    volume = np.zeros((nContours, time_keeper.size))

    for k in range(nContours):
        dredge_elev = cut[k]
        # looping over 6-month increments for nYears worth of dredging predictions
        for n in range(2 * nYears + 1):
            time_til_dredge = (n + 1) * 0.5 - 0.5

            time_keeper[n] = time_til_dredge * 12

            volume[k, n] = dredge_vol_calculator(last_bathy, dredge_elev, time_til_dredge,
                                                 requested_shoaling_rate_results, cell_size, unit_conv)[0]

    # combining depth contours and dredging volume predictions to speed up output
    output_df = pd.DataFrame(volume,
                             columns=['Now', '6_months', '12_months', '18_months', '24_months', '30_months',
                                      '36_months', '42_months', '48_months', '54_months', '60_months'])
    output_df.insert(loc=0, column='dredge_cut_ft', value=cut.T)

    # Go through output_df column-by-column and round to the nearest one hundred, skipping the columns we don't want to round.
    df_column_names = list(output_df.columns)
    for name in df_column_names:
        if name != 'dredge_cut_ft':
            this_column = list(output_df[name])
            floored_column = np.multiply(np.floor(np.divide(this_column, 100)),
                                         100)  # divide by 100 to shift decimal two places left, then floor values, then multiply by 100 to move the decimal back. (no need to convert to int, the float format '%.0f' in pandas.read_csv takes care of that)
            output_df[name] = floored_column

    filename_volumes = os.path.join(output_directory, aoi + '_volumes.csv')  # Changed from reach_name
    output_df.to_csv(filename_volumes, float_format='%.0f', index=False)
    print(f"Volumes table created and exported successfully.")
    return output_df


def generate_spq_mult(output_directory, nYears, aoi, aoi_dir):
    """
    This function generates the SPQ table for the intersecting reaches, only the portions that are located within
    the Area of Interest. One table is returned for each AOI

    Parameters
    ----------
    output_directory --> OutputDir
    nYears --> Number of years for analysis
    aoi --> name of the area of interest being processed
    aoi_dir --> base name of the area of interest to use for the directory

    Returns
    -------
    spq dataframe
    exported spq csv

    Note: This function is used when there is more than one reach that intersects with the area(s) of interest.
    """
    unit_conv = 27
    cell_size = 10

    # Feature Service Dataframe
    fs_df = pd.DataFrame(clipped_gdf)

    # Survey Planning Quantity
    spq_df = pd.DataFrame()

    for reach in intersect:
        intersecting_reach = reach
        reach_id = intersecting_reach[6:]
        # last bathymetry survey (CSAT output from Step 2)
        last_bathy_survey = Path('..\\output\\Sub-Reach Analysis\\' + aoi_dir + '\\Last Surveys\\clipped_' + aoi + '_' + reach_id + '.tif')
        if os.path.exists(last_bathy_survey):
            with rasterio.open(last_bathy_survey) as src:
                raster_array = src.read(1)
                no_data = src.nodata  # Get the no-data value

                # Replace no-data with numpy NaN
                raster_array = np.where(raster_array == no_data, np.nan, raster_array)

                # Flatten to 1D
                last_bathy = raster_array.flatten()
        else:
            print(f"{last_bathy_survey} does not exist.")
            continue

        # requested shoaling rates (CSAT output from Step 2)
        shoaling_rates = Path('..\\output\\Sub-Reach Analysis\\' + aoi_dir + '\\Shoaling Rates\\clipped_' + aoi + '_' + reach_id + '.tif')
        if os.path.exists(shoaling_rates):
            with rasterio.open(shoaling_rates) as src:
                raster_array = src.read(1)
                no_data = src.nodata  # Get the no-data value

                # Replace no-data with numpy NaN
                raster_array = np.where(raster_array == no_data, np.nan, raster_array)

                # Flatten to 1D
                requested_shoaling_rate_results = raster_array.flatten()

                # Enforce extreme shoaling threshold
                requested_shoaling_rate_results[(requested_shoaling_rate_results == -9999)] = 0  # Allowing scour
                shoaling_threshold = 10
                requested_shoaling_rate_results[
                    requested_shoaling_rate_results > shoaling_threshold] = shoaling_threshold
        else:
            print(f"{shoaling_rates} does not exist.")
            continue

        # Find index value of reach in the Feature Service df
        # intersecting_reach = intersecting_reaches[0]
        reach_index_long = fs_df.index[fs_df['channelreachidpk'] == intersecting_reach].tolist()
        reach_index = reach_index_long[0]
        # Find authorized depths and add to array
        authorized_depth = fs_df._get_value(index=reach_index, col='depthauthorized')

        start_elev = np.ceil(authorized_depth + authorized_depth * 2.0) * -1  # deepest allowable cut depth (in elev)

        cutoff_elev = np.ceil(
        authorized_depth - authorized_depth * 1.0) * -1  # shallowest allowable cut depth (in elev)

        # Limits for SPQ table generation
        subtractLimit = 16
        plusLimit = 16

        # ensure that depths 16 ft deeper than AD are reported (a check for shallow draft channels)
        start_elev = min(start_elev, - np.ceil(authorized_depth + subtractLimit))

        OD = 0  # Over-Draft allowance
        AM = 0  # Advanced Maintenance allowance
        inc = 1  # dredging increment

        cut = np.arange(start_elev - OD - AM, cutoff_elev + 1, inc)

        # TODO: Allow user to specify forecast interval (e.g. 1-month, 6-month, annual)
        shoal_forecast_interval = 6  # months
        assert (shoal_forecast_interval > 0)

        time_keeper = np.zeros(2 * nYears + 1, dtype=np.int_)

        nContours = len(cut)
        volume = np.zeros((nContours, time_keeper.size))

        for k in range(nContours):
            dredge_elev = cut[k]
            # looping over 6-month increments for nYears worth of dredging predictions
            for n in range(2 * nYears + 1):
                time_til_dredge = (n + 1) * 0.5 - 0.5

                time_keeper[n] = time_til_dredge * 12

                volume[k, n] = dredge_vol_calculator(last_bathy, dredge_elev, time_til_dredge,
                                                 requested_shoaling_rate_results, cell_size, unit_conv)[0]

        ### SPQ ###
        # Define reach attributes for SPQ table
        sheet_name = fs_df._get_value(index=reach_index, col='sdsfeaturename')
        aoi_area = fs_df._get_value(index=reach_index, col='channelareaidfk')
        aoi_area = aoi_area[6:]

        # Access the avg max_min table from the CSAT outputs
        csv_name = reach_id + "_avg_max_min.csv"
        csv_path = find_csv(ParentDir, csv_name)
        if csv_path is not None:
            csv_df = pd.read_csv(csv_path)

            # Grab the last survey date for the reach from the max_min table
            long_survey_date = csv_df.LastZ_SurvID_ft
            survey_dates = []
            for date in long_survey_date:
                survey_date = find_date(date)
                survey_dates.append(survey_date)

            date_array = np.array(survey_dates)
            if np.all(date_array == date_array[0]):
                survey_date_max = date_array[0]
                survey_date_last = survey_date_max.astype(str)
            else:
                date_array_int = date_array.astype(int)
                survey_date_max = np.max(date_array_int)
                survey_date_last = survey_date_max.astype(str)

            last_survey_date = datetime.strptime(survey_date_last, "%Y%m%d")
        else:
            survey_date_max = "18500101"  # What should be here? Another way to find last survey date? Access feature service?
            last_survey_date = datetime.strptime(survey_date_max, "%Y%m%d")

        reach_depth_int = np.ceil(authorized_depth)
        consideredDepths = np.arange(start=reach_depth_int - subtractLimit, stop=reach_depth_int + plusLimit + 1,
                                     dtype=np.int_)

        consideredDepths[consideredDepths < 0] = 0

        VAs = np.char.add("VA_s", np.arange(1, plusLimit + 1).astype(str))
        VAp = np.char.add("VA_p", np.arange(subtractLimit, -1, -1).astype(str))
        # VBs = np.char.replace(VAs, "VA", "VB")
        # VBp = np.char.replace(VAp, "VA", "VB")

        spq_idx = np.in1d(-cut, consideredDepths).nonzero()[0]
        # numpy in1d returns only unique values, in some cases there are multiple 0's
        # copying the last zero position to keep the indexing same across
        while len(spq_idx) != len(consideredDepths):
            spq_idx = np.append(spq_idx, spq_idx[-1])

        spq_output = volume[spq_idx]

        VAp0top2 = np.arange(reach_depth_int, reach_depth_int + 3)

        VA_idx = np.in1d(-cut, VAp0top2, ).nonzero()[0]

        spq_p0top2_output = volume[VA_idx]

        next_dredge_time_string = []

        for iMonth in range(len(time_keeper)):
            next_dredge_time_string.append(last_survey_date + relativedelta(months=time_keeper[iMonth]))

        spq_temp_df = pd.DataFrame(spq_output.T, columns=np.concatenate((VAp, VAs)))

        spq_temp_df.insert(loc=0, column='TimeToDredge', value=time_keeper)
        spq_temp_df.insert(loc=1, column='Sheet_Name', value=sheet_name)
        spq_temp_df.insert(loc=2, column='Channel_Area_ID_FK', value=aoi_area)
        spq_temp_df.insert(loc=3, column='R_Q_NAME', value=reach_id)  # replace with reach_name or something like that
        spq_temp_df.insert(loc=4, column='Depth', value=authorized_depth)
        spq_temp_df.insert(loc=5, column='PotentialDredgeDate', value=next_dredge_time_string)
        # spq_temp_df['PotentialDredgeDate'] = next_dredge_time_string
        spq_temp_p0_df = pd.DataFrame(np.flip(spq_p0top2_output.T, 1), columns=['VA_P0', 'VA_P1', 'VA_P2'])
        spq_temp_df = pd.concat([spq_temp_df, spq_temp_p0_df], axis=1)
        spq_df = pd.concat([spq_df, spq_temp_df], axis=0)

    spq_filename = os.path.join(output_directory, 'SPQ_' + aoi + '.csv')
    spq_df.to_csv(spq_filename, float_format='%.0f', date_format='%Y-%m-%d', index=False)
    print(f"SPQ table created and exported successfully.")

    return spq_df


# Access Feature Service
gdf_to_clip = feature_service(feature_service_url)

# Clip GeoDataFrame with shapefile to determine which reaches lie within the AOI
boundary = gpd.read_file(PolygonFile)

# Ensure same CRS
if gdf_to_clip.crs is None:
    gdf_to_clip = gdf_to_clip.set_crs(epsg=4326)

if gdf_to_clip.crs != boundary.crs:
    boundary = boundary.to_crs(gdf_to_clip.crs)

num_polygons = len(boundary)

if num_polygons == 1:
    # Perform the clipping
    clipped_gdf = gpd.clip(gdf_to_clip, boundary)

    # Define the fields pulled from the feature service
    intersect = clipped_gdf.channelreachidpk
    auth_depth = clipped_gdf.depthauthorized

    # Remove district from Reach IDs
    intersecting_reaches = remove_district_prefix(intersect)

    # Determine number of reaches that intersect with the shapefile
    num_intersects = len(intersecting_reaches)

    if num_intersects == 1:
        # Copy CSAT output files that intersect with AOI into a new directory
        copy_files(intersecting_reaches, ParentDir, MovingDir, [".tif"])

        # Step 2 Functions
        process_raster_onereach(MovingDir, OutputDir, boundary, aoi_name)
        delete_working_files(MovingDir)

        # Step 3 Functions
        generate_tables_one(OutputDir, NY, aoi_name, aoi_name)
        folders_to_keep = ["Last Surveys", "Shoaling Rates"]
        delete_unwanted_folders(OutputDir, folders_to_keep)
    elif num_intersects > 1:
        # Copy CSAT output files that intersect with AOI into a new directory
        copy_files(intersecting_reaches, ParentDir, MovingDir, [".tif"])

        # Step 2 Functions
        process_raster(MovingDir, OutputDir, boundary, aoi_name)
        delete_working_files(MovingDir)
        merge_rasters_in_subdirs(OutputDir, aoi_name)

        # Step 3 Functions
        generate_volumes_mult(OutputDir, NY, aoi_name, aoi_name)
        generate_spq_mult(OutputDir, NY, aoi_name, aoi_name)
        folders_to_keep = ["Last Surveys", "Shoaling Rates"]
        delete_unwanted_folders(OutputDir, folders_to_keep)
    else:
        print(f"No reaches processed by CSAT intersect with shapefile.")
        quit()
    print(f"Analysis complete. Number of reaches analyzed: {num_intersects}")
elif num_polygons > 1:
    individual_gdfs = []
    for index, row in boundary.iterrows():
        single_polygon_gdf = gpd.GeoDataFrame([row], crs=boundary.crs)
        individual_gdfs.append(single_polygon_gdf)

    for i, single_gdf in enumerate(individual_gdfs):
        mult_aoi_name = f"{aoi_name}_{i + 1}"
        polygon = single_gdf

        # Perform the clipping
        clipped_gdf = gpd.clip(gdf_to_clip, polygon)

        # Define the fields pulled from the feature service
        intersect = clipped_gdf.channelreachidpk
        auth_depth = clipped_gdf.depthauthorized

        # Remove district from Reach IDs
        intersecting_reaches = remove_district_prefix(intersect)

        # Determine number of reaches that intersect with the shapefile
        num_intersects = len(intersecting_reaches)

        if num_intersects == 1:
            # Copy CSAT output files that intersect with AOI into a new directory
            copy_files(intersecting_reaches, ParentDir, MovingDir, [".tif"])

            # Step 2 Functions
            process_raster_onereach(MovingDir, OutputDir, polygon, mult_aoi_name)
            delete_working_files(MovingDir)

            # Step 3 Functions
            generate_tables_one(OutputDir, NY, mult_aoi_name, aoi_name)
            folders_to_keep = ["Last Surveys", "Shoaling Rates"]
            delete_unwanted_folders(OutputDir, folders_to_keep)
        elif num_intersects > 1:
            # Copy CSAT output files that intersect with AOI into a new directory
            copy_files(intersecting_reaches, ParentDir, MovingDir, [".tif"])

            # Step 2 Functions
            process_raster(MovingDir, OutputDir, polygon, mult_aoi_name)  # mult version?
            delete_working_files(MovingDir)
            merge_rasters_in_subdirs(OutputDir, mult_aoi_name)

            # Step 3 Functions
            generate_volumes_mult(OutputDir, NY, mult_aoi_name, aoi_name)
            generate_spq_mult(OutputDir, NY, mult_aoi_name, aoi_name)
            folders_to_keep = ["Last Surveys", "Shoaling Rates"]
            delete_unwanted_folders(OutputDir, folders_to_keep)
        else:
            print(f"No reaches processed by CSAT intersect with shapefile.")
            quit()
        print(f"Analysis complete. Multiple polygons detected. Number of reaches analyzed: {num_intersects}")
else:
    print(f"Error: No polygons detected. Use different shapefile.")