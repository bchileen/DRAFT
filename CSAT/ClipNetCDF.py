#!/usr/bin/env python3
"""
CSAT-Compatible NetCDF Clipper

This script clips NetCDF files to a new shapefile boundary while maintaining
full compatibility with the Corps Shoaling Analysis Tool (CSAT).

Key Features:
- Preserves all NetCDF attributes required by CSAT
- Maintains data structure and fill value conventions
- Handles coordinate system transformations
- Creates diagnostic plots
- Generates processing summary

Usage:
    python csat_clipper.py --nc_dir <path> --shp_file <path> --output_dir <path> [--reach_table <path>]
"""

import os
import sys
import argparse
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point, box
import pandas as pd
from pyproj import CRS
import logging
import netCDF4 as nc
from tqdm import tqdm


def setup_logging(output_dir):
    """Setup logging to file and console"""
    log_file = os.path.join(output_dir, 'csat_clipping.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_epsg_code(projection_name):
    """Convert projection names to EPSG codes (State Plane systems)"""
    projection_dict = {
        'Illinois West': '3436',
        'Illinois East': '3435',
        'Illinois_West': '3436',
        'Illinois_East': '3435',
        'IL_West': '3436',
        'IL_East': '3435',
        'Alabama East': '26929',
        'Alabama West': '26930',
        'Florida North': '2238',
        'Florida East': '2236',
        'Florida West': '2237',
        'Texas South': '2279',
        'Louisiana South': '3452',
        'Louisiana North': '3451',
        'Mississippi East': '2254',
        'Mississippi West': '2255',
        # Add more as needed
    }

    epsg = projection_dict.get(projection_name)
    if epsg is None:
        logging.warning(f"Unknown projection: {projection_name}")
        return None
    return epsg


def load_reach_table(reach_table_path):
    """Load reach table with projection information"""
    logger = logging.getLogger()

    if not reach_table_path or not os.path.exists(reach_table_path):
        logger.warning("Reach table not found - will attempt to infer CRS from data")
        return None

    try:
        reach_table = pd.read_csv(reach_table_path)
        logger.info(f"Loaded reach table with {len(reach_table)} entries")
        logger.info(f"Columns: {list(reach_table.columns)}")
        return reach_table
    except Exception as e:
        logger.error(f"Failed to load reach table: {e}")
        return None


def get_reach_crs(reach_name, reach_table):
    """Get CRS for a specific reach from the reach table"""
    logger = logging.getLogger()

    if reach_table is None:
        return None

    try:
        # Try different possible column names
        possible_id_cols = ['Reach_ID', 'reach_id', 'ReachID', 'reach_name', 'name']
        possible_proj_cols = ['Projection', 'projection', 'CRS', 'crs', 'EPSG']

        id_col = next((col for col in possible_id_cols if col in reach_table.columns), None)
        proj_col = next((col for col in possible_proj_cols if col in reach_table.columns), None)

        if id_col is None or proj_col is None:
            logger.warning(f"Could not find required columns in reach table")
            return None

        # Look up the reach
        matches = reach_table[reach_table[id_col] == reach_name]
        if len(matches) == 0:
            logger.warning(f"Reach {reach_name} not found in reach table")
            return None

        projection_name = matches[proj_col].iloc[0]
        epsg_code = get_epsg_code(projection_name)

        if epsg_code:
            logger.info(f"Found projection for {reach_name}: {projection_name} (EPSG:{epsg_code})")
            return CRS.from_epsg(int(epsg_code))

        return None

    except Exception as e:
        logger.error(f"Error looking up CRS for {reach_name}: {e}")
        return None


def detect_coordinate_system(ds, reach_name, reach_table):
    """
    Detect coordinate system using reach table first, then fallback methods

    Returns: (CRS object, detection_method_string)
    """
    logger = logging.getLogger()

    # Method 1: Try reach table lookup
    reach_crs = get_reach_crs(reach_name, reach_table)
    if reach_crs is not None:
        return reach_crs, "reach_table"

    # Method 2: Analyze coordinate ranges
    lats = ds['latitudes'].values
    lons = ds['longitudes'].values

    # Filter for finite values only
    valid_mask = np.isfinite(lats) & np.isfinite(lons)
    if valid_mask.sum() == 0:
        logger.error("No valid coordinates found")
        return None, "error"

    valid_lats = lats[valid_mask]
    valid_lons = lons[valid_mask]

    lat_range = valid_lats.max() - valid_lats.min()
    lon_range = valid_lons.max() - valid_lons.min()

    logger.info(f"Coordinate analysis for {reach_name}:")
    logger.info(f"  Latitude range: {valid_lats.min():.1f} to {valid_lats.max():.1f} (span: {lat_range:.1f})")
    logger.info(f"  Longitude range: {valid_lons.min():.1f} to {valid_lons.max():.1f} (span: {lon_range:.1f})")

    # Check if coordinates are geographic (WGS84)
    if (valid_lats.min() >= -90 and valid_lats.max() <= 90 and
            valid_lons.min() >= -180 and valid_lons.max() <= 360 and
            lat_range < 10 and lon_range < 10):
        logger.info("Detected geographic coordinates (WGS84)")
        return CRS.from_epsg(4326), "geographic_detection"

    # Check if coordinates are projected (State Plane)
    elif abs(valid_lats.min()) > 100 or abs(valid_lons.min()) > 100:
        # Try to guess based on coordinate ranges
        if (valid_lons.min() > 300000 and valid_lons.max() < 500000 and
                valid_lats.min() > 300000 and valid_lats.max() < 500000):
            logger.info("Guessing Illinois East State Plane based on coordinate ranges")
            return CRS.from_epsg(3435), "range_detection"
        elif (valid_lons.min() > 200000 and valid_lons.max() < 400000 and
              valid_lats.min() > 300000 and valid_lats.max() < 500000):
            logger.info("Guessing Illinois West State Plane based on coordinate ranges")
            return CRS.from_epsg(3436), "range_detection"
        else:
            logger.warning("Detected projected coordinates but couldn't determine specific system")
            return CRS.from_epsg(3435), "default_state_plane"

    # Default fallback
    logger.warning("Could not determine coordinate system, defaulting to WGS84")
    return CRS.from_epsg(4326), "fallback"


def clip_netcdf_for_csat(nc_file, boundary, output_dir, reach_table):
    """
    Clip NetCDF file to boundary while maintaining CSAT compatibility

    This function preserves:
    - All NetCDF attributes (especially valid_range and _FillValue)
    - Original data structure
    - Fill value conventions
    - Coordinate system information
    """
    logger = logging.getLogger()

    try:
        reach_name = Path(nc_file).stem
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {reach_name}")
        logger.info(f"{'=' * 60}")

        # Open NetCDF file
        ds = xr.open_dataset(nc_file)

        # Detect coordinate system
        source_crs, detection_method = detect_coordinate_system(ds, reach_name, reach_table)
        if source_crs is None:
            logger.error(f"Could not determine CRS for {reach_name}")
            return None, None

        logger.info(f"Using CRS: {source_crs} (detected via: {detection_method})")

        # Get coordinates (ALL of them, including potential fill values)
        lats = ds['latitudes'].values
        lons = ds['longitudes'].values

        logger.info(f"Total points in file: {len(lats)}")

        # Identify valid coordinates (finite values only)
        coord_valid_mask = np.isfinite(lats) & np.isfinite(lons)
        n_valid_coords = coord_valid_mask.sum()
        logger.info(f"Points with valid coordinates: {n_valid_coords} ({n_valid_coords / len(lats) * 100:.1f}%)")

        if n_valid_coords == 0:
            logger.error("No valid coordinates found")
            return None, None

        # Transform boundary to match NetCDF coordinate system
        logger.info(f"Transforming boundary from {boundary.crs} to {source_crs}")
        boundary_reproj = boundary.to_crs(source_crs)

        # Dissolve to single polygon if multiple features
        if len(boundary_reproj) > 1:
            boundary_reproj = boundary_reproj.dissolve()

        boundary_union = boundary_reproj.geometry.unary_union

        # Prepare geometry for faster intersection tests
        from shapely import prepare
        prepare(boundary_union)

        # Perform spatial filtering in batches
        logger.info("Performing spatial intersection test...")
        within_boundary_mask = np.zeros(len(lats), dtype=bool)

        # Only test points with valid coordinates
        valid_indices = np.where(coord_valid_mask)[0]
        batch_size = 10000

        for i in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[i:min(i + batch_size, len(valid_indices))]
            batch_lons = lons[batch_indices]
            batch_lats = lats[batch_indices]

            # Create points and test intersection
            batch_points = [Point(lon, lat) for lon, lat in zip(batch_lons, batch_lats)]
            batch_mask = np.array([boundary_union.intersects(pt) for pt in batch_points])
            within_boundary_mask[batch_indices] = batch_mask

            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"  Processed {min(i + batch_size, len(valid_indices))}/{len(valid_indices)} points")

        within_count = within_boundary_mask.sum()
        logger.info(
            f"Points within boundary: {within_count} / {n_valid_coords} ({within_count / n_valid_coords * 100:.1f}%)")

        if within_count == 0:
            logger.warning(f"No points from {reach_name} fall within boundary")
            extent = (lons[coord_valid_mask].min(), lats[coord_valid_mask].min(),
                      lons[coord_valid_mask].max(), lats[coord_valid_mask].max())
            create_diagnostic_plot(nc_file, boundary_reproj, extent, output_dir, source_crs, within_count)
            return None, extent

        # Get indices of points to keep
        clipped_indices = np.where(within_boundary_mask)[0]

        # Create output NetCDF file
        output_file = os.path.join(output_dir, f"{reach_name}.nc")
        logger.info(f"Writing clipped NetCDF to: {output_file}")

        with nc.Dataset(nc_file, 'r') as src, nc.Dataset(output_file, 'w') as dst:

            # Copy ALL global attributes
            dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})

            # Add clipping metadata
            dst.setncattr('clipping_applied', 'True')
            dst.setncattr('clipping_date', str(np.datetime64('now')))
            dst.setncattr('original_file', os.path.basename(nc_file))
            dst.setncattr('original_point_count', str(len(lats)))
            dst.setncattr('clipped_point_count', str(within_count))
            dst.setncattr('source_crs', str(source_crs))
            dst.setncattr('detection_method', detection_method)

            # Copy dimensions
            for name, dimension in src.dimensions.items():
                if name == 'points':
                    dst.createDimension(name, within_count)
                else:
                    dst.createDimension(name, len(dimension))

            # Copy variables with ALL attributes preserved
            for name, variable in src.variables.items():
                try:
                    # Create variable with same type and dimensions
                    x = dst.createVariable(name, variable.datatype, variable.dimensions)

                    # CRITICAL: Copy ALL attributes including valid_range and _FillValue
                    # These are essential for CSAT compatibility
                    dst[name].setncatts({a: variable.getncattr(a) for a in variable.ncattrs()})

                    # Copy data
                    if 'points' in variable.dimensions:
                        # This variable has a points dimension - clip it
                        if len(variable.dimensions) == 1:
                            # 1D variable (e.g., latitudes, longitudes)
                            data = variable[:]
                            dst[name][:] = data[clipped_indices]
                        else:
                            # 2D variable (e.g., elevations: time x points or points x time)
                            data = variable[:]
                            if variable.dimensions[0] == 'points':
                                dst[name][:] = data[clipped_indices, :]
                            else:
                                dst[name][:] = data[:, clipped_indices]
                    else:
                        # Variable without points dimension - copy as-is
                        dst[name][:] = variable[:]

                except Exception as e:
                    logger.warning(f"Warning copying variable {name}: {e}")

        logger.info(f"Successfully clipped {reach_name}")

        # Calculate extent for plotting
        extent = (lons[clipped_indices].min(), lats[clipped_indices].min(),
                  lons[clipped_indices].max(), lats[clipped_indices].max())

        ds.close()
        return output_file, extent

    except Exception as e:
        logger.error(f"Error clipping {nc_file}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if 'ds' in locals():
            ds.close()
        return None, None


def create_diagnostic_plot(nc_file, boundary_reproj, extent, output_dir, source_crs, within_count):
    """Create diagnostic plot showing the clipping result - similar to original script"""
    logger = logging.getLogger()

    try:
        reach_name = Path(nc_file).stem

        if within_count == 0:
            # No overlap - single plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            # Load original data
            ds_orig = xr.open_dataset(nc_file)
            lats = ds_orig['latitudes'].values
            lons = ds_orig['longitudes'].values

            # Filter to valid coordinates only
            valid_mask = np.isfinite(lats) & np.isfinite(lons)
            valid_lats = lats[valid_mask]
            valid_lons = lons[valid_mask]

            # Plot survey points
            ax.scatter(valid_lons, valid_lats, c='blue', s=1, alpha=0.5, label='Survey Points')

            # Plot boundary
            boundary_reproj.boundary.plot(ax=ax, color='red', linewidth=2, alpha=0.8)

            ax.set_title(f'No Overlap: {reach_name}\nCRS: {source_crs}')
            ax.set_xlabel(f'X ({source_crs})')
            ax.set_ylabel(f'Y ({source_crs})')
            ax.grid(True, alpha=0.3)

            # Add legend
            blue_patch = mpatches.Patch(color='blue', label='Survey Points')
            red_patch = mpatches.Patch(color='red', label='Boundary (No Overlap)')
            ax.legend(handles=[blue_patch, red_patch])

            # Zoom to data extent with margin
            minx = valid_lons.min()
            maxx = valid_lons.max()
            miny = valid_lats.min()
            maxy = valid_lats.max()
            margin_x = (maxx - minx) * 0.1
            margin_y = (maxy - miny) * 0.1

            ax.set_xlim(minx - margin_x, maxx + margin_x)
            ax.set_ylim(miny - margin_y, maxy + margin_y)

            plt.tight_layout()

            # Save plot
            plot_dir = os.path.join(output_dir, 'diagnostic_plots')
            os.makedirs(plot_dir, exist_ok=True)
            plot_name = f"{reach_name}_no_overlap.png"
            plot_path = os.path.join(plot_dir, plot_name)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"No-overlap plot saved: {plot_path}")
            ds_orig.close()

        else:
            # Successful clipping - comparison plot (before/after)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Load datasets
            ds_orig = xr.open_dataset(nc_file)
            orig_lats = ds_orig['latitudes'].values
            orig_lons = ds_orig['longitudes'].values

            # Filter to valid coordinates
            orig_valid_mask = np.isfinite(orig_lats) & np.isfinite(orig_lons)
            orig_lats_valid = orig_lats[orig_valid_mask]
            orig_lons_valid = orig_lons[orig_valid_mask]

            # Load clipped data
            clipped_file = os.path.join(output_dir, f"{reach_name}.nc")
            ds_clip = xr.open_dataset(clipped_file)
            clip_lats = ds_clip['latitudes'].values
            clip_lons = ds_clip['longitudes'].values

            # Plot 1: Original data
            ax1.scatter(orig_lons_valid, orig_lats_valid, c='blue', s=1, alpha=0.5, label='Original Points')
            boundary_reproj.boundary.plot(ax=ax1, color='red', linewidth=2, alpha=0.8)
            ax1.set_title(f'Original: {reach_name}\n({len(orig_lats_valid)} points)')
            ax1.set_xlabel(f'X ({source_crs})')
            ax1.set_ylabel(f'Y ({source_crs})')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Clipped data
            ax2.scatter(clip_lons, clip_lats, c='green', s=1, alpha=0.7, label='Clipped Points')
            boundary_reproj.boundary.plot(ax=ax2, color='red', linewidth=2, alpha=0.8)
            ax2.set_title(f'Clipped: {reach_name}\n({len(clip_lats)} points)')
            ax2.set_xlabel(f'X ({source_crs})')
            ax2.set_ylabel(f'Y ({source_crs})')
            ax2.grid(True, alpha=0.3)

            # Zoom both plots to clipped extent with margin
            minx = clip_lons.min()
            maxx = clip_lons.max()
            miny = clip_lats.min()
            maxy = clip_lats.max()
            margin_x = (maxx - minx) * 0.1  # 10% padding
            margin_y = (maxy - miny) * 0.1

            xlim = (minx - margin_x, maxx + margin_x)
            ylim = (miny - margin_y, maxy + margin_y)

            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)

            # Add legend
            blue_patch = mpatches.Patch(color='blue', label='Original Points')
            green_patch = mpatches.Patch(color='green', label='Clipped Points')
            red_patch = mpatches.Patch(color='red', label='Boundary')
            fig.legend(handles=[blue_patch, green_patch, red_patch],
                       loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)

            plt.tight_layout()

            # Save plot
            plot_dir = os.path.join(output_dir, 'diagnostic_plots')
            os.makedirs(plot_dir, exist_ok=True)
            plot_name = f"{reach_name}_comparison.png"
            plot_path = os.path.join(plot_dir, plot_name)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Comparison plot saved: {plot_path}")

            ds_orig.close()
            ds_clip.close()

    except Exception as e:
        logger.error(f"Error creating diagnostic plot: {e}")
        import traceback
        logger.error(traceback.format_exc())
        plt.close()


def process_directory(nc_dir, shp_file, output_dir, reach_table_path=None):
    """Main processing function"""

    # Setup
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)

    logger.info("=" * 60)
    logger.info("CSAT-Compatible NetCDF Clipper")
    logger.info("=" * 60)
    logger.info(f"Input directory: {nc_dir}")
    logger.info(f"Shapefile: {shp_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Reach table: {reach_table_path}")
    logger.info("=" * 60)

    # Load boundary shapefile
    logger.info("\nLoading boundary shapefile...")
    boundary = gpd.read_file(shp_file)
    logger.info(f"Boundary CRS: {boundary.crs}")
    logger.info(f"Boundary bounds: {boundary.total_bounds}")
    logger.info(f"Number of features: {len(boundary)}")

    if len(boundary) > 1:
        logger.info("Multiple boundary features found, dissolving to single polygon")
        boundary = boundary.dissolve()

    # Fix invalid geometries
    if not all(boundary.geometry.is_valid):
        logger.info("Fixing invalid geometries in boundary...")
        boundary['geometry'] = boundary.geometry.buffer(0)

    # Load reach table
    reach_table = load_reach_table(reach_table_path)

    # Find NetCDF files
    nc_files = sorted(list(Path(nc_dir).glob('*.nc')))
    logger.info(f"\nFound {len(nc_files)} NetCDF files to process")

    if len(nc_files) == 0:
        logger.error(f"No NetCDF files found in {nc_dir}")
        return

    # Process files
    successful = 0
    no_overlap = 0
    failed = 0
    summary_data = []

    for nc_file in tqdm(nc_files, desc="Processing NetCDF files"):
        try:
            reach_name = nc_file.stem

            # Clip file
            clipped_file, extent = clip_netcdf_for_csat(
                str(nc_file), boundary, output_dir, reach_table
            )

            if clipped_file:
                successful += 1

                # Create diagnostic plot
                try:
                    with xr.open_dataset(str(nc_file)) as ds_orig:
                        source_crs, _ = detect_coordinate_system(ds_orig, reach_name, reach_table)

                    with xr.open_dataset(clipped_file) as ds_clip:
                        within_count = len(ds_clip['latitudes'])

                    boundary_reproj = boundary.to_crs(source_crs)
                    create_diagnostic_plot(str(nc_file), boundary_reproj, extent,
                                           output_dir, source_crs, within_count)
                except Exception as e:
                    logger.warning(f"Could not create plot for {reach_name}: {e}")

                # Add to summary
                with xr.open_dataset(str(nc_file)) as ds_orig:
                    with xr.open_dataset(clipped_file) as ds_clip:
                        total_points = len(ds_orig['latitudes'])
                        clipped_points = len(ds_clip['latitudes'])
                        summary_data.append({
                            'Reach_Name': reach_name,
                            'Original_Points': total_points,
                            'Clipped_Points': clipped_points,
                            'Percent_Kept': round((clipped_points / total_points * 100), 2),
                            'Status': 'Success'
                        })
            else:
                no_overlap += 1
                try:
                    with xr.open_dataset(str(nc_file)) as ds_orig:
                        total_points = len(ds_orig['latitudes'])
                except:
                    total_points = 0

                summary_data.append({
                    'Reach_Name': reach_name,
                    'Original_Points': total_points,
                    'Clipped_Points': 0,
                    'Percent_Kept': 0,
                    'Status': 'No Overlap'
                })

        except Exception as e:
            logger.error(f"Failed to process {nc_file.name}: {e}")
            failed += 1
            summary_data.append({
                'Reach_Name': nc_file.stem,
                'Original_Points': 0,
                'Clipped_Points': 0,
                'Percent_Kept': 0,
                'Status': f'Failed: {str(e)[:50]}'
            })

    # Save summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, 'clipping_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"\nSummary saved: {summary_file}")

    # Copy reach table if it exists
    if reach_table_path and os.path.exists(reach_table_path):
        try:
            import shutil
            shutil.copy2(reach_table_path, output_dir)
            logger.info("Copied reach table to output directory")
        except Exception as e:
            logger.warning(f"Failed to copy reach table: {e}")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Successfully clipped: {successful} files")
    logger.info(f"No overlap found: {no_overlap} files")
    logger.info(f"Failed: {failed} files")
    logger.info(f"Total files: {len(nc_files)}")
    logger.info("=" * 60)

    logger.info("\nClipping complete! Output files are CSAT-compatible.")
    logger.info(f"You can now run CSAT on the clipped NetCDF files in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Clip NetCDF files to shapefile boundary (CSAT-compatible)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python csat_clipper.py --nc_dir ./data/CEMVR --shp_file ./boundary.shp --output_dir ./clipped
  python csat_clipper.py --nc_dir ./data/CEMVR --shp_file ./boundary.shp --output_dir ./clipped --reach_table ./reach_table.txt
        """
    )
    parser.add_argument('--nc_dir', required=True, help='Directory containing .nc files')
    parser.add_argument('--shp_file', required=True, help='Path to boundary shapefile')
    parser.add_argument('--output_dir', required=True, help='Output directory for clipped files')
    parser.add_argument('--reach_table', help='Path to reach table with CRS info (optional)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.nc_dir):
        print(f"Error: NetCDF directory does not exist: {args.nc_dir}")
        sys.exit(1)

    if not os.path.isfile(args.shp_file):
        print(f"Error: Shapefile does not exist: {args.shp_file}")
        sys.exit(1)

    try:
        process_directory(args.nc_dir, args.shp_file, args.output_dir, args.reach_table)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_with_cemvr_paths():
    """Run with CEMVR paths - convenience function"""
    nc_folder = "C:/workspace/CSAT/CSAT_distribution/data/CEMVR"
    shapefile_path = "C:/workspace/CSAT/CSAT_distribution/data/testshp/MVR_AOI.shp"
    output_folder = "C:/workspace/CSAT/CSAT_distribution/data/Clipped"
    reach_table_path = "C:/workspace/CSAT/CSAT_distribution/data/CEMVR/reach_table.txt"

    try:
        process_directory(nc_folder, shapefile_path, output_folder, reach_table_path)
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line arguments provided
        main()
    else:
        # No arguments, use hardcoded CEMVR paths
        print("Running clipper with CEMVR paths...")
        success = run_with_cemvr_paths()
        if success:
            print("\nClipping completed successfully!")
            print("Clipped files are ready for CSAT processing.")
        else:
            print("\nClipping failed. Check the log for details.")