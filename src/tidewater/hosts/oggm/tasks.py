"""The tasks of the scheme in one namespace, as ``oggm.tasks`` holds OGGM's."""
# flake8: noqa
from tidewater.hosts.oggm.forcing import process_ocean_data
from tidewater.hosts.oggm.forcing import process_destine_ocean_data
from tidewater.hosts.oggm.forcing import write_subglacial_discharge
from tidewater.hosts.oggm.shop_bed import bedmachine_bed_to_gdir
from tidewater.hosts.oggm.shop_bed import bedmachine_bed_statistics
from tidewater.hosts.oggm.shop_bed import compile_bedmachine_bed_statistics
from tidewater.hosts.oggm.inversion import terminus_water_depth_from_bed
from tidewater.hosts.oggm.inversion import find_inversion_calving
from tidewater.hosts.oggm.inversion import find_inversion_calving_from_bathymetry
from tidewater.hosts.oggm.inversion import set_inversion_k_from_ocean
from tidewater.hosts.oggm.inversion import set_inversion_k_from_model
from tidewater.hosts.oggm.inversion import calving_vs_geodetic_residual
from tidewater.hosts.oggm.bed import bedmachine_calving_extension
from tidewater.hosts.oggm.bed import bedmachine_terminus_bed
from tidewater.hosts.oggm.bed import calving_front_width_check
from tidewater.hosts.oggm.bed import bed_extension_statistics
from tidewater.hosts.oggm.bed import calving_vs_bed_extension
from tidewater.hosts.oggm.bed import compile_bed_extension_statistics
from tidewater.hosts.oggm.bed import compile_calving_front_width_check
from tidewater.hosts.oggm.bed import compile_calving_vs_bed_extension
from tidewater.hosts.oggm.laws import run_with_ocean_forcing
from tidewater.hosts.oggm.laws import compile_frontal_components
from tidewater.hosts.oggm.calibration import run_dynamic_calving_k_calibration
from tidewater.hosts.oggm.calibration import calving_k_for_glaciers_without_target
