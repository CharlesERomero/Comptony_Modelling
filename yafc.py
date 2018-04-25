import astropy.units as u
import scipy.interpolate as spint
import astropy.constants as const
import scipy.constants as spconst
import numpy as np
import get_data_info as gdi
from astropy.coordinates import Angle #

def get_sz_values():
    
    ########################################################
    ### Astronomical value...
    tcmb = 2.72548*u.K # Kelvin (uncertainty = 0.00057)
    ### Reference:
    ### http://iopscience.iop.org/article/10.1088/0004-637X/707/2/916/meta
    
    ### Standard physical values.
    thom_cross = (spconst.value("Thomson cross section") *u.m**2).to("cm**2")
    m_e_c2 = (const.m_e *const.c**2).to("keV")
    kpctocm = 3.0856776 *10**21
    boltzmann = spconst.value("Boltzmann constant in eV/K")/1000.0 # keV/K  
    planck = spconst.value("Planck constant in eV s")/1000.0 # keV s
    c = const.c
    keVtoJ = (u.keV).to("J") # I think I need this...) 
    Icmb = 2.0 * (boltzmann*tcmb.value)**3 / (planck*c.value)**2
    Icmb *= keVtoJ*u.W *u.m**-2*u.Hz**-1*u.sr**-1 # I_{CMB} in W m^-2 Hz^-1 sr^-1
    JyConv = (u.Jy).to("W * m**-2 Hz**-1")
    Jycmb = Icmb.to("Jy sr**-1")  # I_{CMB} in Jy sr^-1
    MJycmb= Jycmb.to("MJy sr**-1")

    ### The following constants (and conversions) are just the values (in Python):
    sz_cons_values={"thom_cross":thom_cross.value,"m_e_c2":m_e_c2.value,
                    "kpctocm":kpctocm,"boltzmann":boltzmann,
                    "planck":planck,"tcmb":tcmb.value,"c":c.value,}
    ### The following "constants" have units attached (in Python)!
    sz_cons_units={"Icmb":Icmb,"Jycmb":Jycmb,"thom_cross":thom_cross,
                   "m_e_c2":m_e_c2}

    return sz_cons_values, sz_cons_units

def get_underlying_vars():

    ### Some cluster-dependent variables:
    rxj1347_priors = priors()
    m500   = rxj1347_priors.M500 * u.M_sun
    z      = rxj1347_priors.z
    #racen  = rxj1347_priors.ra.to('deg')
    #deccen = rxj1347_priors.dec.to('deg')
    ### Some fitting variables:
    beamvolume=120.0 # in arcsec^2
    radminmax = np.array([9.0,4.25*60.0])*(u.arcsec).to('rad')
    nbins     = 6    # It's just a good number...so good, you could call it a perfect number.

    ##############
    bins      = np.logspace(np.log10(radminmax[0]),np.log10(radminmax[1]), nbins) 
    #geom     = [X_shift, Y_shift, Rotation, Ella*, Ellb*, Ellc*, Xi*, Opening Angle]
    geom      = [0,0,0,1,1,1,0,0] # This gives spherical geometry
    map_vars  = gdi.get_map_vars(rxj1347_priors, instrument='MUSTANG2')
    alphas    = np.zeros(nbins) #??
    d_ang     = gdi.get_d_ang(z)
    #binskpc   = bins * d_ang
    sz_vars,szcu = get_sz_values()
    sz_vars   = gdi.get_SZ_vars(temp=rxj1347_priors.Tx)
    Pdl2y     = (szcu['thom_cross']*d_ang/szcu['m_e_c2']).to("cm**3 keV**-1")

    return sz_vars, map_vars, bins, Pdl2y, geom

class priors:
        
    def __init__(self):
        
        ###############################################################################
        ### Prior known values regarding the RXJ1053. Redshift, ra, and dec *MUST* be
        ### known / accurate. M_500 and Tx are useful for creating initial guesses.
        ### Tx is still important if relativistic corrections may be severe.
        
        self.z=0.4510                      # Redshift
        self.ra = Angle('13h47m30.5s')     # Right Ascencion, in hours
        self.dec= Angle('-11d45m9s')       # Declination, in degrees
        self.M500 = 2.2e15                 # Solar masses
        self.Tx    = 10.8                  # keV
        self.name  = 'rxj1347'
        
        ###  For when the time comes to use the *actual* coordinates for Abell 2146,
        ###  Here they are. Even now, it's useful to calculate the offsets of the centroids
        ###  for the radius of curvature of the shocks.
